"""M4 Paged Batch Generate：用 PagedKVCache 替换 fixed-slot cache 的 batch generation。

与 M3 batch_core.py 的对应关系：
- batch_generate     → batch_generate_paged
- max_num_slots      → num_blocks × block_size
- BatchedKVCache     → PagedKVCache
- allocate_slot      → allocate_request
- free_slot          → free_request
- cache_slots        → request_ids
- cache_positions    → block_tables[rid].seq_len
- kv_cache           → paged_kv_cache

T7-A6 改造：
  - 真实模型使用 PagedCacheAdapter + ForwardContext
  - FakeModel 走旧参数路径兼容

主循环结构不变：admit → prefill → decode → finish/free。
"""

import time

import torch

from inferlite.cache import PagedKVCache
from inferlite.cache.adapter import PagedCacheAdapter
from inferlite.config import ModelConfig
from inferlite.engine.forward_context import set_forward_context
from inferlite.engine.metrics import MetricsCollector
from inferlite.engine.protocol import LLMModel
from inferlite.sampler.greedy import GreedySampler
from inferlite.scheduler.fcfs import FCFSScheduler
from inferlite.scheduler.request import RequestState, RequestStatus


def batch_generate_paged(
    model: LLMModel,
    sampler: GreedySampler,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    num_blocks: int,
    block_size: int,
    config: ModelConfig,
    eos_token_id: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    metrics: MetricsCollector | None = None,
) -> list[torch.Tensor]:
    """M4 paged continuous batching generate。

    Args:
        model: 推理模型（Qwen3ForCausalLM 或符合 LLMModel 协议的对象）。
        sampler: 采样器。
        prompts: 多个 prompt，每个 shape 为 [1, T_i]。
        max_new_tokens: 每个请求最多生成的新 token 数。
        num_blocks: 物理 block 总数。
        block_size: 每个 block 的 token 容量。
        config: 模型配置，用于创建 PagedKVCache。
        eos_token_id: EOS token id，生成到时提前停止。
        device: 计算设备。
        dtype: 数据类型。
        metrics: 可选的 MetricsCollector。

    Returns:
        每个请求的生成结果列表（按 request_id 排序）。
    """
    # ── 初始化 scheduler ──
    scheduler = FCFSScheduler(max_num_seqs=num_blocks)
    for i, prompt_ids in enumerate(prompts):
        req = RequestState(
            request_id=str(i),
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        scheduler.submit(req)
        if metrics:
            metrics.record_arrival(req.request_id)
            metrics.record_prompt_tokens(req.request_id, prompt_ids.shape[1])

    # ── 初始化 PagedKVCache + Adapter ──
    paged_cache = PagedKVCache.from_config(
        config=config,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )
    real_model = getattr(model, "model", None)
    use_adapter = real_model is not None and hasattr(real_model, "layers")
    adapter = PagedCacheAdapter(paged_cache) if use_adapter else None
    if adapter:
        adapter.bind_kv_cache(model)

    # ── 主循环 ──
    while scheduler.has_unfinished():
        # ── 1. admit + batched prefill ──
        admitted = _paged_admit(scheduler, paged_cache)
        if admitted:
            for request in admitted:
                prompt_len = request.prompt_ids.shape[1]
                paged_cache.allocate_request(request.request_id, prompt_len)
                if adapter:
                    adapter._current_request_ids.append(request.request_id)
                if metrics:
                    metrics.record_scheduled(request.request_id)
                    metrics.record_prefill_start(request.request_id)

            # 拼 batched prefill
            prompt_lens = [req.prompt_ids.shape[1] for req in admitted]
            max_prompt_len = max(prompt_lens)
            batch_input_ids = torch.zeros(
                len(admitted), max_prompt_len, dtype=torch.long, device=device
            )
            for i, req in enumerate(admitted):
                batch_input_ids[i, : req.prompt_ids.shape[1]] = req.prompt_ids.squeeze(0)

            batch_position_ids = torch.zeros(
                len(admitted), max_prompt_len, dtype=torch.long, device=device
            )
            for i, plen in enumerate(prompt_lens):
                batch_position_ids[i, :plen] = torch.arange(plen, device=device)

            batch_request_ids = [req.request_id for req in admitted]

            # prefill forward
            if adapter:
                metadata = adapter.make_prefill_metadata(
                    batch_input_ids, batch_position_ids, request_ids=batch_request_ids
                )
                with set_forward_context(metadata):
                    logits = model(batch_input_ids, positions=batch_position_ids)
            else:
                logits = model(
                    input_ids=batch_input_ids,
                    position_ids=batch_position_ids,
                    paged_kv_cache=paged_cache,
                    request_ids=batch_request_ids,
                    is_prefill=True,
                )

            # 逐请求采样
            for i, request in enumerate(admitted):
                plen = prompt_lens[i]
                last_logits = logits[i, plen - 1, :].unsqueeze(0)
                request.seq_len = plen
                request.last_token = sampler(last_logits)
                request.generated_tokens.append(request.last_token)
                request.num_generated = 1
                if metrics:
                    metrics.record_prefill_end(request.request_id)
                    metrics.record_first_token(request.request_id)

        # ── 2. batched decode ──
        if not scheduler.running:
            break
        running = list(scheduler.running.values())
        request_ids = [req.request_id for req in running]

        # append_token：为每个请求分配下一个 token 的 block 空间
        for rid in request_ids:
            paged_cache.append_token(rid)

        # position_ids：写入位置 = append 后的 seq_len - 1
        positions = torch.tensor(
            [paged_cache.block_tables[rid].seq_len - 1 for rid in request_ids],
            dtype=torch.long,
            device=device,
        )
        position_ids = positions.unsqueeze(1)

        next_tokens = torch.cat(
            [req.last_token for req in running if req.last_token is not None], dim=0
        )

        decode_start = time.perf_counter()
        if adapter:
            metadata = adapter.make_decode_metadata(next_tokens, position_ids)
            with set_forward_context(metadata):
                logits = model(next_tokens, positions=position_ids)
        else:
            logits = model(
                input_ids=next_tokens,
                position_ids=position_ids,
                paged_kv_cache=paged_cache,
                request_ids=request_ids,
                is_prefill=False,
            )
        decode_ms = (time.perf_counter() - decode_start) * 1000

        # ── 3. sample + update + finish ──
        sampled = sampler(logits[:, -1, :])
        for request, next_token in zip(running, sampled, strict=False):
            request.last_token = next_token.unsqueeze(0)
            request.generated_tokens.append(next_token.unsqueeze(0))
            request.num_generated += 1
            request.seq_len += 1

            is_max = request.num_generated >= request.max_new_tokens
            is_eos = eos_token_id is not None and next_token.item() == eos_token_id
            if is_max or is_eos:
                scheduler.mark_finished(request)
                paged_cache.free_request(request.request_id)
                if adapter:
                    adapter._current_request_ids.remove(request.request_id)
                if metrics:
                    metrics.record_output_tokens(request.request_id, request.num_generated)
                    metrics.record_finished(request.request_id)

        if metrics:
            step_idx = len(metrics.step_metrics)
            max_seq_len_step = int(positions.max().item())
            metrics.record_step(
                step_idx=step_idx,
                batch_size=len(running),
                max_seq_len=max_seq_len_step,
                decode_ms=decode_ms,
                output_tokens=len(running),
                running_count=len(scheduler.running),
                waiting_count=len(scheduler.waiting),
                occupied_slots=len(scheduler.running),
            )

    # ── 收集结果 ──
    results = []
    for req_id in sorted(scheduler.finished.keys(), key=int):
        req = scheduler.finished[req_id]
        results.append(torch.cat([req.prompt_ids] + req.generated_tokens, dim=1))
    return results


def _paged_admit(scheduler: FCFSScheduler, paged_cache: PagedKVCache) -> list[RequestState]:
    """Block-aware admission。"""
    admitted = []
    while scheduler.waiting:
        req: RequestState = scheduler.waiting[0]
        prompt_len = req.prompt_ids.shape[1]
        if not paged_cache.can_allocate(prompt_len):
            break
        scheduler.waiting.popleft()
        req.status = RequestStatus.RUNNING
        scheduler.running[req.request_id] = req
        admitted.append(req)
    return admitted
