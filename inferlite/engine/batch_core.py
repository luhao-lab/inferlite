"""M3 Continuous Batching generate。

`batch_generate()` 是 M3 的核心入口，串起 scheduler + BatchedKVCache + batched attention，
实现 continuous batching 的最小执行流。

与 M2 `generate()` 的关系：
  - M2 `generate()`：单请求，prefill + decode 两阶段
  - M3 `batch_generate()`：多请求，continuous batching

T7-A6 改造：
  - 真实模型使用 BatchedCacheAdapter + ForwardContext
  - FakeModel 走旧参数路径兼容

调用示例：
    outputs = batch_generate(
        model, sampler,
        prompts=[ids_a, ids_b, ids_c],
        max_new_tokens=16,
        max_num_slots=2,
        config=config,
        max_seq_len=512,
    )
"""

import time

import torch

from inferlite.cache import BatchedKVCache
from inferlite.cache.adapter import BatchedCacheAdapter
from inferlite.config import ModelConfig
from inferlite.engine.forward_context import set_forward_context
from inferlite.engine.metrics import MetricsCollector
from inferlite.engine.protocol import LLMModel
from inferlite.sampler.greedy import GreedySampler
from inferlite.scheduler.fcfs import FCFSScheduler
from inferlite.scheduler.request import RequestState


def batch_generate(
    model: LLMModel,
    sampler: GreedySampler,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    max_num_slots: int,
    config: ModelConfig,
    max_seq_len: int,
    eos_token_id: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    metrics: MetricsCollector | None = None,
) -> list[torch.Tensor]:
    """M3 continuous batching generate。

    Args:
        model: 推理模型（Qwen3ForCausalLM 或符合 LLMModel 协议的对象）。
        sampler: 采样器。
        prompts: 多个 prompt，每个 shape 为 [1, T_i]。
        max_new_tokens: 每个请求最多生成的新 token 数。
        max_num_slots: KV cache 的最大槽位数（= 最大并发请求数）。
        config: 模型配置，用于创建 BatchedKVCache。
        max_seq_len: 每个请求的最大序列长度。
        eos_token_id: EOS token id，生成到时提前停止。
        device: 计算设备。
        dtype: 数据类型。
        metrics: 可选的 MetricsCollector，传入则在执行过程中记录请求级和步级指标。

    Returns:
        每个请求的生成结果列表（按 request_id 排序），
        每个元素为 prompt + generated token ids，shape [1, T_i + n_i]。
    """
    # ── 初始化 scheduler ──
    scheduler = FCFSScheduler(max_num_seqs=max_num_slots)
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

    # ── 创建 BatchedKVCache + Adapter ──
    cache = BatchedKVCache.from_config(
        config=config,
        max_num_slots=max_num_slots,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )
    # 判断 model 是否支持 ForwardContext（真实模型有 .model.layers）
    real_model = getattr(model, "model", None)
    use_adapter = real_model is not None and hasattr(real_model, "layers")
    adapter = BatchedCacheAdapter(cache) if use_adapter else None
    if adapter:
        adapter.bind_kv_cache(model)

    # ── 主循环 ──
    while scheduler.has_unfinished():
        # ── 1. admit + prefill ──
        admitted = scheduler.admit_until_full()
        for request in admitted:
            if metrics:
                metrics.record_scheduled(request.request_id)

            slot = cache.allocate_slot(request.request_id)
            request.slot_id = slot
            if adapter:
                adapter._current_request_ids.append(request.request_id)

            prompt_len = request.prompt_ids.shape[1]
            position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)

            if metrics:
                metrics.record_prefill_start(request.request_id)

            # prefill forward
            if adapter:
                metadata = adapter.make_prefill_metadata(request.prompt_ids, position_ids)
                with set_forward_context(metadata):
                    logits = model(request.prompt_ids, positions=position_ids)
            else:
                logits = model(
                    request.prompt_ids,
                    position_ids=position_ids,
                    kv_cache=cache,
                    cache_slots=torch.tensor([slot]),
                )

            request.seq_len = prompt_len
            request.last_token = sampler(logits[:, -1, :])
            request.generated_tokens.append(request.last_token)
            request.num_generated = 1
            cache.seq_lens[slot] = prompt_len
            if metrics:
                metrics.record_prefill_end(request.request_id)
                metrics.record_first_token(request.request_id)

        # ── 2. batched decode ──
        if not scheduler.running:
            break
        running = list(scheduler.running.values())
        cache_slots = torch.tensor([req.slot_id for req in running])
        cache_positions = cache.seq_lens[cache_slots]  # 当前写入位置
        position_ids = cache_positions.unsqueeze(1)
        next_tokens = torch.cat(
            [req.last_token for req in running if req.last_token is not None], dim=0
        )
        # 关键：adapter 路径需要在 make_decode_metadata 前将 cache.seq_lens 设为
        # 「写入后长度」= cache_positions + 1，与 _build_metadata 中 cache_positions + 1 一致
        if adapter:
            for req in running:
                cache.seq_lens[req.slot_id] += 1

        decode_start = time.perf_counter()
        if adapter:
            metadata = adapter.make_decode_metadata(next_tokens, position_ids)
            with set_forward_context(metadata):
                logits = model(next_tokens, positions=position_ids)
        else:
            logits = model(
                next_tokens,
                position_ids=position_ids,
                kv_cache=cache,
                cache_slots=cache_slots,
                cache_positions=cache_positions,
            )
        decode_ms = (time.perf_counter() - decode_start) * 1000

        # ── 3. sample + update + finish ──
        sampled = sampler(logits[:, -1, :])
        for request, next_token in zip(running, sampled, strict=False):
            request.last_token = next_token.unsqueeze(0)
            request.generated_tokens.append(next_token.unsqueeze(0))
            request.num_generated += 1
            request.seq_len += 1
            cache.seq_lens[request.slot_id] = request.seq_len

            is_max = request.num_generated >= request.max_new_tokens
            is_eos = eos_token_id is not None and next_token.item() == eos_token_id
            if is_max or is_eos:
                scheduler.mark_finished(request)
                cache.free_slot(request.request_id)
                if adapter:
                    adapter._current_request_ids.remove(request.request_id)
                if metrics:
                    metrics.record_output_tokens(request.request_id, request.num_generated)
                    metrics.record_finished(request.request_id)

        if metrics:
            step_idx = len(metrics.step_metrics)
            max_seq_len_step = int(cache_positions.max().item()) + 1
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
