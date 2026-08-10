"""M4 Paged Batch Generate（薄包装）。

真实模型委托给 loop.py 统一主循环；FakeModel 保留旧路径兼容。

T7-A8 改造：
- 真实模型（有 .model.layers）：创建 PagedCacheAdapter + FCFSScheduler，
  委托 loop.batch_generate_loop() 执行
- FakeModel（无 .model.layers）：保留旧参数传递路径
"""

import time

import torch

from inferlite.cache import PagedKVCache
from inferlite.cache.adapter import PagedCacheAdapter
from inferlite.config import ModelConfig
from inferlite.engine.loop import batch_generate_loop
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

    真实模型委托 loop.py，FakeModel 走旧路径。
    """
    has_layers = hasattr(model, "model") and hasattr(model.model, "layers")

    if has_layers:
        # ── 真实模型：委托 loop.py ──
        paged_cache = PagedKVCache.from_config(
            config=config,
            num_blocks=num_blocks,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )
        adapter = PagedCacheAdapter(paged_cache)
        scheduler = FCFSScheduler(max_num_seqs=num_blocks)
        for i, prompt_ids in enumerate(prompts):
            req = RequestState(
                request_id=str(i),
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
            scheduler.submit(req)
        return batch_generate_loop(
            model,
            sampler,
            scheduler,
            adapter,
            prompts,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            device=device,
            metrics=metrics,
        )

    # ── FakeModel 旧路径（兼容单测）──
    return _legacy_batch_generate_paged(
        model,
        sampler,
        prompts,
        max_new_tokens,
        num_blocks,
        block_size,
        config,
        eos_token_id,
        device,
        dtype,
        metrics,
    )


def _legacy_batch_generate_paged(
    model,
    sampler,
    prompts,
    max_new_tokens,
    num_blocks,
    block_size,
    config,
    eos_token_id,
    device,
    dtype,
    metrics,
):
    """FakeModel 兼容的旧路径。真实模型不走这里。"""
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

    paged_cache = PagedKVCache.from_config(
        config=config,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )

    while scheduler.has_unfinished():
        # ── 1. admit + batched prefill ──
        admitted = _paged_admit(scheduler, paged_cache)
        if admitted:
            for request in admitted:
                prompt_len = request.prompt_ids.shape[1]
                paged_cache.allocate_request(request.request_id, prompt_len)
                if metrics:
                    metrics.record_scheduled(request.request_id)
                    metrics.record_prefill_start(request.request_id)

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

            logits = model(
                input_ids=batch_input_ids,
                position_ids=batch_position_ids,
                paged_kv_cache=paged_cache,
                request_ids=batch_request_ids,
                is_prefill=True,
            )

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

        for rid in request_ids:
            paged_cache.append_token(rid)

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
        logits = model(
            input_ids=next_tokens,
            position_ids=position_ids,
            paged_kv_cache=paged_cache,
            request_ids=request_ids,
            is_prefill=False,
        )
        decode_ms = (time.perf_counter() - decode_start) * 1000

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

    results = []
    for req_id in sorted(scheduler.finished.keys(), key=int):
        req = scheduler.finished[req_id]
        results.append(torch.cat([req.prompt_ids] + req.generated_tokens, dim=1))
    return results


def _paged_admit(scheduler: FCFSScheduler, paged_cache: PagedKVCache) -> list[RequestState]:
    """Block-aware admission（FakeModel 路径专用）。"""
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
