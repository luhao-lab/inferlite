"""测试共享工具：FakeModel 兼容的旧参数 generate 函数。

batch_core.py / paged_core.py 对真实模型委托 loop.py，
对 FakeModel（无 .model.layers）调用这里的 legacy 函数。
"""

import time

import torch

from inferlite.cache import BatchedKVCache, PagedKVCache
from inferlite.scheduler.fcfs import FCFSScheduler
from inferlite.scheduler.request import RequestState, RequestStatus


def legacy_batch_generate(
    model,
    sampler,
    prompts,
    max_new_tokens,
    max_num_slots,
    config,
    max_seq_len,
    eos_token_id,
    device,
    dtype,
    metrics,
):
    """FakeModel 兼容的 M3 batched generate（旧参数路径）。"""
    cache = BatchedKVCache.from_config(
        config,
        max_num_slots=max_num_slots,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )
    scheduler = FCFSScheduler(max_num_seqs=max_num_slots)
    for i, prompt_ids in enumerate(prompts):
        request = RequestState(
            request_id=str(i),
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        scheduler.submit(request)
        if metrics:
            metrics.record_arrival(request.request_id)
            metrics.record_prompt_tokens(request.request_id, prompt_ids.shape[1])

    step_idx = 0
    while scheduler.has_unfinished():
        step_start = time.perf_counter()
        admitted = scheduler.admit_until_full()

        for request in admitted:
            slot_id = cache.allocate_slot(request.request_id)
            request.slot_id = slot_id
            request.seq_len = request.prompt_ids.shape[1]
            if metrics:
                metrics.record_scheduled(request.request_id)

        for request in admitted:
            prompt_ids = request.prompt_ids
            position_ids = torch.arange(prompt_ids.shape[1], device=prompt_ids.device).unsqueeze(0)
            cache_slots = torch.tensor([request.slot_id], device=device)
            cache_positions = torch.tensor([0], device=device)
            if metrics:
                metrics.record_prefill_start(request.request_id)
            logits = model(
                prompt_ids,
                position_ids=position_ids,
                kv_cache=cache,
                cache_slots=cache_slots,
                cache_positions=cache_positions,
            )
            first_token = sampler(logits[:, -1, :])
            request.last_token = first_token
            request.generated_tokens.append(first_token)
            request.num_generated = 1
            if metrics:
                metrics.record_prefill_end(request.request_id)
                metrics.record_first_token(request.request_id)

        running = list(scheduler.running.values())
        if not running:
            break

        cache_slots = torch.tensor([req.slot_id for req in running], device=device)
        cache_positions = cache.seq_lens[cache_slots]
        position_ids = cache_positions.unsqueeze(1)
        next_tokens = torch.cat(
            [req.last_token for req in running if req.last_token is not None], dim=0
        )
        for req in running:
            cache.seq_lens[req.slot_id] += 1

        decode_start = time.perf_counter()
        logits = model(
            next_tokens,
            position_ids=position_ids,
            kv_cache=cache,
            cache_slots=cache_slots,
            cache_positions=cache_positions,
        )
        decode_time = time.perf_counter() - decode_start

        sampled = sampler(logits[:, -1, :])
        for req, tok in zip(running, sampled, strict=False):
            req.last_token = tok.unsqueeze(0)
            req.generated_tokens.append(tok.unsqueeze(0))
            req.num_generated += 1
            req.seq_len += 1
            is_done = req.num_generated >= req.max_new_tokens or (
                eos_token_id is not None and tok.item() == eos_token_id
            )
            if is_done:
                scheduler.mark_finished(req)
                cache.free_slot(req.request_id)
                if metrics:
                    metrics.record_output_tokens(req.request_id, req.num_generated)
                    metrics.record_finished(req.request_id)

        step_time = time.perf_counter() - step_start
        if metrics:
            metrics.record_step(
                step_idx=step_idx,
                num_seqs=len(running),
                step_time=step_time,
                decode_time=decode_time,
                prefill_time=0.0,
            )
        step_idx += 1

    return [
        torch.cat([req.prompt_ids] + req.generated_tokens, dim=1)
        for req_id in sorted(scheduler.finished, key=int)
        for req in [scheduler.finished[req_id]]
    ]


def _paged_admit(scheduler, paged_cache):
    """Block-aware admission（FakeModel 路径专用）。"""
    admitted = []
    while scheduler.waiting:
        req = scheduler.waiting[0]
        prompt_len = req.prompt_ids.shape[1]
        if not paged_cache.can_allocate(prompt_len):
            break
        scheduler.waiting.popleft()
        req.status = RequestStatus.RUNNING
        scheduler.running[req.request_id] = req
        admitted.append(req)
    return admitted


def legacy_batch_generate_paged(
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
    """FakeModel 兼容的 M4 paged generate（旧参数路径）。"""
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
