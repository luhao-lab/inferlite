"""M3 Continuous Batching generate（薄包装）。

真实模型委托给 loop.py 统一主循环；FakeModel 保留旧路径兼容。

T7-A8 改造：
- 真实模型（有 .model.layers）：创建 BatchedCacheAdapter + FCFSScheduler，
  委托 loop.batch_generate_loop() 执行
- FakeModel（无 .model.layers）：保留旧参数传递路径
"""

import time

import torch

from inferlite.cache import BatchedKVCache
from inferlite.cache.adapter import BatchedCacheAdapter
from inferlite.config import ModelConfig
from inferlite.engine.loop import batch_generate_loop
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
    """M3 batched generate 入口。

    真实模型委托 loop.py，FakeModel 走旧路径。
    """
    has_layers = hasattr(model, "model") and hasattr(model.model, "layers")

    if has_layers:
        # ── 真实模型：委托 loop.py ──
        cache = BatchedKVCache.from_config(
            config,
            max_num_slots=max_num_slots,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )
        adapter = BatchedCacheAdapter(cache)
        scheduler = FCFSScheduler(max_num_seqs=max_num_slots)
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
    return _legacy_batch_generate(
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
    )


def _legacy_batch_generate(
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
    """FakeModel 兼容的旧路径。真实模型不走这里。"""
    # 创建 BatchedKVCache 容器
    cache = BatchedKVCache.from_config(
        config,
        max_num_slots=max_num_slots,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )

    scheduler = FCFSScheduler(max_num_seqs=max_num_slots)
    # 将所有 prompt 提交到调度器
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

        # ── Admit：FCFS 调度器取请求 ──
        admitted = scheduler.admit_until_full()

        # ── 分配 slots ──
        for request in admitted:
            slot_id = cache.allocate_slot(request.request_id)
            request.slot_id = slot_id
            request.seq_len = request.prompt_ids.shape[1]
            if metrics:
                metrics.record_scheduled(request.request_id)

        # ── Prefill 新 admit 的请求 ──
        for request in admitted:
            prompt_ids = request.prompt_ids  # [1, T]
            position_ids = torch.arange(prompt_ids.shape[1], device=prompt_ids.device).unsqueeze(
                0
            )  # [1, T]

            # 旧参数路径
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
            # 取最后一个 token 的 logits → sampler [B, V] -> [B, 1]
            first_token = sampler(logits[:, -1, :])
            request.last_token = first_token  # [1, 1]
            request.generated_tokens.append(first_token)
            request.num_generated = 1

            if metrics:
                metrics.record_prefill_end(request.request_id)
                metrics.record_first_token(request.request_id)

        # ── Decode：所有 running 请求并行执行一步 ──
        running = list(scheduler.running.values())
        if not running:
            break

        cache_slots = torch.tensor([req.slot_id for req in running], device=device)
        cache_positions = cache.seq_lens[cache_slots]  # 当前写入位置
        position_ids = cache_positions.unsqueeze(1)  # [B, 1]

        # 拼接每个请求上一步的 token
        next_tokens = torch.cat(
            [req.last_token for req in running if req.last_token is not None],
            dim=0,
        )

        # 更新 seq_lens
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

        # 采样
        sampled = sampler(logits[:, -1, :])  # [B]
        for req, tok in zip(running, sampled, strict=False):
            req.last_token = tok.unsqueeze(0)  # [1] → [1, 1]
            req.generated_tokens.append(tok.unsqueeze(0))
            req.num_generated += 1
            req.seq_len += 1

            # 完成检查
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

    # ── 收集结果（按 request_id 排序）──
    return [
        torch.cat([req.prompt_ids] + req.generated_tokens, dim=1)
        for req_id in sorted(scheduler.finished, key=int)
        for req in [scheduler.finished[req_id]]
    ]
