"""M3/M4 batched generate 入口（薄包装）。

真实模型委托 loop.py；FakeModel 委托 tests/legacy.py。
"""

import torch

from inferlite.cache import BatchedKVCache, PagedKVCache
from inferlite.cache.adapter import BatchedCacheAdapter, PagedCacheAdapter
from inferlite.config import ModelConfig
from inferlite.engine.context import LLMModel
from inferlite.engine.loop import batch_generate_loop
from inferlite.engine.metrics import MetricsCollector
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
    """M3 batched generate 入口。"""
    has_layers = hasattr(model, "model") and hasattr(model.model, "layers")

    if has_layers:
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
            scheduler.submit(
                RequestState(
                    request_id=str(i),
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                )
            )
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

    from legacy import legacy_batch_generate

    return legacy_batch_generate(
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
    """M4 paged continuous batching generate。"""
    has_layers = hasattr(model, "model") and hasattr(model.model, "layers")

    if has_layers:
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
            scheduler.submit(
                RequestState(
                    request_id=str(i),
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                )
            )
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

    from legacy import legacy_batch_generate_paged

    return legacy_batch_generate_paged(
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
