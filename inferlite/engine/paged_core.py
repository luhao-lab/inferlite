"""M4 Paged Batch Generate（薄包装）。

真实模型委托 loop.py；FakeModel 委托 conftest.legacy_batch_generate_paged()。
"""

import torch

from inferlite.cache import PagedKVCache
from inferlite.cache.adapter import PagedCacheAdapter
from inferlite.config import ModelConfig
from inferlite.engine.loop import batch_generate_loop
from inferlite.engine.metrics import MetricsCollector
from inferlite.engine.protocol import LLMModel
from inferlite.sampler.greedy import GreedySampler
from inferlite.scheduler.fcfs import FCFSScheduler
from inferlite.scheduler.request import RequestState


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
