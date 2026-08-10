"""M3 Continuous Batching generate（薄包装）。

真实模型委托 loop.py；FakeModel 委托 conftest.legacy_batch_generate()。
"""

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
    """M3 batched generate 入口。"""
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
