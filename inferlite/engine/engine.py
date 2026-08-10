"""推理引擎：M1/M2 单请求 + M3/M4 batched generate 入口。

EngineCore: 最小单步推理引擎（M1 路径）
generate(): M1（无 cache）/ M2（prefill/decode 两阶段）
batch_generate(): M3 continuous batching（委托 loop.py）
batch_generate_paged(): M4 paged attention（委托 loop.py）
"""

import torch

from inferlite.cache import BatchedKVCache, PagedKVCache
from inferlite.cache.adapter import BatchedCacheAdapter, PagedCacheAdapter
from inferlite.cache.kv_cache import KVCache
from inferlite.config import ModelConfig
from inferlite.engine.context import LLMModel, set_forward_context
from inferlite.engine.loop import batch_generate_loop
from inferlite.engine.metrics import MetricsCollector
from inferlite.sampler.greedy import GreedySampler
from inferlite.scheduler.fcfs import FCFSScheduler
from inferlite.scheduler.request import RequestState

# ── M1/M2: EngineCore + generate ──


class EngineCore:
    """最小单步推理引擎。

    EngineCore 只依赖 LLMModel 协议，不绑定具体模型类。
    只要一个对象能 model(input_ids) -> logits，就可以被 EngineCore 使用。
    """

    def __init__(self, model: LLMModel, sampler: GreedySampler) -> None:
        self.model: LLMModel = model
        self.sampler: GreedySampler = sampler

    def step(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行一步 greedy decode（M1 路径）。"""
        logits = self.model(input_ids, logits_to_keep=1)
        return self.sampler(logits[:, -1, :])


def generate(
    engine: EngineCore,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    kv_cache: KVCache | None = None,
) -> torch.Tensor:
    """M1/M2 greedy generate loop，支持 EOS 提前停止。

    Args:
        engine: 已经组装好 model + sampler 的单步推理引擎。
        input_ids: prompt token ids [B, T]。
        max_new_tokens: 最多生成多少个新 token。
        eos_token_id: EOS token id，None 时不检查。
        kv_cache: None → M1 full forward；非 None → M2 prefill/decode 两阶段。

    Returns:
        output_ids: [B, T + n]，n <= max_new_tokens。
    """
    if kv_cache is None:
        # M1 路径：每步 full forward
        for _ in range(max_new_tokens):
            next_token = engine.step(input_ids)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return input_ids

    # M2 路径：prefill/decode 两阶段
    real_model = getattr(engine.model, "model", None)
    has_layers = real_model is not None and hasattr(real_model, "layers")

    if has_layers:
        from inferlite.cache.adapter import SingleCacheAdapter

        adapter = SingleCacheAdapter(kv_cache)
        adapter.bind_kv_cache(engine.model)
        kv_cache.reset()

        T_p = input_ids.shape[1]
        position_ids = torch.arange(T_p, device=input_ids.device).unsqueeze(0)
        metadata = adapter.make_prefill_metadata(input_ids, position_ids)
        with set_forward_context(metadata):
            logits = engine.model(input_ids, positions=position_ids)
        adapter.cur_len = T_p
        kv_cache.cur_len = T_p
        next_token = engine.sampler(logits[:, -1, :])
        input_ids = torch.cat([input_ids, next_token], dim=1)

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            pos = torch.tensor([[kv_cache.cur_len]], device=input_ids.device)
            metadata = adapter.make_decode_metadata(next_token, pos)
            with set_forward_context(metadata):
                logits = engine.model(next_token, positions=pos, logits_to_keep=1)
            next_token = engine.sampler(logits[:, -1, :])
            input_ids = torch.cat([input_ids, next_token], dim=1)
    else:
        # FakeModel 旧路径（兼容单测）
        kv_cache.reset()
        T_p = input_ids.shape[1]
        position_ids = torch.arange(T_p, device=input_ids.device).unsqueeze(0)
        logits = engine.model(input_ids, position_ids=position_ids, kv_cache=kv_cache)
        kv_cache.cur_len = T_p
        next_token = engine.sampler(logits[:, -1, :])
        input_ids = torch.cat([input_ids, next_token], dim=1)
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            pos = torch.tensor([[kv_cache.cur_len]], device=input_ids.device)
            logits = engine.model(next_token, position_ids=pos, kv_cache=kv_cache)
            kv_cache.cur_len += 1
            next_token = engine.sampler(logits[:, -1, :])
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids


# ── M3/M4: batched generate ──


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
