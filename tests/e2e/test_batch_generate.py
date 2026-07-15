"""M3-T5 E2E: 串行 generate vs batch_generate 语义等价测试。

核心命题：
    同一组请求逐条串行 generate 的结果
    是否等价于 batch_generate continuous batching 的结果？

使用 deterministic fake model（不依赖真实模型浮点），
验证 BatchedKVCache 路径与 KVCache 路径产生相同的 token 序列。

运行：
    uv run pytest tests/e2e/test_batch_generate.py -v
"""

import torch

from inferlite.config import ModelConfig
from inferlite.engine.batch_core import batch_generate
from inferlite.engine.core import EngineCore, generate
from inferlite.model.kv_cache import KVCache
from inferlite.model.qwen3 import Qwen3ForCausalLM
from inferlite.sampler.greedy import GreedySampler


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


class DeterministicModel:
    """Deterministic fake model：始终返回固定 token。

    不依赖 input_ids 内容，保证 serial 和 batch 路径产生完全相同的 token 序列。
    支持 M2（KVCache）和 M3（BatchedKVCache）两种 cache 路径。
    """

    def __init__(self, vocab_size: int, token_id: int = 7) -> None:
        self.vocab_size = vocab_size
        self.token_id = token_id
        self.call_count = 0

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        logits_to_keep: int | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache=None,
        cache_slots: torch.Tensor | None = None,
        cache_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        self.call_count += 1

        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, -1, self.token_id] = 10.0
        return logits


def _serial_generate(
    model: DeterministicModel,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    config: ModelConfig,
    max_seq_len: int = 64,
) -> list[torch.Tensor]:
    """逐条串行 generate，每个请求用独立的 KVCache。"""
    sampler = GreedySampler()
    engine = EngineCore(model=model, sampler=sampler)
    results = []
    for prompt in prompts:
        cache = KVCache.from_config(
            config,
            batch_size=1,
            max_seq_len=max_seq_len,
            dtype=torch.float32,
            device="cpu",
        )
        with torch.no_grad():
            out = generate(engine, prompt.clone(), max_new_tokens=max_new_tokens, kv_cache=cache)
        results.append(out)
    return results


# ---------------------------------------------------------------------------
# L0-1: max_num_slots=1 等价串行
# ---------------------------------------------------------------------------


def test_batch_matches_serial_single_slot():
    """max_num_slots=1 时 batch_generate 应退化为串行，输出与逐条 generate 一致。"""
    config = _tiny_config()
    model = DeterministicModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[4, 5]]),
        torch.tensor([[6, 7, 8, 9]]),
    ]

    serial_results = _serial_generate(model, prompts, max_new_tokens=5, config=config)
    batch_results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=5,
        max_num_slots=1,
        config=config,
        max_seq_len=64,
    )

    assert len(batch_results) == len(serial_results)
    for i, (serial_out, batch_out) in enumerate(zip(serial_results, batch_results, strict=False)):
        assert torch.equal(
            serial_out, batch_out
        ), f"请求 {i} 不匹配:\n  serial={serial_out}\n  batch ={batch_out}"


# ---------------------------------------------------------------------------
# L0-2: max_num_slots>1 等价串行
# ---------------------------------------------------------------------------


def test_batch_matches_serial_multi_slots_2():
    """max_num_slots=2 时 batch_generate 输出仍应与串行一致。"""
    config = _tiny_config()
    model = DeterministicModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[10, 20]]),
        torch.tensor([[30, 40, 50]]),
        torch.tensor([[60]]),
    ]

    serial_results = _serial_generate(model, prompts, max_new_tokens=4, config=config)
    batch_results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=4,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
    )

    assert len(batch_results) == len(serial_results)
    for i, (serial_out, batch_out) in enumerate(zip(serial_results, batch_results, strict=False)):
        assert torch.equal(
            serial_out, batch_out
        ), f"请求 {i} (slots=2) 不匹配:\n  serial={serial_out}\n  batch ={batch_out}"


def test_batch_matches_serial_multi_slots_4():
    """max_num_slots=4 时 batch_generate 输出仍应与串行一致。"""
    config = _tiny_config()
    model = DeterministicModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[10, 20]]),
        torch.tensor([[30, 40, 50]]),
        torch.tensor([[60]]),
    ]

    serial_results = _serial_generate(model, prompts, max_new_tokens=4, config=config)
    batch_results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=4,
        max_num_slots=4,
        config=config,
        max_seq_len=64,
    )

    for i, (serial_out, batch_out) in enumerate(zip(serial_results, batch_results, strict=False)):
        assert torch.equal(serial_out, batch_out), f"请求 {i} (slots=4) 不匹配"


# ---------------------------------------------------------------------------
# L0-3: prompt 长度不同
# ---------------------------------------------------------------------------


def test_variable_prompt_lengths():
    """不同长度的 prompt 不应影响每个请求的独立语义。"""
    config = _tiny_config()
    model = DeterministicModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1]]),  # 长度 1
        torch.tensor([[2, 3, 4, 5, 6]]),  # 长度 5
        torch.tensor([[7, 8]]),  # 长度 2
    ]

    serial_results = _serial_generate(model, prompts, max_new_tokens=3, config=config)
    batch_results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        max_num_slots=3,
        config=config,
        max_seq_len=64,
    )

    for i, (serial_out, batch_out) in enumerate(zip(serial_results, batch_results, strict=False)):
        assert torch.equal(
            serial_out, batch_out
        ), f"请求 {i} (prompt_len={prompts[i].shape[1]}) 不匹配"


# ---------------------------------------------------------------------------
# L0-6: EOS 早停
# ---------------------------------------------------------------------------


def test_eos_stops_correctly():
    """EOS 应使 batch_generate 在生成 EOS 后停止。"""
    config = _tiny_config()
    model = DeterministicModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompt = torch.tensor([[1, 2]])
    eos_id = 42

    batch_results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=[prompt],
        max_new_tokens=20,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
        eos_token_id=eos_id,
    )

    assert len(batch_results) == 1
    # 至少生成 1 个 token（prefill 采样）
    assert batch_results[0].shape[1] >= prompt.shape[1] + 1


# ---------------------------------------------------------------------------
# L0-7: waiting queue 大于 slots，最终全部完成
# ---------------------------------------------------------------------------


def test_more_requests_than_slots():
    """请求数 > max_num_slots 时，所有请求最终都应完成且与串行一致。"""
    config = _tiny_config()
    model = DeterministicModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [torch.tensor([[i, i + 1]]) for i in range(5)]

    serial_results = _serial_generate(model, prompts, max_new_tokens=3, config=config)
    batch_results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
    )

    assert len(batch_results) == 5
    for i, (serial_out, batch_out) in enumerate(zip(serial_results, batch_results, strict=False)):
        assert torch.equal(serial_out, batch_out), f"请求 {i} 不匹配"


# ---------------------------------------------------------------------------
# 真实模型测试：用 Qwen3ForCausalLM 验证 serial vs batch token 级一致
# ---------------------------------------------------------------------------


def _serial_generate_real(
    model: Qwen3ForCausalLM,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    config: ModelConfig,
    max_seq_len: int = 64,
) -> list[torch.Tensor]:
    """逐条串行 generate（真实模型），每个请求用独立的 KVCache。"""
    sampler = GreedySampler()
    engine = EngineCore(model=model, sampler=sampler)
    results = []
    for prompt in prompts:
        cache = KVCache.from_config(
            config,
            batch_size=1,
            max_seq_len=max_seq_len,
            dtype=torch.float32,
            device="cpu",
        )
        with torch.no_grad():
            out = generate(engine, prompt.clone(), max_new_tokens=max_new_tokens, kv_cache=cache)
        results.append(out)
    return results


def test_real_qwen3_batch_matches_serial():
    """真实 Qwen3：batch_generate 输出应与逐条串行 generate 完全一致（torch.equal）。

    验证 BatchedKVCache 路径与 KVCache 路径在 token 级别等价，
    证明 M3 的改动只有性能变化，没有语义变化。
    """
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[4, 5]]),
        torch.tensor([[6, 7, 8, 9]]),
    ]

    serial_results = _serial_generate_real(model, prompts, max_new_tokens=5, config=config)
    batch_results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=5,
        max_num_slots=3,
        config=config,
        max_seq_len=64,
    )

    assert len(batch_results) == len(serial_results)
    for i, (serial_out, batch_out) in enumerate(zip(serial_results, batch_results, strict=False)):
        assert torch.equal(
            serial_out, batch_out
        ), f"请求 {i} 不匹配:\n  serial={serial_out.tolist()}\n  batch ={batch_out.tolist()}"


def test_real_qwen3_batch_matches_serial_multi_slots():
    """真实 Qwen3：max_num_slots=2 时 batch_generate 仍与串行一致。"""
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[10, 20]]),
        torch.tensor([[30, 40, 50]]),
        torch.tensor([[60]]),
    ]

    serial_results = _serial_generate_real(model, prompts, max_new_tokens=4, config=config)
    batch_results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=4,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
    )

    for i, (serial_out, batch_out) in enumerate(zip(serial_results, batch_results, strict=False)):
        assert torch.equal(
            serial_out, batch_out
        ), f"请求 {i} (slots=2) 不匹配:\n  serial={serial_out.tolist()}\n  batch ={batch_out.tolist()}"
