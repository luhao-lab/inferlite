"""Unit tests for M2-T4: generate() prefill/decode two-stage with KV cache.

测试目标（对应 DoD）：
1. kv_cache=None 路径（M1）所有旧测试继续通过 —— 由 test_generate.py 覆盖，这里不重复。
2. kv_cache 路径与 kv_cache=None 路径输出完全一致（torch.equal）。
3. 有 cache 时生成长度正确（max_new_tokens）。
4. EOS 提前停止（M2 路径）。
5. decode 步 position_ids 是绝对位置（非从 0 开始）。
6. kv_cache.reset() 后可以重新 generate，结果与首次相同。

运行：
  uv run pytest tests/unit/test_generate_kv.py -v
"""

import pytest
import torch

from inferlite.config import ModelConfig
from inferlite.engine import generate
from inferlite.engine.core import EngineCore
from inferlite.model.kv_cache import KVCache
from inferlite.model.qwen3 import Qwen3ForCausalLM
from inferlite.sampler.greedy import GreedySampler

# ---------------------------------------------------------------------------
# 小模型 config
# ---------------------------------------------------------------------------


def _tiny_config(num_hidden_layers: int = 2) -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
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


# ---------------------------------------------------------------------------
# FakeModel：记录每次 __call__ 的 position_ids，用来验证 decode 步绝对位置
# ---------------------------------------------------------------------------


class PositionRecordingModel:
    """记录每次 __call__ 传入的 position_ids（仅 kv_cache 路径）。"""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.position_ids_log: list[torch.Tensor | None] = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        logits_to_keep: int | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache=None,
    ) -> torch.Tensor:
        if kv_cache is not None:
            self.position_ids_log.append(position_ids.clone() if position_ids is not None else None)
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, :, 0] = 10.0  # token 0 得分最高，greedy 始终选 0
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return logits


# ---------------------------------------------------------------------------
# EosModel：在 decode 的第 eos_after 步返回 eos_token_id 得分最高
# ---------------------------------------------------------------------------


class EosModel:
    def __init__(self, vocab_size: int, eos_token_id: int, eos_after: int) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.eos_after = eos_after
        self._decode_step = 0

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        logits_to_keep: int | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache=None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        if T > 1:
            # prefill：最后位置选 token 99（非 EOS）
            logits[:, :, 99] = 10.0
        else:
            # decode 步
            self._decode_step += 1
            if self._decode_step >= self.eos_after:
                logits[:, :, self.eos_token_id] = 10.0
            else:
                logits[:, :, 99] = 10.0
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return logits


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model_and_cache():
    """返回 (Qwen3ForCausalLM, KVCache, EngineCore)。"""
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()
    cache = KVCache.from_config(
        config, batch_size=1, max_seq_len=64, dtype=torch.float32, device="cpu"
    )
    sampler = GreedySampler()
    engine = EngineCore(model=model, sampler=sampler)
    return model, cache, engine


# ---------------------------------------------------------------------------
# 测试 1：M2 路径与 M1 路径输出一致
# ---------------------------------------------------------------------------


def test_kv_cache_output_equals_no_cache(tiny_model_and_cache):
    """有 cache 时的 generate 输出应与无 cache 完全一致（torch.equal）。"""
    model, cache, engine = tiny_model_and_cache
    input_ids = torch.randint(0, 10, (1, 4))

    with torch.no_grad():
        out_no_cache = generate(engine, input_ids.clone(), max_new_tokens=5)
        out_with_cache = generate(engine, input_ids.clone(), max_new_tokens=5, kv_cache=cache)

    assert torch.equal(
        out_no_cache, out_with_cache
    ), f"M1 输出 {out_no_cache} != M2 输出 {out_with_cache}"


# ---------------------------------------------------------------------------
# 测试 2：生成长度正确
# ---------------------------------------------------------------------------


def test_kv_cache_generates_correct_length(tiny_model_and_cache):
    """有 cache 时生成长度应为 prompt_len + max_new_tokens。"""
    _, cache, engine = tiny_model_and_cache
    prompt_len = 3
    max_new_tokens = 5
    input_ids = torch.randint(0, 10, (1, prompt_len))

    with torch.no_grad():
        output_ids = generate(engine, input_ids, max_new_tokens=max_new_tokens, kv_cache=cache)

    assert output_ids.shape == (1, prompt_len + max_new_tokens)


# ---------------------------------------------------------------------------
# 测试 3：EOS 提前停止（M2 路径）
# ---------------------------------------------------------------------------


def test_kv_cache_eos_stops_early():
    """M2 路径：生成到 EOS token 时应提前停止。"""
    vocab_size = 200
    eos_token_id = 2
    config = _tiny_config()
    # decode 第 2 步生成 EOS
    eos_model = EosModel(vocab_size=vocab_size, eos_token_id=eos_token_id, eos_after=2)
    cache = KVCache.from_config(
        config, batch_size=1, max_seq_len=64, dtype=torch.float32, device="cpu"
    )
    sampler = GreedySampler()
    engine = EngineCore(model=eos_model, sampler=sampler)

    input_ids = torch.tensor([[10, 20]])  # prompt len = 2
    with torch.no_grad():
        output_ids = generate(
            engine, input_ids, max_new_tokens=10, eos_token_id=eos_token_id, kv_cache=cache
        )

    # prefill 后采 1 个 token（99），decode step1 采 99，decode step2 采 EOS → 停止
    # prompt(2) + prefill_sample(1) + decode1(1) + decode2_eos(1) = 5
    assert output_ids.shape[1] == 5
    assert output_ids[0, -1].item() == eos_token_id


# ---------------------------------------------------------------------------
# 测试 4：decode 步 position_ids 是绝对位置
# ---------------------------------------------------------------------------


def test_decode_position_ids_are_absolute():
    """decode 步的 position_ids 应该等于 kv_cache.cur_len，而不是从 0 重新计数。"""
    vocab_size = 100
    prompt_len = 4
    max_new_tokens = 3

    config = _tiny_config()
    model = PositionRecordingModel(vocab_size=vocab_size)
    cache = KVCache.from_config(
        config, batch_size=1, max_seq_len=64, dtype=torch.float32, device="cpu"
    )
    sampler = GreedySampler()
    engine = EngineCore(model=model, sampler=sampler)

    input_ids = torch.randint(0, 10, (1, prompt_len))
    with torch.no_grad():
        generate(engine, input_ids, max_new_tokens=max_new_tokens, kv_cache=cache)

    logs = model.position_ids_log
    # prefill 1 次 + decode loop (max_new_tokens - 1) 次
    expected_calls = 1 + (max_new_tokens - 1)
    assert len(logs) == expected_calls

    # prefill position_ids 应为 [[0, 1, 2, 3]]
    expected_prefill = torch.arange(prompt_len).unsqueeze(0)
    assert torch.equal(logs[0], expected_prefill), f"prefill position_ids 不对：{logs[0]}"

    # decode 步位置应为绝对位置：prompt_len, prompt_len+1, ...
    for step, pos_tensor in enumerate(logs[1:]):
        expected_pos = prompt_len + step
        actual_pos = pos_tensor[0, 0].item()
        assert (
            actual_pos == expected_pos
        ), f"decode step {step}: 期望绝对位置 {expected_pos}，实际 {actual_pos}"


# ---------------------------------------------------------------------------
# 测试 5：reset 后重新 generate 结果一致
# ---------------------------------------------------------------------------


def test_kv_cache_reset_allows_reuse(tiny_model_and_cache):
    """同一个 cache 对象 reset 后应可以重新使用，输出与首次相同。"""
    model, cache, engine = tiny_model_and_cache
    input_ids = torch.randint(0, 10, (1, 3))

    with torch.no_grad():
        out1 = generate(engine, input_ids.clone(), max_new_tokens=4, kv_cache=cache)
        out2 = generate(engine, input_ids.clone(), max_new_tokens=4, kv_cache=cache)

    assert torch.equal(out1, out2), "cache reset 后重新 generate 应该得到相同输出"
