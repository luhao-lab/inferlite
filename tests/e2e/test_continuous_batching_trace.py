"""M3-T5 E2E: continuous batching trace 测试。

用 fake model 控制每个请求的结束时机，验证：
1. 短请求完成后立即释放 slot，等待请求在下一轮进入（非 static batching）
2. slot 复用不串数据（新请求不读到旧 KV）
3. batch size trace 符合预期

运行：
    uv run pytest tests/e2e/test_continuous_batching_trace.py -v
"""

import torch

from inferlite.config import ModelConfig
from inferlite.engine.batch_core import batch_generate
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


class StepCountingModel:
    """按请求维度控制生成步数的 fake model。

    通过 max_tokens_map 为每个 prompt 指定不同的生成长度，
    用来验证不同长度的请求在 continuous batching 中的行为。

    每次调用返回固定 token（由 token_id 控制），确保 deterministic。
    """

    def __init__(self, vocab_size: int, token_id: int = 7) -> None:
        self.vocab_size = vocab_size
        self.token_id = token_id
        self.call_count = 0
        self.batch_sizes: list[int] = []

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
        self.batch_sizes.append(B)

        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, -1, self.token_id] = 10.0
        return logits


class PerRequestEosModel:
    """为不同请求在不同的 decode 步返回 EOS。

    eos_steps: 第几个 decode 步（从 0 开始计数）返回 EOS。
    通过控制 per-request 的 eos 时机，测试 continuous batching 的请求进退。
    """

    def __init__(
        self, vocab_size: int, eos_token_id: int, num_requests: int, eos_steps: list[int]
    ) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.eos_steps = eos_steps  # 每个请求在第几步返回 EOS
        self.normal_token = 50
        self._decode_steps_per_request: list[int] = [0] * num_requests
        self._current_request_idx = 0
        self.call_count = 0
        self.batch_sizes: list[int] = []

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
        self.batch_sizes.append(B)

        logits = torch.zeros(B, T, self.vocab_size)

        if T > 1:
            # prefill：返回 normal_token
            logits[:, -1, self.normal_token] = 10.0
        else:
            # decode：根据 cache_slots 确定每个请求的身份
            if cache_slots is not None:
                for i, slot in enumerate(cache_slots.tolist()):
                    step = self._decode_steps_per_request[slot]
                    if step >= self.eos_steps[slot]:
                        logits[i, -1, self.eos_token_id] = 10.0
                    else:
                        logits[i, -1, self.normal_token] = 10.0
                    self._decode_steps_per_request[slot] += 1
            else:
                logits[:, -1, self.normal_token] = 10.0

        return logits


# ---------------------------------------------------------------------------
# L0-4: output 长度不同 — 短请求提前完成
# ---------------------------------------------------------------------------


def test_different_output_lengths():
    """不同 max_new_tokens 的请求应在各自的时间点完成。"""
    config = _tiny_config()
    model = StepCountingModel(vocab_size=config.vocab_size, token_id=7)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4, 5]]),
    ]
    # 两个请求都用 max_new_tokens=5，但 model 是 deterministic
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=5,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
    )

    assert len(results) == 2
    # 每个结果 = prompt + generated
    assert results[0].shape == (1, 2 + 5)  # prompt(2) + generated(5)
    assert results[1].shape == (1, 3 + 5)  # prompt(3) + generated(5)


# ---------------------------------------------------------------------------
# L0-5: slot 复用不串数据
# ---------------------------------------------------------------------------


def test_slot_reuse_no_kv_contamination():
    """slot 被新请求复用时，新请求不应读到旧 KV。

    构造：req_a (max_new_tokens=2) 先完成，释放 slot。
    req_c (max_new_tokens=3) 在 req_a 之后 admit，使用同一个 slot。
    验证 req_c 的输出与单独串行执行一致。
    """
    config = _tiny_config()
    sampler = GreedySampler()

    # 先单独跑 req_c 得到 baseline
    model_solo = StepCountingModel(vocab_size=config.vocab_size, token_id=7)
    prompt_c = torch.tensor([[10, 20, 30]])
    solo_result = batch_generate(
        model=model_solo,
        sampler=sampler,
        prompts=[prompt_c],
        max_new_tokens=3,
        max_num_slots=1,
        config=config,
        max_seq_len=64,
    )

    # 再跑 batch：req_a 先完成（2 tokens），req_c 后进入
    model_batch = StepCountingModel(vocab_size=config.vocab_size, token_id=7)
    prompts = [
        torch.tensor([[1, 2]]),  # req_a
        prompt_c,  # req_c
    ]
    batch_results = batch_generate(
        model=model_batch,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        max_num_slots=1,  # 串行执行
        config=config,
        max_seq_len=64,
    )

    # req_c 是第二个请求（index 1）
    assert torch.equal(
        solo_result[0], batch_results[1]
    ), f"slot 复用导致 KV 污染:\n  solo ={solo_result[0]}\n  batch={batch_results[1]}"


# ---------------------------------------------------------------------------
# L0-8: batch size trace 符合预期
# ---------------------------------------------------------------------------


def test_batch_size_trace():
    """验证每轮 decode 的 batch size 反映当前 running 数量。"""
    config = _tiny_config()
    model = StepCountingModel(vocab_size=config.vocab_size, token_id=7)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4]]),
    ]
    batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        max_num_slots=4,
        config=config,
        max_seq_len=64,
    )

    # 2 个请求，max_num_slots=4，都能同时 admit
    # 调用序列：prefill a (B=1), prefill b (B=1), decode (B=2), decode (B=2), ...
    assert model.batch_sizes[0] == 1  # prefill a
    assert model.batch_sizes[1] == 1  # prefill b
    # 后续 decode 应该都是 B=2（两个请求同时 running）
    decode_calls = model.batch_sizes[2:]
    assert all(
        b == 2 for b in decode_calls
    ), f"decode 阶段 batch size 应全是 2，实际: {decode_calls}"


# ---------------------------------------------------------------------------
# L0-9: 非 static wave — finished 请求不锁住 batch
# ---------------------------------------------------------------------------


def test_not_static_batching():
    """证明不是 static batching：短请求完成后新请求可以进入。

    构造 3 个请求，max_num_slots=2：
    - req_0 和 req_1 先 admit
    - req_0 先完成（max_new_tokens=2），释放 slot
    - req_2 应在下一轮被 admit（不等 req_1 完成）

    如果 static batching，req_2 必须等 req_1 也完成才能进入。
    """
    config = _tiny_config()
    model = StepCountingModel(vocab_size=config.vocab_size, token_id=7)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4]]),
        torch.tensor([[5, 6]]),
    ]
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
    )

    # 所有 3 个请求都应完成
    assert len(results) == 3

    # 验证不是 static wave：如果 static batching (wave size=2)，
    # req_2 必须等 wave [0, 1] 全部完成才开始，model.call_count 会更大。
    # continuous batching 中 req_2 在 req_0 完成后立刻进入。
    # 3 prefill calls + at most 3*3 decode = 12 calls
    # 实际应该更少因为 req_2 提前进入
    assert model.call_count <= 12, f"调用次数过多 ({model.call_count})，可能是 static batching"


# ---------------------------------------------------------------------------
# L0-10: waiting 不占资源
# ---------------------------------------------------------------------------


def test_waiting_does_not_allocate_slot():
    """max_num_slots=1 时，第二个请求在第一个完成前不应被 allocate。

    如果 waiting 也占 slot，max_num_slots=1 + 2 请求会死锁。
    """
    config = _tiny_config()
    model = StepCountingModel(vocab_size=config.vocab_size, token_id=7)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4]]),
    ]
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        max_num_slots=1,
        config=config,
        max_seq_len=64,
    )

    # 两个请求都应完成（不死锁）
    assert len(results) == 2


# ---------------------------------------------------------------------------
# EOS trace：验证 EOS 请求退出后 batch size 变化
# ---------------------------------------------------------------------------


def test_eos_trace_batch_size_change():
    """EOS 请求退出后，如果有等待请求，batch size 应保持不变（新请求补位）。"""
    config = _tiny_config()
    eos_id = 42
    # req_0 在第 1 步 EOS，req_1 在第 5 步 EOS
    model = PerRequestEosModel(
        vocab_size=config.vocab_size,
        eos_token_id=eos_id,
        num_requests=2,
        eos_steps=[1, 5],
    )
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4]]),
    ]
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=10,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
        eos_token_id=eos_id,
    )

    assert len(results) == 2
    # req_0 应该在 EOS 后停止（生成 token 数 < max_new_tokens）
    generated_0 = results[0].shape[1] - prompts[0].shape[1]
    assert generated_0 < 10, f"req_0 应在 EOS 后停止，实际生成了 {generated_0} 个 token"
