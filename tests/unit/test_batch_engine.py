"""Unit tests for M3-T4: batch_generate continuous batching。

测试目标（对应 L0 测试清单）：
1. 单请求 batch_generate 等价 M2 generate
2. 多请求输出数量等于输入请求数
3. max_num_slots=2 时 running 不超过 2
4. 短请求完成后释放 slot 可被等待请求复用
5. 每轮 batch 重新形成
6. EOS 请求提前退出
7. max_new_tokens 到达即 finished
8. waiting queue 最终清空
9. waiting 不占 KV slot
10. finished 后下一轮可 admit

运行：
  uv run pytest tests/unit/test_batch_engine.py -v
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


class FakeBatchModel:
    """满足 LLMModel Protocol（含 cache_slots/cache_positions）的 fake model。

    行为：logits[:, -1, token_id] = 10.0，其余为 0。
    token_id 由 _token_seq 控制，每次调用递增。
    """

    def __init__(self, vocab_size: int, token_seq: list[int] | None = None) -> None:
        self.vocab_size = vocab_size
        self.token_seq = token_seq or list(range(vocab_size))
        self._step = 0
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
        # 每行返回 token_seq 中对应的 token（循环使用）
        tok_id = self.token_seq[self._step % len(self.token_seq)]
        logits[:, -1, tok_id] = 10.0
        self._step += 1
        return logits


class EosBatchModel:
    """在 decode 的第 eos_after 步返回 eos_token_id 得分最高。"""

    def __init__(self, vocab_size: int, eos_token_id: int, eos_after: int) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.eos_after = eos_after
        self._step = 0

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
        logits = torch.zeros(B, T, self.vocab_size)
        if T > 1:
            # prefill：选 token 50（非 EOS）
            logits[:, -1, 50] = 10.0
        else:
            # decode：前 eos_after 步选 token 50，之后选 EOS
            if self._step < self.eos_after:
                logits[:, -1, 50] = 10.0
            else:
                logits[:, -1, self.eos_token_id] = 10.0
            self._step += 1
        return logits


# ---------------------------------------------------------------------------
# L0-1: 单请求 batch_generate 等价 M2 generate
# ---------------------------------------------------------------------------


def test_single_request_matches_m2():
    """单请求时 batch_generate 应与 M2 generate 输出一致。"""
    config = _tiny_config()
    model = FakeBatchModel(vocab_size=config.vocab_size, token_seq=[7])
    sampler = GreedySampler()

    prompt = torch.tensor([[1, 2, 3]])
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=[prompt],
        max_new_tokens=5,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
    )

    assert len(results) == 1
    # prompt [1, 2, 3] + 5 个生成的 token（全是 7）
    expected = torch.tensor([[1, 2, 3, 7, 7, 7, 7, 7]])
    assert torch.equal(results[0], expected)


# ---------------------------------------------------------------------------
# L0-2: 多请求输出数量
# ---------------------------------------------------------------------------


def test_multiple_requests_output_count():
    """输出数量应等于输入请求数。"""
    config = _tiny_config()
    model = FakeBatchModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4, 5]]),
        torch.tensor([[6]]),
    ]
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        max_num_slots=4,
        config=config,
        max_seq_len=64,
    )

    assert len(results) == 3


# ---------------------------------------------------------------------------
# L0-3: max_num_slots=2 时 running 不超过 2
# ---------------------------------------------------------------------------


def test_max_num_slots_limits_running():
    """max_num_slots=2 时，batch_generate 应能正确处理 3 个请求（排队执行）。"""
    config = _tiny_config()
    model = FakeBatchModel(vocab_size=config.vocab_size)
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

    # 3 个请求都应完成
    assert len(results) == 3
    # 每个结果 = prompt(2) + generated(3) = 5 tokens
    for r in results:
        assert r.shape[1] == 5


# ---------------------------------------------------------------------------
# L0-4: 短请求完成后释放 slot
# ---------------------------------------------------------------------------


def test_slot_reuse_after_short_request():
    """短请求完成后释放 slot，长请求能继续执行。"""
    config = _tiny_config()
    model = FakeBatchModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1]]),  # 短 prompt
        torch.tensor([[2, 3, 4]]),  # 长 prompt
    ]
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


# ---------------------------------------------------------------------------
# L0-5: 每轮 batch 重新形成
# ---------------------------------------------------------------------------


def test_batch_reformed_each_step():
    """model 每步被调用时 batch size 应反映当前 running 数量。"""
    config = _tiny_config()
    model = FakeBatchModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]
    batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        max_num_slots=4,
        config=config,
        max_seq_len=64,
    )

    # 第一次调用：prefill req 0 (B=1)
    # 第二次调用：prefill req 1 (B=1)
    # 第三次起：decode (B=2, 因为两个请求都在 running)
    # prefill 调用 B=1，decode 调用 B=2
    prefill_calls = [b for b in model.batch_sizes if b == 1]
    decode_calls = [b for b in model.batch_sizes if b == 2]
    assert len(prefill_calls) >= 2  # 至少 2 次 prefill
    assert len(decode_calls) >= 1  # 至少 1 次 batched decode


# ---------------------------------------------------------------------------
# L0-6: EOS 请求提前退出
# ---------------------------------------------------------------------------


def test_eos_early_stop():
    """EOS 应使请求提前结束，输出长度小于 max_new_tokens。"""
    config = _tiny_config()
    eos_id = 42
    model = EosBatchModel(vocab_size=config.vocab_size, eos_token_id=eos_id, eos_after=2)
    sampler = GreedySampler()

    prompt = torch.tensor([[1, 2]])
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=[prompt],
        max_new_tokens=10,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
        eos_token_id=eos_id,
    )

    assert len(results) == 1
    # prefill 产出 token 50，decode 2 步产出 token 50, 50，第 3 步产出 EOS
    # 总生成 = 3（prefill 采样 1 + decode 2 步 + EOS 1 步）= 但 EOS 也被 append
    # prompt_len=2 + generated <= 10
    assert results[0].shape[1] < 2 + 10  # 应小于 prompt + max_new_tokens


# ---------------------------------------------------------------------------
# L0-7: max_new_tokens 到达即 finished
# ---------------------------------------------------------------------------


def test_max_new_tokens_termination():
    """达到 max_new_tokens 后请求应 finished，输出长度正确。"""
    config = _tiny_config()
    model = FakeBatchModel(vocab_size=config.vocab_size, token_seq=[7])
    sampler = GreedySampler()

    prompt = torch.tensor([[1, 2, 3]])
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=[prompt],
        max_new_tokens=4,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
    )

    assert len(results) == 1
    # prompt(3) + generated(4) = 7 tokens
    # prefill 产出 1 个 token + decode 3 步产出 3 个 = 4 个 generated
    assert results[0].shape[1] == 7


# ---------------------------------------------------------------------------
# L0-8: waiting queue 最终清空
# ---------------------------------------------------------------------------


def test_waiting_queue_drained():
    """所有请求最终都应 finished（waiting 清空）。"""
    config = _tiny_config()
    model = FakeBatchModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [torch.tensor([[i, i + 1]]) for i in range(5)]
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
    )

    # 5 个请求全部完成
    assert len(results) == 5


# ---------------------------------------------------------------------------
# L0-9: waiting 不占 KV slot
# ---------------------------------------------------------------------------


def test_waiting_does_not_occupy_slot():
    """submit 后请求在 waiting 状态时不应占用 KV slot。

    通过 max_num_slots=1 + 2 个请求验证：如果 waiting 也占 slot，
    第二个请求将无法 admit（因为 slot 已满），导致死锁。
    """
    config = _tiny_config()
    model = FakeBatchModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        max_num_slots=1,  # 只有 1 个 slot
        config=config,
        max_seq_len=64,
    )

    # 两个请求都应完成（第二个等第一个 finished 后才 admit）
    assert len(results) == 2


# ---------------------------------------------------------------------------
# L0-10: finished 后下一轮可 admit
# ---------------------------------------------------------------------------


def test_finished_enables_next_admit():
    """第一个请求 finished 后释放 slot，第二个请求应能在下一轮被 admit 并执行。"""
    config = _tiny_config()
    model = FakeBatchModel(vocab_size=config.vocab_size, token_seq=[7])
    sampler = GreedySampler()

    prompts = [torch.tensor([[1]]), torch.tensor([[2]])]
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=2,
        max_num_slots=1,  # 串行执行
        config=config,
        max_seq_len=64,
    )

    assert len(results) == 2
    # 两个请求都有完整输出
    for r in results:
        assert r.shape[1] > 0
