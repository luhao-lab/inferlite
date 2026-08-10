"""Unit tests for Model + Engine integration with logits_to_keep=1.

T12-pre: 验证 LLMModel 和 Engine 的 logits_to_keep 支持。

运行：
  uv run pytest tests/unit/test_logits_to_keep.py -q
"""

import torch

from inferlite.engine.engine import EngineCore
from inferlite.sampler import GreedySampler


class FakeModel:
    """满足 LLMModel Protocol 的 fake model，支持 logits_to_keep。"""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.calls: list[tuple[int | None, int | None]] = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        assert self.logits.shape[:2] == (batch_size, seq_len)
        self.calls.append((seq_len, logits_to_keep))
        if logits_to_keep is not None:
            return self.logits[:, -logits_to_keep:, :]
        return self.logits


def test_engine_step_logits_to_keep_1():
    """EngineCore.step 传入 logits_to_keep=1，只获取最后 token logits。"""
    logits = torch.tensor(
        [
            [
                [100.0, 0.0, 0.0],
                [0.0, 0.0, 100.0],
            ]
        ]
    )
    model = FakeModel(logits)
    engine = EngineCore(model, GreedySampler())

    next_token = engine.step(torch.tensor([[1, 2]]))

    assert torch.equal(next_token, torch.tensor([[2]]))
    assert model.calls[-1][1] == 1


def test_engine_step_no_logits_to_keep_respects_default():
    """不传 logits_to_keep 时，EngineCore.step 仍使用 logits_to_keep=1。"""
    logits = torch.tensor(
        [
            [
                [0.0, 100.0, 0.0],
                [0.0, 0.0, 100.0],
            ]
        ]
    )
    model = FakeModel(logits)
    engine = EngineCore(model, GreedySampler())

    next_token = engine.step(torch.tensor([[1, 2]]))

    assert torch.equal(next_token, torch.tensor([[2]]))
    # step 内部固定 logits_to_keep=1 优化
    # calls 格式: (seq_len, logits_to_keep)
    assert model.calls[-1][1] == 1
