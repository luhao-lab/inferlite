"""Unit tests for T9 LLMModel Protocol and GreedySampler.

T9 只做 tensor 级别的 next-token 基础组件：
- LLMModel: 约定模型可以 `model(input_ids) -> logits [B, T, V]`
- GreedySampler: 约定采样器可以 `sampler(logits[:, -1, :]) -> next_token [B, 1]`

运行：
  uv run pytest tests/unit/test_sampler.py -q
"""

import torch

from inferlite.engine.protocol import LLMModel
from inferlite.sampler import GreedySampler


class FakeModel:
    """满足 LLMModel Protocol 的最小 fake model。"""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        assert self.logits.shape[:2] == (batch_size, seq_len)
        return self.logits


def _run_model(model: LLMModel, input_ids: torch.Tensor) -> torch.Tensor:
    """测试辅助函数：只依赖 LLMModel 协议，不依赖具体模型类。"""
    return model(input_ids)


def test_greedy_sampler_returns_argmax_keepdim():
    """单条样本时，GreedySampler 应返回最大 logit 的下标，并保持 [B, 1]。"""
    sampler = GreedySampler()
    logits = torch.tensor([[0.1, 0.9, 0.2]])

    next_token = sampler(logits)

    assert torch.equal(next_token, torch.tensor([[1]]))
    assert next_token.shape == (1, 1)
    assert next_token.dtype == torch.long


def test_greedy_sampler_handles_batch_independently():
    """batch 场景下，每一行 logits 都应独立 argmax。"""
    sampler = GreedySampler()
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [3.0, 1.0, 2.0],
            [-1.0, -0.5, -0.1],
        ]
    )

    next_token = sampler(logits)

    assert torch.equal(next_token, torch.tensor([[1], [0], [2]]))
    assert next_token.shape == (3, 1)


def test_llm_model_protocol_accepts_fake_model():
    """FakeModel 不继承任何基类，只要能 __call__(input_ids)->logits 就满足 LLMModel。"""
    input_ids = torch.tensor([[1, 2]])
    logits = torch.zeros(1, 2, 5)
    logits[0, 1, 3] = 10.0
    model = FakeModel(logits)

    output = _run_model(model, input_ids)

    assert torch.equal(output, logits)


def test_model_and_greedy_sampler_single_step_composition_uses_last_position():
    """组合 model + sampler 时，应取 logits 的最后一个 token 位置。

    这里不实现独立 next_token helper，只在测试中模拟 T10 EngineCore.step 的核心逻辑：
    logits = model(input_ids)
    next_token_logits = logits[:, -1, :]
    next_token = sampler(next_token_logits)
    """
    input_ids = torch.tensor([[1, 2]])
    logits = torch.tensor(
        [
            [
                [100.0, 0.0, 0.0],
                [0.0, 0.0, 100.0],
            ]
        ]
    )
    model = FakeModel(logits)
    sampler = GreedySampler()

    output_logits = _run_model(model, input_ids)
    next_token_logits = output_logits[:, -1, :]
    next_token = sampler(next_token_logits)

    # 如果错误地用了第 0 个位置，会选 0；正确使用最后位置应选 2。
    assert torch.equal(next_token, torch.tensor([[2]]))
