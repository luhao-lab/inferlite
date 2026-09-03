"""M7-T3 EagleProposer 单元测试。

7 个测试用例覆盖：
1. K=0（num_draft_tokens=0）
2. K=1（num_draft_tokens=1）
3. K=5（num_draft_tokens=5）
4. 空 context 边界
5. draft_probs shape + sum
6. last_draft_probs 更新
7. 参数验证
"""

import pytest
import torch
import torch.nn as nn

from inferlite.spec.eagle_head import EagleHead
from inferlite.spec.eagle_proposer import EagleProposer


class MockTargetModel(nn.Module):
    """Mock target model，提供 .model() 和 .lm_head。

    model() 返回固定的 hidden states。
    lm_head 是真实的 nn.Linear，用于测试 lm_head 投影。
    """

    def __init__(self, hidden_size: int = 16, vocab_size: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # 模拟 Qwen3ForCausalLM 的 lm_head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # 内部 model（Qwen3Model 部分）
        self._model = _MockBackbone(hidden_size)

    @property
    def model(self):
        return self._model


class _MockBackbone(nn.Module):
    """模拟 Qwen3Model，forward 返回固定的 hidden states。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # 需要一个 parameter 让 MockTargetModel.parameters() 能推断 device
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        """返回 [batch, seq_len, hidden_size] 的固定 hidden states。"""
        batch_size, seq_len = input_ids.shape
        # 每个位置返回不同的 hidden state（用 position 作 seed）
        hidden = torch.zeros(batch_size, seq_len, self.hidden_size)
        for j in range(seq_len):
            hidden[0, j, j % self.hidden_size] = 1.0
        return hidden


class MockEagleHead(nn.Module):
    """Mock EagleHead，返回可控的输出。

    每次 forward 返回固定的 hidden state（token 3 对应的方向）。
    """

    def __init__(self, hidden_size: int, output_token_idx: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_token_idx = output_token_idx
        # 需要至少一个 parameter
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, h_t):
        """返回固定的 h_next，让 lm_head 映射后 token output_token_idx 概率最高。"""
        h_next = torch.zeros_like(h_t)
        h_next[..., self.output_token_idx % self.hidden_size] = 5.0
        return h_next


class TestEagleProposer:
    """EagleProposer 的测试用例。"""

    def _make_proposer(self, hidden_size=16, vocab_size=10, k=5, temperature=1.0, eagle_head=None):
        """构造 EagleProposer 及其依赖的 mock 对象。"""
        model = MockTargetModel(hidden_size=hidden_size, vocab_size=vocab_size)
        if eagle_head is None:
            eagle_head = MockEagleHead(hidden_size=hidden_size)
        proposer = EagleProposer(
            target_model=model,
            eagle_head=eagle_head,
            num_draft_tokens=k,
            temperature=temperature,
        )
        return proposer

    def test_1_k_zero(self):
        """测试 1：K=0（num_draft_tokens=0）。

        预期：返回空列表，last_draft_probs 也是空列表。
        """
        proposer = self._make_proposer(k=0)
        draft = proposer.propose([1, 2, 3])

        assert draft == []
        assert proposer.last_draft_probs == []

    def test_2_k_one(self):
        """测试 2：K=1（num_draft_tokens=1）。

        预期：返回 1 个 token，last_draft_probs 长度为 1。
        """
        proposer = self._make_proposer(k=1)
        draft = proposer.propose([1, 2, 3])

        assert len(draft) == 1
        assert isinstance(draft[0], int)
        assert len(proposer.last_draft_probs) == 1

    def test_3_k_five(self):
        """测试 3：K=5（num_draft_tokens=5）。

        预期：返回 5 个 token，last_draft_probs 长度为 5。
        """
        proposer = self._make_proposer(k=5)
        draft = proposer.propose([1, 2, 3])

        assert len(draft) == 5
        assert all(isinstance(t, int) for t in draft)
        assert len(proposer.last_draft_probs) == 5

    def test_4_empty_context(self):
        """测试 4：空 context 边界。

        预期：返回空列表，last_draft_probs 也是空列表。
        """
        proposer = self._make_proposer(k=5)
        draft = proposer.propose([])

        assert draft == []
        assert proposer.last_draft_probs == []

    def test_5_draft_probs_shape_and_sum(self):
        """测试 5：draft_probs shape + sum。

        预期：每个 prob 是 [vocab_size] tensor，且和 ≈ 1.0。
        """
        vocab_size = 10
        proposer = self._make_proposer(vocab_size=vocab_size, k=3)
        proposer.propose([1, 2, 3])

        assert len(proposer.last_draft_probs) == 3
        for prob in proposer.last_draft_probs:
            assert prob.shape == (vocab_size,)
            assert torch.allclose(prob.sum(), torch.tensor(1.0), atol=1e-5)

    def test_6_last_draft_probs_update(self):
        """测试 6：last_draft_probs 更新。

        预期：每次 propose 后 last_draft_probs 被更新为新的 list 对象。
        """
        proposer = self._make_proposer(k=2)

        proposer.propose([1, 2, 3])
        probs1 = proposer.last_draft_probs

        proposer.propose([4, 5, 6, 7])
        probs2 = proposer.last_draft_probs

        assert probs1 is not probs2
        assert len(probs2) == 2

    def test_7_parameter_validation(self):
        """测试 7：参数验证。

        确保 num_draft_tokens >= 0, temperature > 0。
        """
        model = MockTargetModel()
        head = MockEagleHead(hidden_size=16)

        with pytest.raises(AssertionError):
            EagleProposer(target_model=model, eagle_head=head, num_draft_tokens=-1)

        with pytest.raises(AssertionError):
            EagleProposer(target_model=model, eagle_head=head, num_draft_tokens=5, temperature=0)

        with pytest.raises(AssertionError):
            EagleProposer(target_model=model, eagle_head=head, num_draft_tokens=5, temperature=-0.5)


class TestEagleHead:
    """EagleHead 结构测试。"""

    def test_output_shape(self):
        """输出 shape 和输入一致。"""
        head = EagleHead(hidden_size=16)
        h = torch.randn(16)
        out = head(h)
        assert out.shape == (16,)

    def test_batch_input(self):
        """支持 batch 输入。"""
        head = EagleHead(hidden_size=16)
        h = torch.randn(4, 16)
        out = head(h)
        assert out.shape == (4, 16)

    def test_parameters_registered(self):
        """参数正确注册（super().__init__() 已调用）。"""
        head = EagleHead(hidden_size=16)
        params = list(head.parameters())
        assert len(params) > 0
        # fc1: weight + bias, fc2: weight + bias = 4 parameters
        assert len(params) == 4

    def test_param_count(self):
        """参数量正确：2 * (hidden_size^2 + hidden_size)。"""
        hidden_size = 16
        head = EagleHead(hidden_size=hidden_size)
        expected = 2 * (hidden_size * hidden_size + hidden_size)
        actual = sum(p.numel() for p in head.parameters())
        assert actual == expected
