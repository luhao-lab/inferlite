"""M7-T4 MTPProposer 单元测试。

8 个测试用例覆盖：
1. K=0（num_draft_tokens=0）
2. K=1（num_draft_tokens=1）
3. K=3（num_draft_tokens=3，全量 heads）
4. 空 context 边界
5. draft_probs shape + sum
6. num_draft_tokens > len(heads) 截断
7. last_draft_probs 更新
8. 参数验证
"""

import pytest
import torch
import torch.nn as nn

from inferlite.spec.mtp_proposer import MTPProposer


class MockTargetModel(nn.Module):
    """Mock target model，提供 .model() 和 .lm_head。

    model() 返回固定的 hidden states。
    lm_head 是真实的 nn.Linear，用于测试 lm_head 投影。
    """

    def __init__(self, hidden_size: int = 16, vocab_size: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._model = _MockBackbone(hidden_size)

    @property
    def model(self):
        return self._model


class _MockBackbone(nn.Module):
    """模拟 Qwen3Model，forward 返回固定的 hidden states。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        """返回 [batch, seq_len, hidden_size] 的固定 hidden states。"""
        batch_size, seq_len = input_ids.shape
        hidden = torch.zeros(batch_size, seq_len, self.hidden_size)
        for j in range(seq_len):
            hidden[0, j, j % self.hidden_size] = 1.0
        return hidden


class MockMTPHead(nn.Module):
    """Mock MTP head，返回可控的输出。

    每个 head 用不同的 output_token_idx，验证 head 独立性。
    """

    def __init__(self, hidden_size: int, output_token_idx: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_token_idx = output_token_idx
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, h_t):
        """返回固定的 h_next，让 lm_head 映射后指定 token 概率最高。"""
        h_next = torch.zeros_like(h_t)
        h_next[..., self.output_token_idx % self.hidden_size] = 5.0
        return h_next


class TestMTPProposer:
    """MTPProposer 的测试用例。"""

    def _make_proposer(self, hidden_size=16, vocab_size=10, num_heads=3, k=3, temperature=1.0):
        """构造 MTPProposer 及其依赖的 mock 对象。"""
        model = MockTargetModel(hidden_size=hidden_size, vocab_size=vocab_size)
        # 每个 head 对应不同的 token（验证独立性）
        heads = [MockMTPHead(hidden_size=hidden_size, output_token_idx=i) for i in range(num_heads)]
        proposer = MTPProposer(
            target_model=model,
            heads=heads,
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

    def test_3_k_three(self):
        """测试 3：K=3（num_draft_tokens=3，全量 heads）。

        预期：返回 3 个 token，last_draft_probs 长度为 3。
        每个 head 预测不同的 token（因为 mock head 输出不同）。
        """
        proposer = self._make_proposer(num_heads=3, k=3)
        draft = proposer.propose([1, 2, 3])

        assert len(draft) == 3
        assert all(isinstance(t, int) for t in draft)
        assert len(proposer.last_draft_probs) == 3

    def test_4_empty_context(self):
        """测试 4：空 context 边界。

        预期：返回空列表，last_draft_probs 也是空列表。
        """
        proposer = self._make_proposer(k=3)
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

    def test_6_num_draft_tokens_truncation(self):
        """测试 6：num_draft_tokens > len(heads) 时截断。

        预期：draft 数量不超过 heads 数量。
        """
        # 只有 2 个 head，但要求 draft 5 个
        proposer = self._make_proposer(num_heads=2, k=5)

        # num_draft_tokens 应该被截断为 2
        assert proposer.num_draft_tokens == 2

        draft = proposer.propose([1, 2, 3])
        assert len(draft) == 2
        assert len(proposer.last_draft_probs) == 2

    def test_7_last_draft_probs_update(self):
        """测试 7：last_draft_probs 更新。

        预期：每次 propose 后 last_draft_probs 被更新为新的 list 对象。
        """
        proposer = self._make_proposer(k=2, num_heads=3)

        proposer.propose([1, 2, 3])
        probs1 = proposer.last_draft_probs

        proposer.propose([4, 5, 6, 7])
        probs2 = proposer.last_draft_probs

        assert probs1 is not probs2
        assert len(probs2) == 2

    def test_8_parameter_validation(self):
        """测试 8：参数验证。

        确保 num_draft_tokens >= 0, temperature > 0, heads 不为空。
        """
        model = MockTargetModel()
        heads = [MockMTPHead(hidden_size=16, output_token_idx=0)]

        with pytest.raises(AssertionError):
            MTPProposer(target_model=model, heads=heads, num_draft_tokens=-1)

        with pytest.raises(AssertionError):
            MTPProposer(target_model=model, heads=heads, num_draft_tokens=3, temperature=0)

        with pytest.raises(AssertionError):
            MTPProposer(target_model=model, heads=heads, num_draft_tokens=3, temperature=-0.5)

        with pytest.raises(AssertionError):
            MTPProposer(target_model=model, heads=[], num_draft_tokens=3)


class TestMTPHeadIndependence:
    """验证 MTP 多 head 独立性（T4 核心特性）。"""

    def test_heads_independent_from_same_ht(self):
        """所有 head 从同一个 h_t 预测，互不影响。

        用 3 个不同的 mock head，每个输出不同的 hidden state。
        验证每个 head 的输出独立——改变 head_0 不影响 head_1/2 的 probs。
        """
        hidden_size = 16
        vocab_size = 10
        model = MockTargetModel(hidden_size=hidden_size, vocab_size=vocab_size)

        # head_0 → token 0, head_1 → token 1, head_2 → token 2
        heads = [MockMTPHead(hidden_size=hidden_size, output_token_idx=i) for i in range(3)]

        proposer = MTPProposer(target_model=model, heads=heads, num_draft_tokens=3)
        torch.manual_seed(42)
        proposer.propose([1, 2, 3])
        probs1 = [p.clone() for p in proposer.last_draft_probs]

        # 重新构造，head_0 换成输出不同的 head
        heads_v2 = [
            MockMTPHead(hidden_size=hidden_size, output_token_idx=9),  # 改 head_0
            MockMTPHead(hidden_size=hidden_size, output_token_idx=1),  # head_1 不变
            MockMTPHead(hidden_size=hidden_size, output_token_idx=2),  # head_2 不变
        ]

        proposer_v2 = MTPProposer(target_model=model, heads=heads_v2, num_draft_tokens=3)
        torch.manual_seed(42)
        proposer_v2.propose([1, 2, 3])
        probs2 = [p.clone() for p in proposer_v2.last_draft_probs]

        # head_1 和 head_2 的 probs 应该不受 head_0 变化影响（精确相等）
        assert torch.allclose(probs1[1], probs2[1]), "head_1 的 probs 不应受 head_0 变化影响"
        assert torch.allclose(probs1[2], probs2[2]), "head_2 的 probs 不应受 head_0 变化影响"
        # head_0 的 probs 应该不同（因为 mock 输出不同）
        assert not torch.allclose(probs1[0], probs2[0]), "head_0 改变后 probs 应该不同"
