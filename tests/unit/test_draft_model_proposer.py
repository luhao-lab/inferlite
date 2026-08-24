"""M7-T2 DraftModelProposer 单元测试。

8 个测试用例覆盖：
1. K=0（num_draft_tokens=0）
2. K=1（num_draft_tokens=1）
3. K=5（num_draft_tokens=5）
4. 空 context 边界
5. draft_probs shape
6. draft_probs 数值（softmax 后和为 1）
7. last_draft_probs 更新
8. 温度参数影响
"""

import pytest
import torch

from inferlite.spec.draft_model_proposer import DraftModelProposer


class MockDraftModel:
    """Mock 模型，返回可控的 logits。

    用于测试 DraftModelProposer 的行为，不需要真实模型。
    """

    def __init__(self, vocab_size: int = 10, fixed_logits: torch.Tensor | None = None):
        """
        Args:
            vocab_size: 词表大小
            fixed_logits: 固定的 logits（如果提供，每次 forward 返回这个）
                         shape: [1, 1, vocab_size]
        """
        self.vocab_size = vocab_size
        self.fixed_logits = fixed_logits

    def __call__(self, input_ids):
        """Mock forward，返回可控的 logits。

        Args:
            input_ids: [batch_size, seq_len] 的 token ids

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size = len(input_ids)
        seq_len = len(input_ids[0])

        if self.fixed_logits is not None:
            # 使用固定 logits（重复以匹配 batch 和 seq）
            return self.fixed_logits.expand(batch_size, seq_len, -1)
        else:
            # 默认：均匀分布 logits
            return torch.zeros(batch_size, seq_len, self.vocab_size)


class TestDraftModelProposer:
    """DraftModelProposer 的测试用例。"""

    def test_1_k_zero(self):
        """测试 1：K=0（num_draft_tokens=0）。

        预期：返回空列表，last_draft_probs 也是空列表。
        """
        model = MockDraftModel(vocab_size=10)
        proposer = DraftModelProposer(draft_model=model, num_draft_tokens=0)

        context = [1, 2, 3]
        draft = proposer.propose(context)

        assert draft == []
        assert proposer.last_draft_probs == []

    def test_2_k_one(self):
        """测试 2：K=1（num_draft_tokens=1）。

        预期：返回 1 个 token，last_draft_probs 长度为 1。
        """
        # 构造 logits：token 5 的概率最高
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 5] = 10.0  # token 5 的 logit 很高
        model = MockDraftModel(vocab_size=10, fixed_logits=logits)

        proposer = DraftModelProposer(draft_model=model, num_draft_tokens=1, temperature=1.0)
        context = [1, 2, 3]
        draft = proposer.propose(context)

        assert len(draft) == 1
        assert draft[0] == 5  # 应该选中 token 5（概率最高）
        assert len(proposer.last_draft_probs) == 1
        assert proposer.last_draft_probs[0].shape == (10,)

    def test_3_k_five(self):
        """测试 3：K=5（num_draft_tokens=5）。

        预期：返回 5 个 token，last_draft_probs 长度为 5。
        """
        # 构造 logits：每个位置都选 token 7
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 7] = 10.0
        model = MockDraftModel(vocab_size=10, fixed_logits=logits)

        proposer = DraftModelProposer(draft_model=model, num_draft_tokens=5, temperature=1.0)
        context = [1, 2, 3]
        draft = proposer.propose(context)

        assert len(draft) == 5
        assert all(t == 7 for t in draft)  # 每次都应该选中 token 7
        assert len(proposer.last_draft_probs) == 5

    def test_4_empty_context(self):
        """测试 4：空 context 边界。

        预期：返回空列表，last_draft_probs 也是空列表。
        """
        model = MockDraftModel(vocab_size=10)
        proposer = DraftModelProposer(draft_model=model, num_draft_tokens=5)

        draft = proposer.propose([])

        assert draft == []
        assert proposer.last_draft_probs == []

    def test_5_draft_probs_shape(self):
        """测试 5：draft_probs shape。

        预期：每个 prob 是 [vocab_size] tensor。
        """
        vocab_size = 15
        model = MockDraftModel(vocab_size=vocab_size)
        proposer = DraftModelProposer(draft_model=model, num_draft_tokens=3)

        context = [1, 2, 3]
        proposer.propose(context)

        assert len(proposer.last_draft_probs) == 3
        for prob in proposer.last_draft_probs:
            assert prob.shape == (vocab_size,)

    def test_6_draft_probs_sum_to_one(self):
        """测试 6：draft_probs 数值（softmax 后和为 1）。

        预期：每个 prob 的和 ≈ 1.0。
        """
        model = MockDraftModel(vocab_size=10)
        proposer = DraftModelProposer(draft_model=model, num_draft_tokens=3)

        context = [1, 2, 3]
        proposer.propose(context)

        for prob in proposer.last_draft_probs:
            assert torch.allclose(prob.sum(), torch.tensor(1.0), atol=1e-5)

    def test_7_last_draft_probs_update(self):
        """测试 7：last_draft_probs 更新。

        预期：每次 propose 后 last_draft_probs 被正确更新。
        """
        model = MockDraftModel(vocab_size=10)
        proposer = DraftModelProposer(draft_model=model, num_draft_tokens=2)

        # 第一次 propose
        context1 = [1, 2, 3]
        proposer.propose(context1)
        probs1 = proposer.last_draft_probs

        # 第二次 propose
        context2 = [4, 5, 6, 7]
        proposer.propose(context2)
        probs2 = proposer.last_draft_probs

        # last_draft_probs 应该被更新（不同的 list 对象）
        assert probs1 is not probs2
        assert len(probs2) == 2

    def test_8_temperature_effect(self):
        """测试 8：温度参数影响。

        temperature < 1 时分布更集中（选中高概率 token 的概率更高）
        temperature > 1 时分布更分散

        这里测试 temperature 参数的正确应用，不测试采样随机性。
        """
        # 构造 logits：token 0 概率 0.7，token 1 概率 0.3（temperature=1 时）
        logits = torch.zeros(1, 1, 2)
        logits[0, 0, 0] = 0.85  # log(0.7/0.3) ≈ 0.85
        logits[0, 0, 1] = 0.0
        model = MockDraftModel(vocab_size=2, fixed_logits=logits)

        # temperature=1.0：softmax([0.85, 0]) ≈ [0.7, 0.3]
        proposer = DraftModelProposer(draft_model=model, num_draft_tokens=1, temperature=1.0)
        proposer.propose([1])
        prob_t1 = proposer.last_draft_probs[0]
        assert torch.allclose(prob_t1[0], torch.tensor(0.7), atol=0.05)

        # temperature=0.1：softmax([8.5, 0]) ≈ [0.9998, 0.0002]（更集中）
        proposer = DraftModelProposer(draft_model=model, num_draft_tokens=1, temperature=0.1)
        proposer.propose([1])
        prob_t01 = proposer.last_draft_probs[0]
        assert prob_t01[0] > 0.99  # 更集中

    def test_parameter_validation(self):
        """参数验证测试。

        确保 num_draft_tokens >= 0, temperature > 0
        """
        model = MockDraftModel(vocab_size=10)

        # num_draft_tokens < 0 应该报错
        with pytest.raises(AssertionError):
            DraftModelProposer(draft_model=model, num_draft_tokens=-1)

        # temperature <= 0 应该报错
        with pytest.raises(AssertionError):
            DraftModelProposer(draft_model=model, num_draft_tokens=5, temperature=0)

        with pytest.raises(AssertionError):
            DraftModelProposer(draft_model=model, num_draft_tokens=5, temperature=-0.5)
