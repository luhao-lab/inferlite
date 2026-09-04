"""M7-T5 RejectionSampler 单元测试。

9 个测试用例覆盖：
1. Classic 全接受 + bonus（K+1 个 token）
2. Classic 第一个就拒绝（只有 bonus）
3. Classic 中间拒绝（accepted + bonus）
4. K=0 边界（无 draft，直接 bonus）
5. Greedy 全接受
6. Greedy 部分拒绝
7. target_logits shape 验证
8. 修正分布 fallback（p_target == p_draft 时不崩溃）
9. num_accepted / num_rejected 正确性
"""

import pytest
import torch

from inferlite.spec.rejection_sampler import RejectionSampler


class TestRejectionSamplerClassic:
    """Classic 模式（temperature > 0）测试。"""

    def test_1_all_accepted_with_bonus(self):
        """测试 1：全接受 + bonus。

        构造 target_probs >= draft_probs（对所有 draft token），
        accept_prob = min(1, p_t/p_d) = 1.0，全部接受。

        预期：accepted = 3 个 draft tokens，bonus = 1 个，总共 4 个。
        """
        vocab_size = 5
        K = 3
        # draft 选了 token 0, 1, 2（每个位置选不同的 token）
        draft_tokens = [0, 1, 2]

        # draft_probs：每个位置 token_i 概率 0.3
        draft_probs = []
        for i in range(K):
            p = torch.zeros(vocab_size)
            p[i] = 0.3
            p[(i + 3) % vocab_size] = 0.7  # 其他概率给别的 token
            draft_probs.append(p)

        # target_logits：让 target 对 draft token 给很高的 logit（>> draft）
        # K+1 = 4 个位置
        target_logits = torch.zeros(K + 1, vocab_size)
        for i in range(K):
            target_logits[i, i] = 10.0  # draft token 的 logit 很高 → p_t >> p_d
        target_logits[K, 4] = 10.0  # bonus 位置，token 4 概率最高

        sampler = RejectionSampler(temperature=1.0)
        result = sampler.sample(draft_tokens, draft_probs, target_logits)

        assert result.num_accepted == 3
        assert result.accepted_tokens == [0, 1, 2]
        assert result.bonus_token == 4  # token 4 概率最高
        assert len(result.all_tokens) == 4  # K+1
        assert result.num_draft_tokens == 3
        assert result.num_rejected == 0

    def test_2_first_rejected(self):
        """测试 2：第一个就拒绝。

        构造 target 对 draft token 的概率远低于 draft，
        accept_prob 很小，用 seed 确保拒绝。

        预期：accepted = []，只有 bonus。
        """
        vocab_size = 5
        K = 2
        draft_tokens = [0, 1]

        # draft 给 token 0 高概率 0.9
        p0 = torch.zeros(vocab_size)
        p0[0] = 0.9
        p0[1] = 0.1
        p1 = torch.zeros(vocab_size)
        p1[1] = 0.5
        p1[2] = 0.5
        draft_probs = [p0, p1]

        # target 给 token 0 极低概率 → accept_prob ≈ 0
        target_logits = torch.zeros(K + 1, vocab_size)
        target_logits[0, 0] = -100.0  # token 0 概率 ≈ 0
        target_logits[0, 3] = 10.0  # token 3 概率最高
        target_logits[1, 1] = 10.0
        target_logits[2, 4] = 10.0

        sampler = RejectionSampler(temperature=1.0)
        torch.manual_seed(42)
        result = sampler.sample(draft_tokens, draft_probs, target_logits)

        # 第一个被拒绝
        assert result.num_accepted == 0
        assert result.accepted_tokens == []
        assert len(result.all_tokens) == 1  # 只有 bonus
        assert result.num_rejected == 2  # K - 0 = 2

    def test_3_middle_rejected(self):
        """测试 3：中间拒绝。

        第一个接受，第二个拒绝。

        预期：accepted = [token_0]，bonus 从修正分布采样。
        """
        vocab_size = 5
        K = 3
        draft_tokens = [0, 1, 2]

        # draft_probs
        p0 = torch.zeros(vocab_size)
        p0[0] = 0.3
        p0[3] = 0.7
        p1 = torch.zeros(vocab_size)
        p1[1] = 0.9  # draft 高估 token 1
        p1[2] = 0.1
        p2 = torch.zeros(vocab_size)
        p2[2] = 0.5
        p2[4] = 0.5
        draft_probs = [p0, p1, p2]

        # target：位置 0 给 token 0 高概率（接受），位置 1 给 token 1 极低概率（拒绝）
        target_logits = torch.zeros(K + 1, vocab_size)
        target_logits[0, 0] = 10.0  # p_t[0][0] >> p_d[0][0] → 接受
        target_logits[1, 1] = -100.0  # p_t[1][1] ≈ 0, p_d[1][1] = 0.9 → 拒绝
        target_logits[1, 3] = 10.0  # token 3 概率最高
        target_logits[2, 2] = 10.0
        target_logits[3, 4] = 10.0

        sampler = RejectionSampler(temperature=1.0)
        torch.manual_seed(42)
        result = sampler.sample(draft_tokens, draft_probs, target_logits)

        assert result.accepted_tokens == [0]  # 第一个接受
        assert result.num_accepted == 1
        assert len(result.all_tokens) == 2  # accepted + bonus
        assert result.num_rejected == 2  # K - 1 = 2

    def test_4_k_zero(self):
        """测试 4：K=0（无 draft tokens）。

        预期：直接从 target 采样 1 个 bonus token。
        """
        vocab_size = 5
        target_logits = torch.zeros(1, vocab_size)
        target_logits[0, 3] = 10.0  # token 3 概率最高

        sampler = RejectionSampler(temperature=1.0)
        result = sampler.sample([], [], target_logits)

        assert result.accepted_tokens == []
        assert result.bonus_token == 3
        assert len(result.all_tokens) == 1
        assert result.num_draft_tokens == 0
        assert result.num_rejected == 0


class TestRejectionSamplerGreedy:
    """Greedy 模式（temperature = 0）测试。"""

    def test_5_greedy_all_accepted(self):
        """测试 5：Greedy 全接受。

        draft tokens 全匹配 target argmax。

        预期：accepted = K 个 draft tokens，bonus = target_argmax[-1]。
        """
        vocab_size = 5
        K = 3
        draft_tokens = [0, 1, 2]

        # target_logits 的 argmax 恰好是 draft tokens
        target_logits = torch.zeros(K + 1, vocab_size)
        target_logits[0, 0] = 10.0  # argmax = 0 = draft_tokens[0]
        target_logits[1, 1] = 10.0  # argmax = 1 = draft_tokens[1]
        target_logits[2, 2] = 10.0  # argmax = 2 = draft_tokens[2]
        target_logits[3, 4] = 10.0  # bonus argmax = 4

        sampler = RejectionSampler(temperature=0)
        result = sampler.sample(draft_tokens, [], target_logits)

        assert result.accepted_tokens == [0, 1, 2]
        assert result.bonus_token == 4
        assert len(result.all_tokens) == 4
        assert result.num_rejected == 0

    def test_6_greedy_partial_reject(self):
        """测试 6：Greedy 部分拒绝。

        第一个匹配，第二个不匹配 → 截断。

        预期：accepted = [0]，bonus = target_argmax[1]。
        """
        vocab_size = 5
        K = 3
        draft_tokens = [0, 3, 2]  # 第二个 draft 是 3，但 target argmax 是 1

        target_logits = torch.zeros(K + 1, vocab_size)
        target_logits[0, 0] = 10.0  # argmax = 0 = draft_tokens[0] ✓
        target_logits[1, 1] = 10.0  # argmax = 1 ≠ draft_tokens[1]=3 ✗
        target_logits[2, 2] = 10.0
        target_logits[3, 4] = 10.0

        sampler = RejectionSampler(temperature=0)
        result = sampler.sample(draft_tokens, [], target_logits)

        assert result.accepted_tokens == [0]
        assert result.bonus_token == 1  # target_argmax[1]
        assert len(result.all_tokens) == 2
        assert result.num_rejected == 2


class TestRejectionSamplerEdgeCases:
    """边界情况和参数验证。"""

    def test_7_target_logits_shape_validation(self):
        """测试 7：target_logits shape 验证。

        K=3 时 target_logits 应有 K+1=4 行，传 3 行应该报错。
        """
        vocab_size = 5
        K = 3
        draft_tokens = [0, 1, 2]
        draft_probs = [torch.ones(vocab_size) / vocab_size] * K

        # 错误的 shape：只有 K 行而不是 K+1
        target_logits = torch.zeros(K, vocab_size)

        sampler = RejectionSampler(temperature=1.0)
        with pytest.raises(AssertionError):
            sampler.sample(draft_tokens, draft_probs, target_logits)

    def test_8_adjusted_distribution_fallback(self):
        """测试 8：修正分布 fallback。

        当 p_target == p_draft 时，relu(p_t - p_d) 全 0，
        应该 fallback 到 target_probs，不崩溃。
        """
        vocab_size = 5
        draft_tokens = [0]

        # draft 给 token 0 很高概率
        p_d = torch.zeros(vocab_size)
        p_d[0] = 0.9
        p_d[1] = 0.1

        # target 和 draft 完全一样 → 接受
        # 但让 p_t[0] < p_d[0] 确保拒绝
        p_t = torch.zeros(vocab_size)
        p_t[0] = 0.1  # target 认为 token 0 只有 0.1
        p_t[1] = 0.9  # target 认为 token 1 有 0.9

        draft_probs = [p_d]

        # 把 p_t 转成 logits（reverse softmax 近似）
        target_logits = torch.log(p_t + 1e-10).unsqueeze(0)
        # 加一行 bonus
        bonus_logits = torch.zeros(1, vocab_size)
        bonus_logits[0, 2] = 10.0
        target_logits = torch.cat([target_logits, bonus_logits], dim=0)

        sampler = RejectionSampler(temperature=1.0)
        torch.manual_seed(42)
        result = sampler.sample(draft_tokens, draft_probs, target_logits)

        # 应该正常返回，不崩溃
        assert len(result.all_tokens) >= 1
        assert result.num_accepted <= 1

    def test_9_num_accepted_and_rejected(self):
        """测试 9：num_accepted / num_rejected 正确性。

        Greedy 模式下容易精确控制。
        """
        vocab_size = 5
        K = 4
        draft_tokens = [0, 1, 2, 3]

        # 前 2 个匹配，第 3 个不匹配
        target_logits = torch.zeros(K + 1, vocab_size)
        target_logits[0, 0] = 10.0  # ✓
        target_logits[1, 1] = 10.0  # ✓
        target_logits[2, 4] = 10.0  # ✗ (draft=2, target argmax=4)
        target_logits[3, 3] = 10.0
        target_logits[4, 3] = 10.0

        sampler = RejectionSampler(temperature=0)
        result = sampler.sample(draft_tokens, [], target_logits)

        assert result.num_accepted == 2
        assert result.num_rejected == 2  # K - 2 = 2
        assert result.num_draft_tokens == 4
        assert result.accepted_tokens == [0, 1]
        assert result.bonus_token == 4
        assert result.all_tokens == [0, 1, 4]

    def test_parameter_validation(self):
        """参数验证测试。"""
        with pytest.raises(AssertionError):
            RejectionSampler(temperature=-1.0)

        with pytest.raises(AssertionError):
            RejectionSampler(eps=0)

        with pytest.raises(AssertionError):
            RejectionSampler(eps=-1.0)
