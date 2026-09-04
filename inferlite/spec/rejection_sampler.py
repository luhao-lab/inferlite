"""RejectionSampler：推测解码的验证环节。

核心保证：lossless — 输出分布与无推测时完全一致。

算法：
1. 对每个 draft token，计算 accept_prob = min(1, p_target / p_draft)
2. 接受 → 继续下一个
3. 拒绝 → 从修正分布采样 bonus，截断返回
4. 全接受 → 从 target 最后一个位置采样 bonus

修正分布 = normalize(relu(p_target - p_draft))
  → 只从"target 比 draft 多给的概率"里采样
  → 数学上恰好补偿 draft 和 target 的差异

两种模式：
- Classic (temperature > 0): 概率接受 + 修正分布 bonus
- Greedy (temperature = 0): argmax 匹配 + argmax bonus

参考：
- Leviathan et al. 2023: arxiv 2211.17192
- Chen et al. 2023: arxiv 2302.01318
"""

import torch


class SampleResult:
    """Rejection sampling 的结果。

    Attributes:
        accepted_tokens: 被接受的 draft tokens（0 ~ K 个）
        bonus_token: 修正分布 / target 分布采样的 bonus（总是有 1 个）
    """

    def __init__(self, accepted_tokens: list[int], bonus_token: int, num_draft_tokens: int = 0):
        self.accepted_tokens = accepted_tokens
        self.bonus_token = bonus_token
        self.num_draft_tokens = num_draft_tokens  # 原始 draft 数量 K

    @property
    def all_tokens(self) -> list[int]:
        """accepted + bonus，总共输出的 token 列表。

        长度范围：1 ~ K+1
        - 最短 1 个：第一个就被拒绝，只有 bonus
        - 最长 K+1 个：全接受 + bonus
        """
        return self.accepted_tokens + [self.bonus_token]

    @property
    def num_accepted(self) -> int:
        """接受的 draft token 数量。"""
        return len(self.accepted_tokens)

    @property
    def num_rejected(self) -> int:
        """拒绝的 draft token 数量（不含 bonus）。"""
        return self.num_draft_tokens - self.num_accepted

    def __repr__(self) -> str:
        return (
            f"SampleResult(accepted={self.accepted_tokens}, "
            f"bonus={self.bonus_token}, total={len(self.all_tokens)})"
        )


class RejectionSampler:
    """推测解码的 rejection sampler。

    输入 draft_tokens (K 个) + draft_probs (K 个 [V]) + target_logits (K+1 个 [V])，
    输出 accepted_tokens + bonus_token (总共 1 ~ K+1 个 token)。

    保证输出分布 = target 模型单独 decode 的分布（lossless）。

    Args:
        temperature: 采样温度（0 = greedy，>0 = classic）
        eps: 防除零的最小概率值
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-10):
        assert temperature >= 0, f"temperature must be >= 0, got {temperature}"
        assert eps > 0, f"eps must be > 0, got {eps}"

        self.temperature = temperature
        self.eps = eps

    def sample(
        self,
        draft_tokens: list[int],
        draft_probs: list[torch.Tensor],
        target_logits: torch.Tensor,
    ) -> SampleResult:
        """验证 draft tokens，返回接受结果。

        Args:
            draft_tokens: [K] draft 生成的 token ids
            draft_probs: [K] 个 [vocab_size] 概率向量（Drafter 采样时的分布）
            target_logits: [K+1, vocab_size] target model 的 logits

        Returns:
            SampleResult(accepted_tokens, bonus_token)
            - all_tokens 长度 1 ~ K+1
        """
        K = len(draft_tokens)

        # 边界情况：无 draft tokens → 直接从 target 采样 bonus
        if K == 0:
            bonus = self._sample_bonus_from_logits(target_logits[0])
            return SampleResult([], bonus, num_draft_tokens=0)

        # 验证输入维度
        assert target_logits.shape[0] == K + 1, (
            f"target_logits 应有 {K + 1} 行（K+1 个位置），" f"实际 {target_logits.shape[0]}"
        )

        # 选择模式
        if self.temperature == 0:
            return self._sample_greedy(draft_tokens, target_logits)
        else:
            assert len(draft_probs) == K, f"draft_probs 应有 {K} 个，实际 {len(draft_probs)}"
            return self._sample_classic(draft_tokens, draft_probs, target_logits)

    def _sample_classic(
        self,
        draft_tokens: list[int],
        draft_probs: list[torch.Tensor],
        target_logits: torch.Tensor,
    ) -> SampleResult:
        """Classic 模式：概率接受 + 修正分布 bonus。

        算法：
        1. target_probs = softmax(target_logits / temperature)
        2. 对每个位置 i：
           accept_prob = min(1, p_target[i][token] / p_draft[i][token])
           如果 random < accept_prob → 接受
           否则 → 从修正分布采样 bonus，截断返回
        3. 全接受 → 从 target_probs[-1] 采样 bonus
        """
        K = len(draft_tokens)
        # softmax → 概率分布（K+1 个位置）
        target_probs = torch.softmax(target_logits / self.temperature, dim=-1)

        accepted = []

        for i, token in enumerate(draft_tokens):
            p_d = draft_probs[i][token].item()
            p_t = target_probs[i][token].item()

            # 计算接受概率：min(1, p_target / p_draft)
            # 防除零：p_draft 太小时直接拒绝
            if p_d < self.eps:
                accept_prob = 0.0
            else:
                accept_prob = min(1.0, p_t / p_d)

            if torch.rand(1).item() < accept_prob:
                accepted.append(token)
            else:
                # 拒绝：从修正分布采样 bonus
                bonus = self._sample_from_adjusted(target_probs[i], draft_probs[i])
                return SampleResult(accepted, bonus, num_draft_tokens=K)

        # 全接受：从 target 最后一个位置（第 K 个）采样 bonus
        bonus = torch.multinomial(target_probs[-1:], 1).item()
        return SampleResult(accepted, bonus, num_draft_tokens=K)

    def _sample_greedy(
        self,
        draft_tokens: list[int],
        target_logits: torch.Tensor,
    ) -> SampleResult:
        """Greedy 模式：argmax 匹配 + argmax bonus。

        算法：
        1. target_argmax = argmax(target_logits)
        2. 对每个位置 i：
           如果 draft_token == target_argmax[i] → 接受
           否则 → 用 target_argmax[i] 当 bonus，截断返回
        3. 全接受 → 用 target_argmax[-1] 当 bonus
        """
        K = len(draft_tokens)
        # 每个位置取 argmax（K+1 个位置）
        target_argmax = torch.argmax(target_logits, dim=-1)

        accepted = []

        for i, token in enumerate(draft_tokens):
            if token == target_argmax[i].item():
                accepted.append(token)
            else:
                # 拒绝：target 的 argmax 作为 bonus
                bonus = target_argmax[i].item()
                return SampleResult(accepted, bonus, num_draft_tokens=K)

        # 全接受：最后一个位置的 argmax 作为 bonus
        bonus = target_argmax[-1].item()
        return SampleResult(accepted, bonus, num_draft_tokens=K)

    def _sample_from_adjusted(
        self,
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> int:
        """从修正分布采样 bonus token。

        修正分布 = normalize(relu(p_target - p_draft))
          → 只从"target 比 draft 多给的概率"里采样
          → 补偿 draft 低估的 token

        如果修正分布全 0（极端情况），fallback 到 target_probs。
        """
        # relu：只保留 target > draft 的部分
        adjusted = torch.relu(target_probs - draft_probs)

        total = adjusted.sum()
        if total < self.eps:
            # fallback：修正分布全 0，直接用 target 分布
            adjusted = target_probs

        # 归一化（torch.relu 后可能 sum != 1）
        adjusted = adjusted / adjusted.sum()

        return torch.multinomial(adjusted, 1).item()

    def _sample_bonus_from_logits(self, logits: torch.Tensor) -> int:
        """从 logits 采样 bonus token（K=0 时使用）。"""
        if self.temperature == 0:
            return torch.argmax(logits).item()
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs, 1).item()
