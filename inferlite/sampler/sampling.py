"""M6 采样参数与 logits 处理器。

支持 vLLM 风格的采样参数：temperature / top_k / top_p / repetition_penalty。
GreedySampler 保持不变（向后兼容），SamplingProcessor 是新的可配置采样器。

用法：
    params = SamplingParams(temperature=0.7, top_p=0.9, top_k=50)
    processor = SamplingProcessor(params)
    next_token = processor(logits, input_ids)

参考：
- temperature: 标准 softmax temperature scaling
- top_k: 只保留 logits 最高的 k 个 token
- top_p (nucleus): 保留累积概率不超过 p 的最小 token 集合
- repetition_penalty: CTRL 论文 (https://arxiv.org/abs/1909.05858) 的 logit 惩罚
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    """采样参数，不可变数据类。

    所有参数都有合理默认值，temperature=0 退化为 greedy。

    Attributes:
        temperature: Softmax 温度。0 = greedy (argmax)，>1 更随机，<1 更确定。
        top_k: 保留 logits 最高的 k 个 token。0 = 不过滤（保留全部）。
        top_p: Nucleus sampling，保留累积概率 <= top_p 的最小 token 集。1.0 = 不过滤。
        repetition_penalty: 重复惩罚。1.0 = 无惩罚，>1 惩罚已出现的 token。
    """

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    @property
    def is_greedy(self) -> bool:
        """temperature=0 且无 repetition_penalty 时为纯 greedy。"""
        return self.temperature == 0.0 and self.repetition_penalty == 1.0


class SamplingProcessor:
    """可配置的 logits 处理器 + 采样器。

    处理流程：
    1. repetition_penalty: 对已出现 token 的 logits 做惩罚
    2. temperature: logits / T（T=0 时跳过，走 argmax）
    3. top_k: 只保留 top-k
    4. top_p: 保留累积概率 <= p 的最小集合
    5. multinomial sampling（或 argmax if T=0）
    """

    def __init__(self, params: SamplingParams | None = None) -> None:
        self.params = params or SamplingParams()

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        """处理 logits 并采样下一个 token。

        Args:
            logits: [B, V] 最后一层的 logits。
            input_ids: [B, T] 已生成的 token 序列（repetition_penalty 需要）。

        Returns:
            next_token_ids: [B, 1]
        """
        p = self.params

        # 纯 greedy 快速路径（无需任何处理）
        if p.is_greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits.clone()  # 不修改原始 logits

        # ── 1. Repetition penalty ──
        if p.repetition_penalty != 1.0 and input_ids is not None:
            logits = self._apply_repetition_penalty(logits, input_ids, p.repetition_penalty)

        # ── 2. Greedy (T=0 with repetition_penalty) ──
        if p.temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # ── 3. Temperature ──
        logits = logits / p.temperature

        # ── 4. Top-k ──
        if p.top_k > 0:
            logits = self._apply_top_k(logits, p.top_k)

        # ── 5. Top-p (nucleus) ──
        if p.top_p < 1.0:
            logits = self._apply_top_p(logits, p.top_p)

        # ── 6. Softmax + multinomial sampling ──
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token  # [B, 1]

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, input_ids: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """CTRL 论文风格的 repetition penalty。

        对 input_ids 中出现过的 token：
          - logits > 0 → logits / penalty（降低正 logit）
          - logits <= 0 → logits * penalty（增大负 logit 的绝对值）

        效果：让已出现的 token 被选中的概率降低。
        """
        for i in range(logits.shape[0]):
            unique_ids = input_ids[i].unique()
            prev_logits = logits[i, unique_ids]
            logits[i, unique_ids] = torch.where(
                prev_logits > 0,
                prev_logits / penalty,
                prev_logits * penalty,
            )
        return logits

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        """只保留 top-k logits，其余设为 -inf。"""
        top_k_values, _ = torch.topk(logits, min(k, logits.shape[-1]))
        threshold = top_k_values[:, -1:]  # [B, 1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        return logits

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus sampling：保留累积概率 <= p 的最小 token 集合。

        算法：
        1. 对 logits 降序排列
        2. 计算累积 softmax 概率
        3. 找到累积概率 > p 的位置，将这些 token 的 logits 设为 -inf
        4. 注意：第一个 token 总是保留（即使它的概率已经超过 p）
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 找到累积概率 > p 的位置（shift right 确保至少保留第一个 token）
        sorted_mask = cumulative_probs - sorted_probs > p

        # 将超过阈值的 sorted logits 设为 -inf
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

        # 恢复到原始顺序
        logits = torch.zeros_like(logits)
        logits.scatter_(1, sorted_indices, sorted_logits)
        return logits
