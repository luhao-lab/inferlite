"""Greedy next-token sampler.

Greedy decoding 是最简单的确定性解码策略：每一步都选择 logit 最大的 token。
T9 只实现这个最小 sampler；temperature/top-k/top-p 等随机采样留到后续任务。
"""

import torch


class GreedySampler:
    """选择每行 logits 中分数最大的 token。

    输入约定：
        logits: [B, V]，已经是最后一个位置的 vocab logits。

    输出约定：
        next_token_ids: [B, 1]，保留二维形状，方便后续与 input_ids 在 seq 维拼接：
        `torch.cat([input_ids, next_token_ids], dim=1)`。
    """

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """logits [B, V] -> next_token_ids [B, 1]。"""
        next_token = torch.argmax(input=logits, dim=-1, keepdim=True)
        return next_token
