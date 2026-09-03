"""EagleHead：feature-level draft head。

用 2 层 MLP 预测下一个 hidden state，比跑完整 decoder layer 快 100 倍。

核心思想：transformer 最后一层 hidden states 在相邻位置变化平滑，
简单 MLP 就能近似 h_t → h_{t+1} 的映射。

参考：
- EAGLE 论文: arxiv 2401.15077
- Qwen3 SwiGLUMLP: Linear → SiLU → Linear（EagleHead 是其极简版）
"""

import torch.nn as nn
import torch.nn.functional as F


class EagleHead(nn.Module):
    """h_t → h_{t+1} 的特征预测器。

    2 层 MLP：Linear → SiLU → Linear
    和 Qwen3 的 SwiGLUMLP 结构一致但更轻量（去掉门控机制）。

    Args:
        hidden_size: hidden state 维度（Qwen3-0.6B 为 1024）
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h_t):
        """h_t → h_{t+1}

        SiLU(x) = x * sigmoid(x)：无参数的自门控机制，
        负值区域不完全归零（vs ReLU），梯度更平滑。

        Args:
            h_t: [hidden_size] 或 [batch, hidden_size]

        Returns:
            h_next: 同 shape，预测的下一个 hidden state
        """
        h = F.silu(self.fc1(h_t))
        h = self.fc2(h)
        return h
