"""MTPProposer：基于多 head 的 MTP drafter。

用 K 个独立的 MTP head，每个直接从 h_t 预测 +N 位置的 hidden state，
避免递归误差累积。

和 EagleProposer（T3）的区别：
- T3：1 个 head 递归 K 次，误差累积
- T4：K 个 head 各预测 1 次，无累积

注意：这是"伪 MTP"——外挂多个 head，不是原生 MTP 模型结构修改。
原生 MTP（如 DeepSeek-V3）在预训练阶段就内置了 MTP module。

参考：
- DeepSeek-V3 论文: arxiv 2412.19437
- EAGLE 论文: arxiv 2401.15077
"""

import torch

from inferlite.spec.eagle_head import EagleHead
from inferlite.spec.protocol import Proposer


class MTPProposer(Proposer):
    """基于多 head 的 MTP drafter。

    用 K 个独立的 head，每个直接从 h_t 预测 +N 位置的 hidden state。
    head 之间没有依赖关系，不存在递归误差累积。

    设计说明：
    - 每个 head 独立预测：head_i(h_t) → h_{t+i}
    - draft 质量随 K 增大下降更慢（对比 EagleProposer）
    - K 在训练时固定（训了几个 head 就只能 draft 几个 token）
    - 本质是"伪 MTP"：外挂多个 head，不是原生 MTP 模型

    Args:
        target_model: target 模型（用于提取 hidden states + lm_head）
        heads: 多个 EagleHead，按顺序预测 +1, +2, +3, ...
        num_draft_tokens: 每次 draft 几个 token（不超过 len(heads)）
        temperature: 采样温度（默认 1.0）
    """

    def __init__(
        self,
        target_model,
        heads: list[EagleHead],
        num_draft_tokens: int = 3,
        temperature: float = 1.0,
    ):
        assert num_draft_tokens >= 0, f"num_draft_tokens must be >= 0, got {num_draft_tokens}"
        assert temperature > 0, f"temperature must be > 0, got {temperature}"
        assert len(heads) > 0, "heads must not be empty"

        self.target_model = target_model
        self.heads = heads
        # draft 数量不超过 head 数量
        self.num_draft_tokens = min(num_draft_tokens, len(heads))
        self.temperature = temperature
        self.last_draft_probs: list[torch.Tensor] = []

    def propose(self, context: list[int]) -> list[int]:
        """用多个 head 独立预测，生成 draft tokens。

        算法：
        1. 从 target model 提取最后一个 hidden state h_t
        2. 对每个 head_i：
           a. h_{t+i} = head_i(h_t)    # 直接从 h_t 预测，不依赖前一个 head
           b. logits = target_model.lm_head(h_{t+i})
           c. probs = softmax(logits / temperature)
           d. token = multinomial(probs)
        3. 返回 [token_{t+1}, token_{t+2}, ..., token_{t+K}]
        """
        if not context:
            self.last_draft_probs = []
            return []

        if self.num_draft_tokens == 0:
            self.last_draft_probs = []
            return []

        # 从 model 参数推断 device
        device = next(self.target_model.parameters()).device

        # 提取最后一个 hidden state
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        hidden_states = self.target_model.model(input_ids)  # [1, seq, hidden_size]
        h_t = hidden_states[0, -1]  # [hidden_size]

        draft_tokens = []
        draft_probs = []

        for i in range(self.num_draft_tokens):
            # 每个 head 独立从 h_t 预测 +i+1 位置
            h_next = self.heads[i](h_t)
            logits = self.target_model.lm_head(h_next)  # [vocab_size]
            logits = logits / self.temperature
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()

            draft_tokens.append(token)
            draft_probs.append(probs)
            # 注意：不更新 h_t，下一个 head 还是从原始 h_t 预测

        self.last_draft_probs = draft_probs
        return draft_tokens
