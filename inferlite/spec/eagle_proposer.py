"""EagleProposer：基于 EAGLE head 的 feature-level drafter。

用训好的小 MLP 预测 hidden state 序列，比完整 model forward 快 100 倍。

核心优势：
- 每步只需 2 个矩阵乘法 + 1 个 SiLU（vs 36 层 decoder layer）
- feature-level drafting：在 hidden state 空间做预测，不是 token 空间

核心限制：
- 递归 K 次会累积误差（teacher forcing vs autoregressive）
- K 越大 draft 质量越低，推荐 K=3-5

参考：
- EAGLE 论文: arxiv 2401.15077
- vLLM EagleProposer
"""

import torch

from inferlite.spec.protocol import Proposer


class EagleProposer(Proposer):
    """基于 EAGLE head 的 feature-level drafter。

    用训好的小 MLP 预测 hidden state 序列，比完整 model forward 快 100 倍。

    设计说明：
    - 递归 K 次：h_{t+1} = eagle_head(h_t)
    - 每步成本：2 个矩阵乘法 + 1 个 SiLU（vs 36 层 decoder layer）
    - 误差累积：递归 K 次会累积误差，K 越大 draft 质量越低

    Args:
        target_model: target 模型（用于提取 hidden states + lm_head）
        eagle_head: 训好的 EagleHead
        num_draft_tokens: 每次 draft 几个 token（默认 5）
        temperature: 采样温度（默认 1.0）
    """

    def __init__(
        self, target_model, eagle_head, num_draft_tokens: int = 5, temperature: float = 1.0
    ):
        assert num_draft_tokens >= 0, f"num_draft_tokens must be >= 0, got {num_draft_tokens}"
        assert temperature > 0, f"temperature must be > 0, got {temperature}"

        self.target_model = target_model
        self.eagle_head = eagle_head
        self.num_draft_tokens = num_draft_tokens
        self.temperature = temperature
        self.last_draft_probs: list[torch.Tensor] = []

    def propose(self, context: list[int]) -> list[int]:
        """用 eagle_head 递归预测，生成 draft tokens。

        算法：
        1. 从 target model 提取最后一个 hidden state h_t
        2. 递归 K 次：
           a. h_next = eagle_head(h_t)
           b. logits = target_model.lm_head(h_next)
           c. probs = softmax(logits / temperature)
           d. token = multinomial(probs)
           e. h_t = h_next  # 递归
        3. 返回 draft_tokens + last_draft_probs
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

        for _ in range(self.num_draft_tokens):
            h_next = self.eagle_head(h_t)
            logits = self.target_model.lm_head(h_next)  # [vocab_size]
            logits = logits / self.temperature
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()

            draft_tokens.append(token)
            draft_probs.append(probs)
            h_t = h_next  # 递归

        self.last_draft_probs = draft_probs
        return draft_tokens
