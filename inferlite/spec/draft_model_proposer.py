"""DraftModelProposer：基于模型的推测解码 drafter。

用 draft_model 做 K 次 forward，每次用 multinomial sampling 生成 1 个 token，
累积 K 个 draft tokens 和对应的 draft_probs。

核心优势：
- draft 质量高（模型理解语义，不只是查表）
- p_draft 是真实的生成分布，rejection sampling 数学保证成立

核心限制：
- 每次 draft 需要 K 次 forward（比 n-gram 慢）
- 需要和 target model 共享 tokenizer（否则 p_target/p_draft 无意义）

参考：
- vLLM DraftModelProposer
- Leviathan et al. 2023（原版 speculative decoding 论文）
"""

import torch

from inferlite.spec.protocol import Proposer


class DraftModelProposer(Proposer):
    """基于模型的推测解码 drafter。

    设计说明：
    - draft_model 和 target_model 在代码上是两个独立引用
    - T2 传入同一个模型实例（pipeline 验证）
    - M10 可传入同模型不同配置（如 top-2 vs top-8 experts）

    Args:
        draft_model: 用于 draft 的模型（实现 LLMModel Protocol）
        num_draft_tokens: 每次 draft 几个 token（默认 5）
        temperature: 采样温度（默认 1.0）
    """

    def __init__(self, draft_model, num_draft_tokens: int = 5, temperature: float = 1.0):
        assert num_draft_tokens >= 0, f"num_draft_tokens must be >= 0, got {num_draft_tokens}"
        assert temperature > 0, f"temperature must be > 0, got {temperature}"

        self.draft_model = draft_model
        self.num_draft_tokens = num_draft_tokens
        self.temperature = temperature
        self.last_draft_probs: list[torch.Tensor] = []

    def propose(self, context: list[int]) -> list[int]:
        """用 draft_model 做 K 次 forward，生成 draft tokens。

        算法：
        1. 初始化：last_token = context[-1]
        2. 循环 K 次：
           a. logits = draft_model([last_token])
           b. logits = logits / temperature
           c. probs = softmax(logits)
           d. next_token = multinomial(probs, 1)
           e. 保存 probs
        3. 返回 draft_tokens

        为什么用 multinomial 而不是 greedy：
        - greedy 让 p_draft 变成 one-hot，rejection sampling 退化
        - multinomial 保持 p_draft 是真实分布，rejection sampling 正常工作
        """
        if not context:  # 空 context 检查
            self.last_draft_probs = []
            return []

        if self.num_draft_tokens == 0:
            self.last_draft_probs = []
            return []

        draft_tokens = []
        draft_probs = []
        last_token = context[-1]

        for _ in range(self.num_draft_tokens):
            # draft_model forward
            logits = self.draft_model(input_ids=[[last_token]])
            logits = logits[0, 0]  # [vocab_size]

            # 温度缩放
            logits = logits / self.temperature

            # softmax → 概率分布
            probs = torch.softmax(logits, dim=-1)  # [vocab_size]

            # multinomial sampling（不是 greedy）
            next_token = torch.multinomial(probs, 1).item()

            draft_tokens.append(next_token)
            draft_probs.append(probs)
            last_token = next_token

        self.last_draft_probs = draft_probs
        return draft_tokens
