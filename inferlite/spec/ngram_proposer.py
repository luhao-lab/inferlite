"""NgramProposer：基于 n-gram 匹配的推测解码 drafter。

在 prompt + 已生成文本中找与当前后缀最长匹配的 n-gram，
然后提取匹配位置后面的 token 作为 draft。

核心优势：
- 零计算成本（只查表，不调用模型）
- 适合代码生成、多轮对话等高重复度场景

核心限制：
- 自然语言重复度低时，draft 命中率低
- 无法生成 context 中从未出现过的 token

实现策略：
- 当前用暴力滑动窗口 O(N × max_n)，教学优先
- vLLM 用 KMP 的 LPS 数组 O(N)，性能优先时可升级

参考：
- vLLM ngram_proposer.py（KMP 实现）
- SGLang NGRAM 模式
"""

from inferlite.spec.protocol import Proposer


class NgramProposer(Proposer):
    """基于 n-gram 匹配的推测解码 drafter。

    Args:
        min_n: 最小 n-gram 匹配长度
        max_n: 最大 n-gram 匹配长度
        num_draft_tokens: 每次 draft 几个 token
    """

    def __init__(self, min_n: int = 1, max_n: int = 4, num_draft_tokens: int = 5):
        """初始化 NgramProposer。

        Args:
            min_n: 最小 n-gram 匹配长度（默认 1）
                - 越小越容易命中，但 draft 质量越低
                - 建议 1-2
            max_n: 最大 n-gram 匹配长度（默认 4）
                - 越大 draft 质量越高，但命中率越低
                - 建议 3-5
            num_draft_tokens: 每次 draft 几个 token（默认 5）
                - 越多 verify 成本越高，但潜在加速越大
                - 建议 3-8
        """
        assert min_n >= 1, f"min_n must be >= 1, got {min_n}"
        assert max_n >= min_n, f"max_n must be >= min_n, got max_n={max_n}, min_n={min_n}"
        assert num_draft_tokens >= 1, f"num_draft_tokens must be >= 1, got {num_draft_tokens}"

        self.min_n = min_n
        self.max_n = max_n
        self.num_draft_tokens = num_draft_tokens

    def propose(self, context: list[int]) -> list[int]:
        """从 context 中找最长 n-gram 匹配，返回 draft tokens。

        算法（暴力滑动窗口，O(N × max_n)）：
        1. 从长到短尝试 n = max_n, max_n-1, ..., min_n
        2. 取 context 末尾 n 个 token 作为 query
        3. 在 context[:-n] 中找匹配（避免自匹配）
        4. 找到则提取匹配位置后的 num_draft_tokens 个 token
        5. 找不到则尝试更短的 n
        """
        # 边界情况：context 太短
        if len(context) < self.min_n:
            return []

        # 从长到短尝试 n-gram 匹配
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(context) < n:
                continue

            query = context[-n:]  # 末尾 n 个 token

            # 在 context[:-n] 中找匹配（避免自匹配）
            for i in range(len(context) - n):
                if context[i : i + n] == query:
                    # 找到匹配，提取后面的 token
                    start = i + n
                    end = min(start + self.num_draft_tokens, len(context))
                    draft = context[start:end]

                    if draft:
                        return draft

        return []
