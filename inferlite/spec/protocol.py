"""推测解码 drafter 的统一接口。"""

from typing import Protocol


class Proposer(Protocol):
    """推测解码 drafter 协议：context -> draft tokens。

    所有 drafter（n-gram、小模型、EAGLE 等）都遵循这个接口。
    engine loop 只依赖此协议，不关心具体 drafter 实现。

    调用示例：

        proposer = NgramProposer(min_n=1, max_n=4, num_draft_tokens=5)
        draft_tokens = proposer.propose(context)
        # draft_tokens 可能是 [] (无匹配) 或 [t1, t2, ...] (draft)
    """

    def propose(self, context: list[int]) -> list[int]: ...
