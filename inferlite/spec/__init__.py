"""推测解码（Speculative Decoding）模块。

本模块实现推测解码的核心组件：
- Drafter：快速猜测后续 token 的 proposer（n-gram、小模型、EAGLE 等）
- RejectionSampler：验证 draft tokens 并决定接受哪些
- KV cache rollback：清理被拒绝的 draft tokens 写入的 cache

推测解码的核心思想：
1. Drafter 快速猜 K 个 token（如 n-gram 查表、小模型 forward）
2. Target model 一次 forward 验证 K 个 token（prefill 模式）
3. Rejection sampling 决定接受前 N 个，拒绝后面的
4. 一次 forward 产出 N+1 个 token（N 个接受 + 1 个 bonus）

关键保证：lossless——输出分布与无推测时完全一致。
"""

from inferlite.spec.ngram_proposer import NgramProposer
from inferlite.spec.protocol import Proposer

__all__ = ["Proposer", "NgramProposer"]
