"""M3 Continuous Batching 的请求状态与生命周期管理。

本模块定义两个核心类型：

- RequestStatus: 请求生命周期枚举（WAITING → RUNNING → FINISHED / CANCELLED）
- RequestState:  单个推理请求的全部运行时状态

RequestState 是调度器、KV Cache、BatchEngine 之间共享的数据载体：
  - Scheduler 管理 status / slot_id 的迁移
  - BatchEngine 读写 last_token / num_generated / seq_len / generated_tokens
  - KV Cache 通过 slot_id 定位该请求的缓存位置
"""

from dataclasses import dataclass, field
from enum import Enum

import torch


class RequestStatus(Enum):
    """请求生命周期的四个状态。

    状态迁移由 FCFSScheduler 独占修改，外部代码只读不写：
        WAITING → RUNNING → FINISHED
                          → CANCELLED（可从 WAITING 或 RUNNING 取消）
    """

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class RequestState:
    """单个推理请求的全部运行时状态。

    字段按职责分为四组：
      身份：request_id
      输入：prompt_ids
      结束条件：max_new_tokens, eos_token_id
      输出 / 进度：last_token, num_generated, generated_tokens, status
      KV Cache 相关：slot_id, seq_len
    """

    # ── 身份 ──
    request_id: str

    # ── 输入 ──
    # prompt_ids: [1, prompt_len]，原始 prompt 的 token id 序列
    prompt_ids: torch.Tensor

    # ── 结束条件 ──
    # 最多生成多少个新 token；达到后强制 finished
    max_new_tokens: int
    # EOS token id；生成到该 token 时提前 finished。None 表示不用 EOS 停止
    eos_token_id: int | None = None

    # ── 输出 / 进度 ──
    # 上一步生成的 token（用于下一步 decode 的 input）
    last_token: torch.Tensor | None = None
    # 已生成的 token 总数
    num_generated: int = 0
    # 生命周期状态，仅由 FCFSScheduler 修改
    status: RequestStatus = RequestStatus.WAITING
    # 每步生成的 token，用 list 存放；最终拼接为完整输出序列
    # 必须用 default_factory 保证每个 RequestState 有独立的 list（避免共享引用）
    generated_tokens: list[torch.Tensor] = field(default_factory=list)

    # ── KV Cache 相关 ──
    # 该请求在 KV Cache 中的 slot 编号，由 SlotManager 分配；None 表示尚未分配
    slot_id: int | None = None
    # KV Cache 中的有效长度 = prompt_len + num_generated
    # 用于 attention mask 和 cache 切片
    seq_len: int = 0
