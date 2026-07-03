# M3-T1 RequestState + FCFSScheduler

> M3 第一张任务卡：先做纯 Python 调度状态机，不碰模型、不碰 KV Cache。

## 元信息
- **任务 ID**: M3-T1
- **里程碑**: M3 — Continuous Batching
- **状态**: ⬜ pending
- **前置**: M2 完成；`docs/plan/M3.md` 已确认 continuous batching 技术选型
- **估时**: 2h

## 目标

**要解决什么问题**：

M2 只有单请求 generate，不存在请求队列。M3 要支持多请求 continuous batching，首先需要把请求生命周期抽象出来：

```text
waiting → running → finished
```

本卡只解决“请求如何进入、运行、完成”的状态机问题，不涉及模型 forward、KV Cache、attention mask。

**做完是什么效果**：

可以用纯 CPU 单测验证：

```python
scheduler.submit(req_a)
scheduler.submit(req_b)

running = scheduler.admit_until_full()
assert len(scheduler.running) <= scheduler.max_num_seqs

scheduler.mark_finished(req_a.request_id)
assert req_a.request_id in scheduler.finished
```

**不做什么（边界）**：

- 不分配 KV slot。
- 不做 prefill / decode。
- 不调用真实模型。
- 不做优先级、抢占、超时、batching window。
- 不做 HTTP server 或异步队列。

**在推理链路中的位置**：

```text
用户提交请求
  ↓
RequestState
  ↓
FCFSScheduler: waiting / running / finished
  ↓
后续 T2 分配 KV slot
  ↓
后续 T4 BatchEngine 执行 prefill/decode
```

## 产出文件

- `inferlite/scheduler/request.py::RequestState`
- `inferlite/scheduler/request.py::RequestStatus`
- `inferlite/scheduler/fcfs.py::FCFSScheduler`
- `inferlite/scheduler/__init__.py`
- `tests/unit/test_scheduler.py`

## 算法核心

```python
class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class RequestState:
    request_id: str
    prompt_ids: torch.Tensor
    max_new_tokens: int
    eos_token_id: int | None = None

    status: RequestStatus = RequestStatus.WAITING
    generated_ids: list[torch.Tensor] = field(default_factory=list)

    slot_id: int | None = None
    seq_len: int = 0
    num_generated: int = 0
    last_token: torch.Tensor | None = None


class FCFSScheduler:
    def __init__(self, max_num_seqs: int) -> None:
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[RequestState] = deque()
        self.running: dict[str, RequestState] = {}
        self.finished: dict[str, RequestState] = {}
        self.cancelled: dict[str, RequestState] = {}

    def submit(self, req: RequestState) -> None:
        ...

    def admit_until_full(self) -> list[RequestState]:
        ...

    def mark_finished(self, request_id: str) -> RequestState:
        ...

    def cancel(self, request_id: str) -> RequestState:
        ...
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|---|---|---|
| 1 | `submit()` 后请求进入 waiting | 队列长度和状态枚举 | 精确 |
| 2 | `admit_until_full()` 按 FCFS 顺序进入 running | request_id 顺序 | 精确 |
| 3 | running 数不超过 `max_num_seqs` | 内部不变量 | 精确 |
| 4 | `mark_finished()` 后从 running 移到 finished | 三队列状态 | 精确 |
| 5 | request_id 不能重复提交 | 抛 `ValueError` | 精确 |
| 6 | finished 请求不能再次 running | 状态不变量 | 精确 |
| 7 | 三队列守恒 | waiting + running + finished + cancelled == total | 精确 |
| 8 | `cancel()` 能从 waiting/running 移除 | 状态不变量 | 精确 |

## DoD

- [ ] `RequestState` / `RequestStatus` 落地。
- [ ] `FCFSScheduler` 支持 `submit` / `admit_until_full` / `mark_finished` / `cancel`。
- [ ] 三队列守恒测试覆盖。
- [ ] `max_num_seqs` 容量限制测试覆盖。
- [ ] `uv run pytest tests/unit/test_scheduler.py -q` 通过。
- [ ] 不修改模型层 / KV Cache / Engine 逻辑。
- [ ] commit `feat(scheduler): add FCFS request state machine (M3-T1 done)`。

## 坑（按概率排序）

1. **把 running list 当成固定 batch**：running 是当前活跃集合，每个 decode step 才从 running 取 batch。
2. **request_id 重复**：会导致 dict 覆盖，必须显式禁止。
3. **状态迁移不守恒**：任何请求只能属于 waiting/running/finished/cancelled 之一。
4. **T1 提前设计复杂策略**：不要做 priority、timeout、SLO-aware scheduling。

## 完成总结

待完成后补：本卡对 continuous batching 状态机的抽象、踩坑和后续 T2 依赖。
