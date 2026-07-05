"""M3 Continuous Batching 的 FCFS 调度器。

FCFS（First Come First Served）调度器管理多请求的生命周期状态机，
维护四个集合：

    waiting  →  running  →  finished / cancelled

核心职责：
  - submit:          接收新请求，放入 waiting 队列
  - admit_until_full: 从 waiting 按 FCFS 顺序取请求到 running，直到容量满
  - mark_finished:   请求生成完毕，从 running 移到 finished
  - cancel:          外部取消请求，从 waiting/running 移到 cancelled

status 字段的写操作仅在本模块内进行，外部代码只读不写。
"""

from collections import deque

from inferlite.scheduler.request import RequestState, RequestStatus


class FCFSScheduler:
    """FCFS 调度器：维护 waiting/running/finished/cancelled 四个集合的状态机。

    在 M3 continuous batching 主循环中的角色：
        while scheduler.has_unfinished():
            finish_done_requests()              # mark_finished
            admit_until_full()                  # admit + prefill
            batched_decode_one_step()           # decode running 集合

    容量限制：running 集合大小不超过 max_num_seqs。
    """

    def __init__(self, max_num_seqs: int) -> None:
        self.max_num_seqs: int = max_num_seqs
        # waiting: deque 保证 popleft O(1)，天然 FCFS 顺序
        self.waiting: deque[RequestState] = deque()
        # running: dict 方便按 request_id 快速查找和删除
        self.running: dict[str, RequestState] = {}
        # finished / cancelled: 终态集合，只进不出（M3 阶段）
        self.finished: dict[str, RequestState] = {}
        self.cancelled: dict[str, RequestState] = {}
        # _known_request_ids: 防重复提交，只加不删（M3 不需要清理）
        self._known_request_ids: set[str] = set()

    def submit(self, request: RequestState) -> None:
        """提交新请求到 waiting 队列。

        重复 request_id 抛 ValueError，防止同一请求被多次提交。
        """
        if request.request_id in self._known_request_ids:
            raise ValueError(f"request_id {request.request_id} already exists")
        request.status = RequestStatus.WAITING
        self.waiting.append(request)
        self._known_request_ids.add(request.request_id)

    def admit_until_full(self) -> list[RequestState]:
        """从 waiting 头部按 FCFS 顺序取请求到 running，直到容量满或 waiting 为空。

        返回本轮新 admit 的请求列表（供 BatchEngine 做 prefill）。
        """
        admitted: list[RequestState] = []
        while len(self.running) < self.max_num_seqs and self.waiting:
            req = self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            self.running[req.request_id] = req
            admitted.append(req)
        return admitted

    def mark_finished(self, request: RequestState) -> None:
        """标记请求生成完毕：running → finished。

        先检查请求是否在 running 中，不在则抛 ValueError。
        """
        if request.request_id not in self.running:
            raise ValueError(f"Trying to finish request {request.request_id} that is not running")
        request.status = RequestStatus.FINISHED
        self.finished[request.request_id] = request
        del self.running[request.request_id]

    def cancel(self, request: RequestState) -> None:
        """取消请求：从 waiting 或 running 移到 cancelled。

        如果请求既不在 running 也不在 waiting，抛 ValueError。
        """
        if request.request_id in self.running:
            del self.running[request.request_id]
        elif request in self.waiting:
            self.waiting.remove(request)
        else:
            raise ValueError(f"Trying to cancel request {request.request_id} that does not exist")
        request.status = RequestStatus.CANCELLED
        self.cancelled[request.request_id] = request

    def has_unfinished(self) -> bool:
        """是否还有待处理的请求（waiting 或 running 非空）。

        供 M3 主循环判断是否继续迭代。
        """
        return bool(self.waiting) or bool(self.running)
