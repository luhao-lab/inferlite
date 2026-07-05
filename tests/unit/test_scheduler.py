"""M3-T1 FCFSScheduler + RequestState 单测。

测试目标（对应任务卡 L0 清单）：
  1. submit 后请求进入 waiting
  2. admit_until_full 按 FCFS 顺序进入 running
  3. running 数不超过 max_num_seqs
  4. mark_finished 后从 running 移到 finished
  5. request_id 不能重复提交
  6. finished 请求不能再次 running
  7. 四队列守恒：waiting + running + finished + cancelled == total
  8. cancel 能从 waiting/running 移除
  9. has_unfinished 正确判断
  10. generated_tokens 各请求独立（不共享 list）
"""

import pytest
import torch

from inferlite.scheduler.fcfs import FCFSScheduler
from inferlite.scheduler.request import RequestState, RequestStatus

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

MAX_NUM_SEQS = 2


def _make_request(req_id: str) -> RequestState:
    """创建测试用 RequestState。"""
    return RequestState(
        request_id=req_id,
        prompt_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=16,
    )


@pytest.fixture
def scheduler() -> FCFSScheduler:
    return FCFSScheduler(max_num_seqs=MAX_NUM_SEQS)


# ---------------------------------------------------------------------------
# Case 1: submit 后请求进入 waiting
# ---------------------------------------------------------------------------


def test_submit_enters_waiting(scheduler: FCFSScheduler) -> None:
    req = _make_request("a")
    scheduler.submit(req)

    assert len(scheduler.waiting) == 1
    assert req.status == RequestStatus.WAITING
    assert scheduler.waiting[0] is req


# ---------------------------------------------------------------------------
# Case 2: admit_until_full 按 FCFS 顺序进入 running
# ---------------------------------------------------------------------------


def test_admit_fcfs_order(scheduler: FCFSScheduler) -> None:
    for rid in ("a", "b", "c"):
        scheduler.submit(_make_request(rid))

    admitted = scheduler.admit_until_full()

    # max_num_seqs=2，所以只 admit a 和 b
    assert [r.request_id for r in admitted] == ["a", "b"]
    assert admitted[0].status == RequestStatus.RUNNING
    assert admitted[1].status == RequestStatus.RUNNING

    # c 还在 waiting
    assert len(scheduler.waiting) == 1
    assert scheduler.waiting[0].request_id == "c"


# ---------------------------------------------------------------------------
# Case 3: running 数不超过 max_num_seqs
# ---------------------------------------------------------------------------


def test_admit_respects_capacity(scheduler: FCFSScheduler) -> None:
    for rid in ("a", "b", "c", "d"):
        scheduler.submit(_make_request(rid))

    scheduler.admit_until_full()
    assert len(scheduler.running) == MAX_NUM_SEQS

    # 再次 admit，running 已满，不会新 admit
    admitted2 = scheduler.admit_until_full()
    assert admitted2 == []
    assert len(scheduler.running) == MAX_NUM_SEQS


# ---------------------------------------------------------------------------
# Case 4: mark_finished 后从 running 移到 finished
# ---------------------------------------------------------------------------


def test_mark_finished(scheduler: FCFSScheduler) -> None:
    scheduler.submit(_make_request("a"))
    scheduler.admit_until_full()

    req_a = scheduler.running["a"]
    scheduler.mark_finished(req_a)

    assert req_a.status == RequestStatus.FINISHED
    assert "a" not in scheduler.running
    assert "a" in scheduler.finished


def test_mark_finished_not_running_raises(scheduler: FCFSScheduler) -> None:
    req = _make_request("a")
    scheduler.submit(req)

    # req 还在 waiting，不在 running
    with pytest.raises(ValueError, match="not running"):
        scheduler.mark_finished(req)


# ---------------------------------------------------------------------------
# Case 5: request_id 不能重复提交
# ---------------------------------------------------------------------------


def test_duplicate_request_id_raises(scheduler: FCFSScheduler) -> None:
    scheduler.submit(_make_request("a"))

    with pytest.raises(ValueError, match="already exists"):
        scheduler.submit(_make_request("a"))


# ---------------------------------------------------------------------------
# Case 6: finished 请求不能再次 admit
# ---------------------------------------------------------------------------


def test_finished_cannot_reenter(scheduler: FCFSScheduler) -> None:
    scheduler.submit(_make_request("a"))
    scheduler.admit_until_full()

    req_a = scheduler.running["a"]
    scheduler.mark_finished(req_a)

    # finished 的请求不能再 submit（_known_request_ids 里有）
    with pytest.raises(ValueError, match="already exists"):
        scheduler.submit(req_a)


# ---------------------------------------------------------------------------
# Case 7: 四队列守恒
# ---------------------------------------------------------------------------


def test_conservation_invariant(scheduler: FCFSScheduler) -> None:
    """waiting + running + finished + cancelled == 总提交数。"""
    reqs = [_make_request(rid) for rid in ("a", "b", "c", "d")]
    for req in reqs:
        scheduler.submit(req)

    # 全部在 waiting
    total = len(reqs)
    _check_conservation(scheduler, total)

    # admit a, b
    scheduler.admit_until_full()
    _check_conservation(scheduler, total)

    # finish a
    scheduler.mark_finished(scheduler.running["a"])
    _check_conservation(scheduler, total)

    # cancel c (还在 waiting)
    scheduler.cancel(reqs[2])
    _check_conservation(scheduler, total)

    # cancel b (在 running)
    scheduler.cancel(scheduler.running["b"])
    _check_conservation(scheduler, total)


def _check_conservation(scheduler: FCFSScheduler, expected_total: int) -> None:
    actual = (
        len(scheduler.waiting)
        + len(scheduler.running)
        + len(scheduler.finished)
        + len(scheduler.cancelled)
    )
    assert actual == expected_total


# ---------------------------------------------------------------------------
# Case 8: cancel 能从 waiting/running 移除
# ---------------------------------------------------------------------------


def test_cancel_from_waiting(scheduler: FCFSScheduler) -> None:
    req = _make_request("a")
    scheduler.submit(req)
    scheduler.cancel(req)

    assert req.status == RequestStatus.CANCELLED
    assert len(scheduler.waiting) == 0
    assert "a" in scheduler.cancelled


def test_cancel_from_running(scheduler: FCFSScheduler) -> None:
    scheduler.submit(_make_request("a"))
    scheduler.admit_until_full()

    req_a = scheduler.running["a"]
    scheduler.cancel(req_a)

    assert req_a.status == RequestStatus.CANCELLED
    assert "a" not in scheduler.running
    assert "a" in scheduler.cancelled


def test_cancel_nonexistent_raises(scheduler: FCFSScheduler) -> None:
    req = _make_request("ghost")

    with pytest.raises(ValueError, match="does not exist"):
        scheduler.cancel(req)


# ---------------------------------------------------------------------------
# Case 9: has_unfinished 正确判断
# ---------------------------------------------------------------------------


def test_has_unfinished(scheduler: FCFSScheduler) -> None:
    assert not scheduler.has_unfinished()

    scheduler.submit(_make_request("a"))
    assert scheduler.has_unfinished()

    scheduler.admit_until_full()
    assert scheduler.has_unfinished()

    scheduler.mark_finished(scheduler.running["a"])
    assert not scheduler.has_unfinished()


def test_has_unfinished_with_waiting_only(scheduler: FCFSScheduler) -> None:
    """waiting 非空也算 unfinished。"""
    for rid in ("a", "b", "c"):
        scheduler.submit(_make_request(rid))

    # max_num_seqs=2，admit 后 c 还在 waiting
    scheduler.admit_until_full()
    assert scheduler.has_unfinished()

    # finish running 的两个
    for rid in list(scheduler.running):
        scheduler.mark_finished(scheduler.running[rid])

    # c 还在 waiting，应该还有 unfinished
    assert scheduler.has_unfinished()


# ---------------------------------------------------------------------------
# Case 10: generated_tokens 各请求独立
# ---------------------------------------------------------------------------


def test_generated_tokens_independent() -> None:
    """两个 RequestState 的 generated_tokens 不共享。"""
    a = _make_request("a")
    b = _make_request("b")

    a.generated_tokens.append(torch.tensor([1]))

    assert len(a.generated_tokens) == 1
    assert len(b.generated_tokens) == 0


# ---------------------------------------------------------------------------
# Case 11: continuous batching 场景模拟
# ---------------------------------------------------------------------------


def test_continuous_batching_trace(scheduler: FCFSScheduler) -> None:
    """模拟 continuous batching 的逐入逐出：短请求先完成释放 slot，等待请求进入。"""
    reqs = [_make_request(rid) for rid in ("a", "b", "c")]
    for req in reqs:
        scheduler.submit(req)

    # iteration 0: admit a, b
    admitted = scheduler.admit_until_full()
    assert [r.request_id for r in admitted] == ["a", "b"]

    # iteration 1: a finished，释放 slot
    scheduler.mark_finished(scheduler.running["a"])
    assert len(scheduler.running) == 1

    # iteration 1: admit c（a 的 slot 空出来了）
    admitted2 = scheduler.admit_until_full()
    assert [r.request_id for r in admitted2] == ["c"]
    assert len(scheduler.running) == 2
    assert "c" in scheduler.running
