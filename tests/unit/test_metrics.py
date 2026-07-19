"""Unit tests for M3-T6 metrics collector。

测试目标（对应 L0 测试清单）：
1. RequestMetrics 时间差计算（queue/prefill/ttft/decode/total/itl）
2. first token 时间只记录一次
3. StepMetrics batch_size
4. avg_batch_size
5. slot_utilization
6. output_tokens_per_s
7. benchmark smoke（summary 返回所有字段）
8. num_requests/max_num_slots 参数影响 trace

运行：
  uv run pytest tests/unit/test_metrics.py -v
"""

import time

from inferlite.engine.metrics import (
    MetricsCollector,
    RequestMetrics,
    StepMetrics,
)

# ---------------------------------------------------------------------------
# L0-1: RequestMetrics 时间差计算
# ---------------------------------------------------------------------------


def test_request_metrics_queue_ms():
    """queue_ms = (scheduled - arrival) * 1000。"""
    rm = RequestMetrics(request_id="0", arrival_ts=1.0)
    rm.scheduled_ts = 1.5
    assert rm.queue_ms == 500.0


def test_request_metrics_queue_ms_unscheduled():
    """未调度时 queue_ms = 0。"""
    rm = RequestMetrics(request_id="0", arrival_ts=1.0)
    assert rm.queue_ms == 0.0


def test_request_metrics_prefill_ms():
    """prefill_ms = (prefill_end - prefill_start) * 1000。"""
    rm = RequestMetrics(request_id="0", arrival_ts=0.0)
    rm.prefill_start_ts = 1.0
    rm.prefill_end_ts = 1.5
    assert rm.prefill_ms == 500.0


def test_request_metrics_ttft_ms():
    """ttft_ms = (first_token - arrival) * 1000（含 queue + prefill）。"""
    rm = RequestMetrics(request_id="0", arrival_ts=1.0)
    rm.first_token_ts = 2.0
    assert rm.ttft_ms == 1000.0


def test_request_metrics_decode_ms():
    """decode_ms = (finished - first_token) * 1000。"""
    rm = RequestMetrics(request_id="0", arrival_ts=0.0)
    rm.first_token_ts = 1.0
    rm.finished_ts = 3.0
    assert rm.decode_ms == 2000.0


def test_request_metrics_total_ms():
    """total_ms = (finished - arrival) * 1000。"""
    rm = RequestMetrics(request_id="0", arrival_ts=1.0)
    rm.finished_ts = 3.0
    assert rm.total_ms == 2000.0


def test_request_metrics_itl_ms():
    """itl_ms = decode_ms / (output_tokens - 1)。"""
    rm = RequestMetrics(request_id="0", arrival_ts=0.0)
    rm.first_token_ts = 1.0
    rm.finished_ts = 3.0
    rm.output_tokens = 5  # 4 个间隔
    assert rm.itl_ms == 500.0  # 2000ms / 4


def test_request_metrics_itl_ms_single_token():
    """output_tokens <= 1 时 itl_ms = 0。"""
    rm = RequestMetrics(request_id="0", arrival_ts=0.0)
    rm.first_token_ts = 1.0
    rm.finished_ts = 3.0
    rm.output_tokens = 1
    assert rm.itl_ms == 0.0


def test_request_metrics_itl_ms_zero_tokens():
    """output_tokens = 0 时 itl_ms = 0。"""
    rm = RequestMetrics(request_id="0", arrival_ts=0.0)
    assert rm.itl_ms == 0.0


# ---------------------------------------------------------------------------
# L0-2: first token 时间只记录一次
# ---------------------------------------------------------------------------


def test_first_token_recorded_once():
    """record_first_token 多次调用不应覆盖首次记录。"""
    collector = MetricsCollector()
    collector.record_arrival("0")
    time.sleep(0.01)
    collector.record_first_token("0")
    first_ts = collector.request_metrics["0"].first_token_ts
    assert first_ts is not None

    time.sleep(0.01)
    collector.record_first_token("0")  # 不应覆盖
    assert collector.request_metrics["0"].first_token_ts == first_ts


# ---------------------------------------------------------------------------
# L0-3: StepMetrics batch_size
# ---------------------------------------------------------------------------


def test_step_metrics_batch_size():
    """StepMetrics 记录 batch_size。"""
    sm = StepMetrics(
        step_idx=0,
        batch_size=4,
        max_seq_len=10,
        decode_ms=1.5,
        output_tokens=4,
        running_count=4,
        waiting_count=2,
        occupied_slots=4,
    )
    assert sm.batch_size == 4
    assert sm.output_tokens == 4
    assert sm.occupied_slots == 4


# ---------------------------------------------------------------------------
# L0-4: avg_batch_size
# ---------------------------------------------------------------------------


def test_avg_batch_size():
    """avg_batch_size = mean(step.batch_size)。"""
    collector = MetricsCollector()
    collector.record_step(
        0,
        batch_size=2,
        max_seq_len=5,
        decode_ms=1.0,
        output_tokens=2,
        running_count=2,
        waiting_count=1,
        occupied_slots=2,
    )
    collector.record_step(
        1,
        batch_size=3,
        max_seq_len=6,
        decode_ms=1.5,
        output_tokens=3,
        running_count=3,
        waiting_count=0,
        occupied_slots=3,
    )
    collector.record_step(
        2,
        batch_size=1,
        max_seq_len=7,
        decode_ms=0.5,
        output_tokens=1,
        running_count=1,
        waiting_count=0,
        occupied_slots=1,
    )
    assert collector.avg_batch_size == 2.0  # (2+3+1)/3


def test_avg_batch_size_empty():
    """无 step 时 avg_batch_size = 0。"""
    collector = MetricsCollector()
    assert collector.avg_batch_size == 0.0


# ---------------------------------------------------------------------------
# L0-5: slot_utilization
# ---------------------------------------------------------------------------


def test_slot_utilization():
    """slot_utilization = mean(occupied_slots / max_num_slots)。"""
    collector = MetricsCollector()
    collector.max_num_slots = 4
    collector.record_step(
        0,
        batch_size=2,
        max_seq_len=5,
        decode_ms=1.0,
        output_tokens=2,
        running_count=2,
        waiting_count=1,
        occupied_slots=2,
    )
    collector.record_step(
        1,
        batch_size=4,
        max_seq_len=6,
        decode_ms=1.5,
        output_tokens=4,
        running_count=4,
        waiting_count=0,
        occupied_slots=4,
    )
    # (2/4 + 4/4) / 2 = (0.5 + 1.0) / 2 = 0.75
    assert abs(collector.slot_utilization - 0.75) < 1e-6


def test_slot_utilization_zero_slots():
    """max_num_slots=0 时 slot_utilization = 0。"""
    collector = MetricsCollector()
    collector.max_num_slots = 0
    collector.record_step(
        0,
        batch_size=1,
        max_seq_len=5,
        decode_ms=1.0,
        output_tokens=1,
        running_count=1,
        waiting_count=0,
        occupied_slots=1,
    )
    assert collector.slot_utilization == 0.0


# ---------------------------------------------------------------------------
# L0-6: output_tokens_per_s / tpot_ms
# ---------------------------------------------------------------------------


def test_output_tokens_per_s():
    """output_tokens_per_s = total_output_tokens / (total_decode_ms / 1000)。"""
    collector = MetricsCollector()
    collector.record_step(
        0,
        batch_size=2,
        max_seq_len=5,
        decode_ms=10.0,
        output_tokens=2,
        running_count=2,
        waiting_count=0,
        occupied_slots=2,
    )
    collector.record_step(
        1,
        batch_size=2,
        max_seq_len=6,
        decode_ms=10.0,
        output_tokens=2,
        running_count=2,
        waiting_count=0,
        occupied_slots=2,
    )
    # total_tokens = 4, total_ms = 20, total_s = 0.02
    # throughput = 4 / 0.02 = 200
    assert abs(collector.output_tokens_per_s - 200.0) < 1e-6


def test_tpot_ms():
    """tpot_ms = total_decode_ms / total_output_tokens。"""
    collector = MetricsCollector()
    collector.record_step(
        0,
        batch_size=2,
        max_seq_len=5,
        decode_ms=10.0,
        output_tokens=2,
        running_count=2,
        waiting_count=0,
        occupied_slots=2,
    )
    # tpot = 10 / 2 = 5
    assert abs(collector.tpot_ms - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# L0-7/8: benchmark smoke + summary
# ---------------------------------------------------------------------------


def test_summary_returns_all_fields():
    """summary() 应返回所有汇总字段。"""
    collector = MetricsCollector()
    collector.max_num_slots = 2
    collector.record_arrival("0")
    collector.record_arrival("1")
    collector.record_scheduled("0")
    collector.record_prefill_start("0")
    collector.record_prefill_end("0")
    collector.record_first_token("0")
    collector.record_finished("0")
    collector.record_output_tokens("0", 5)
    collector.record_step(
        0,
        batch_size=2,
        max_seq_len=5,
        decode_ms=1.0,
        output_tokens=2,
        running_count=2,
        waiting_count=0,
        occupied_slots=2,
    )

    s = collector.summary()
    expected_keys = {
        "num_requests",
        "max_num_slots",
        "prefill_ms_p50",
        "decode_step_ms_p50",
        "ttft_ms_p50",
        "itl_ms_p50",
        "output_tokens_per_s",
        "tpot_ms",
        "avg_batch_size",
        "slot_utilization",
        "total_decode_ms",
        "total_output_tokens",
    }
    assert set(s.keys()) == expected_keys
    assert s["num_requests"] == 2
    assert s["max_num_slots"] == 2


def test_metrics_collector_default_empty():
    """MetricsCollector 默认空。"""
    collector = MetricsCollector()
    assert collector.max_num_slots == 0
    assert collector.request_metrics == {}
    assert collector.step_metrics == []
    assert collector.avg_batch_size == 0.0
    assert collector.slot_utilization == 0.0
    assert collector.total_decode_ms == 0.0
    assert collector.total_output_tokens == 0


# ---------------------------------------------------------------------------
# 采集方法验证
# ---------------------------------------------------------------------------


def test_record_arrival_creates_request_metrics():
    """record_arrival 应创建 RequestMetrics 并记录 arrival_ts。"""
    collector = MetricsCollector()
    collector.record_arrival("0")
    assert "0" in collector.request_metrics
    assert collector.request_metrics["0"].arrival_ts > 0


def test_record_step_appends_step_metrics():
    """record_step 应追加 StepMetrics。"""
    collector = MetricsCollector()
    collector.record_step(
        0,
        batch_size=2,
        max_seq_len=5,
        decode_ms=1.0,
        output_tokens=2,
        running_count=2,
        waiting_count=0,
        occupied_slots=2,
    )
    assert len(collector.step_metrics) == 1
    assert collector.step_metrics[0].step_idx == 0
    assert collector.step_metrics[0].batch_size == 2
