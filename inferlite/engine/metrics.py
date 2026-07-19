import time
from dataclasses import dataclass, field
from statistics import fmean


@dataclass
class RequestMetrics:
    """请求级时间戳 + 派生指标。所有时间戳用 time.perf_counter()，单位秒。"""

    request_id: str
    arrival_ts: float
    scheduled_ts: float | None = None
    prefill_start_ts: float | None = None
    prefill_end_ts: float | None = None
    first_token_ts: float | None = None
    finished_ts: float | None = None

    prompt_tokens: int = 0
    output_tokens: int = 0

    def _elapsed_ms(self, start: float | None, end: float | None) -> float:
        """算 (end - start) 秒差，None 或负数返回 0。"""
        if start is None or end is None:
            return 0.0
        return max(0.0, end - start) * 1000

    @property
    def queue_ms(self) -> float:
        """scheduled - arrival，未调度返回 0。"""
        return self._elapsed_ms(self.arrival_ts, self.scheduled_ts)

    @property
    def prefill_ms(self) -> float:
        """prefill_end - prefill_start, 未开始返回0."""
        return self._elapsed_ms(self.prefill_start_ts, self.prefill_end_ts)

    @property
    def ttft_ms(self) -> float:
        """first_token - arrival，TTFT（含 queue + prefill + 首 token 采样），未生成返回0."""
        return self._elapsed_ms(self.arrival_ts, self.first_token_ts)

    @property
    def decode_ms(self) -> float:
        """finished - first_token, 未生成返回0."""
        return self._elapsed_ms(self.first_token_ts, self.finished_ts)

    @property
    def total_ms(self) -> float:
        """finished - arrival, 未完成返回0."""
        return self._elapsed_ms(self.arrival_ts, self.finished_ts)

    @property
    def itl_ms(self) -> float:
        """decode_ms / (output_tokens - 1)，相邻 token 平均间隔。
        output_tokens <= 1 时返回 0（第一个 token 已计入 TTFT）。"""
        if self.output_tokens <= 1:
            return 0.0
        return self.decode_ms / (self.output_tokens - 1)


@dataclass
class StepMetrics:
    """单个 decode step 的指标快照."""

    step_idx: int
    batch_size: int  # 本轮 running 数（= forward 的 batch 维）
    max_seq_len: int  # 本轮最大 seq_len（cache_positions.max() + 1）
    decode_ms: float  # 本轮 decode forward wall clock
    output_tokens: int  # 本轮产出的 token 数（= batch_size）
    running_count: int  # = batch_size
    waiting_count: int  # 本轮 waiting 队列长度
    occupied_slots: int  # = running_count


@dataclass
class MetricsCollector:
    """采集 + 聚合。"""

    request_metrics: dict[str, RequestMetrics] = field(default_factory=dict)
    step_metrics: list[StepMetrics] = field(default_factory=list)
    max_num_slots: int = 0

    # ── 请求级采集（在 batch_generate 各阶段调用）──
    def record_arrival(self, request_id: str) -> None:
        self.request_metrics[request_id] = RequestMetrics(
            request_id=request_id,
            arrival_ts=time.perf_counter(),
        )

    def record_scheduled(self, request_id: str) -> None:
        self.request_metrics[request_id].scheduled_ts = time.perf_counter()

    def record_prefill_start(self, request_id: str) -> None:
        self.request_metrics[request_id].prefill_start_ts = time.perf_counter()

    def record_prefill_end(self, request_id: str) -> None:
        self.request_metrics[request_id].prefill_end_ts = time.perf_counter()

    def record_first_token(self, request_id: str) -> None:
        """只记一次：first_token_ts is None 才写。"""
        rm = self.request_metrics[request_id]
        if rm.first_token_ts is None:
            rm.first_token_ts = time.perf_counter()

    def record_finished(self, request_id: str) -> None:
        self.request_metrics[request_id].finished_ts = time.perf_counter()

    def record_prompt_tokens(self, request_id: str, n: int) -> None:
        self.request_metrics[request_id].prompt_tokens = n

    def record_output_tokens(self, request_id: str, n: int) -> None:
        self.request_metrics[request_id].output_tokens = n

    # ── 步级采集 ──
    def record_step(
        self,
        step_idx: int,
        batch_size: int,
        max_seq_len: int,
        decode_ms: float,
        output_tokens: int,
        running_count: int,
        waiting_count: int,
        occupied_slots: int,
    ) -> None:
        self.step_metrics.append(
            StepMetrics(
                step_idx=step_idx,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                decode_ms=decode_ms,
                output_tokens=output_tokens,
                running_count=running_count,
                waiting_count=waiting_count,
                occupied_slots=occupied_slots,
            )
        )

    # ── 汇总（@property，延迟计算）──
    @property
    def avg_batch_size(self) -> float:
        """mean(step.batch_size)。"""
        if not self.step_metrics:
            return 0.0
        return fmean(s.batch_size for s in self.step_metrics)

    @property
    def slot_utilization(self) -> float:
        """mean(occupied_slots / max_num_slots)。"""
        if not self.step_metrics or self.max_num_slots == 0:
            return 0.0
        return fmean(s.occupied_slots / self.max_num_slots for s in self.step_metrics)

    @property
    def total_decode_ms(self) -> float:
        """sum(step.decode_ms)。"""
        return sum(s.decode_ms for s in self.step_metrics)

    @property
    def total_output_tokens(self) -> int:
        """sum(step.output_tokens)。"""
        return sum(s.output_tokens for s in self.step_metrics)

    @property
    def output_tokens_per_s(self) -> float:
        """total_output_tokens / (total_decode_ms / 1000)。"""
        total_s = self.total_decode_ms / 1000
        if total_s <= 0:
            return 0.0
        return self.total_output_tokens / total_s

    @property
    def tpot_ms(self) -> float:
        """total_decode_ms / total_output_tokens。"""
        if self.total_output_tokens == 0:
            return 0.0
        return self.total_decode_ms / self.total_output_tokens

    @property
    def ttft_ms_p50(self) -> float:
        """mean of all requests' ttft_ms（教学版用 mean 代替 p50）。"""
        ttfts = [rm.ttft_ms for rm in self.request_metrics.values() if rm.ttft_ms > 0]
        return fmean(ttfts) if ttfts else 0.0

    @property
    def itl_ms_p50(self) -> float:
        """mean of all requests' itl_ms。"""
        itls = [rm.itl_ms for rm in self.request_metrics.values() if rm.itl_ms > 0]
        return fmean(itls) if itls else 0.0

    @property
    def prefill_ms_p50(self) -> float:
        """mean of all requests' prefill_ms。"""
        prefills = [rm.prefill_ms for rm in self.request_metrics.values() if rm.prefill_ms > 0]
        return fmean(prefills) if prefills else 0.0

    def summary(self) -> dict[str, float]:
        """返回所有汇总指标，benchmark 脚本用来输出。"""
        return {
            "num_requests": len(self.request_metrics),
            "max_num_slots": self.max_num_slots,
            "prefill_ms_p50": self.prefill_ms_p50,
            "decode_step_ms_p50": fmean(s.decode_ms for s in self.step_metrics)
            if self.step_metrics
            else 0.0,
            "ttft_ms_p50": self.ttft_ms_p50,
            "itl_ms_p50": self.itl_ms_p50,
            "output_tokens_per_s": self.output_tokens_per_s,
            "tpot_ms": self.tpot_ms,
            "avg_batch_size": self.avg_batch_size,
            "slot_utilization": self.slot_utilization,
            "total_decode_ms": self.total_decode_ms,
            "total_output_tokens": self.total_output_tokens,
        }
