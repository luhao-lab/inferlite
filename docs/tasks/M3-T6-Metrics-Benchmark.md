# M3-T6 Metrics + Benchmark

> M3 第六张任务卡：把 prefill/decode/TTFT/ITL/throughput 等指标拆出来，证明 M3 的收益来自 decode batching。

## 元信息
- **任务 ID**: M3-T6
- **里程碑**: M3 — Continuous Batching
- **状态**: ⬜ pending
- **前置**: M3-T5
- **估时**: 3h

## 目标

**要解决什么问题**：

M3 引入 continuous batching 后，不能只看总耗时。需要拆解：

```text
queue time
prefill time
decode step time
TTFT
ITL / TPOT
slot utilization
batch size per step
output tokens/s
```

这些指标用来回答：

1. prefill 是否仍然逐条执行？
2. decode 是否真的组 batch？
3. slot 是否被有效利用？
4. 多请求吞吐是否高于串行？
5. 短请求是否避免了 head-of-line blocking？

**做完是什么效果**：

可以运行：

```bash
uv run python benchmarks/bench_continuous_batching.py --num-requests 8 --max-num-slots 4
```

输出类似：

```text
requests: 8
max_num_slots: 4
prefill_ms_p50: ...
decode_step_ms_p50: ...
ttft_ms_p50: ...
itl_ms_p50: ...
output_tokens_per_s: ...
avg_batch_size: ...
slot_utilization: ...
```

**不做什么（边界）**：

- 不追求生产压测精度。
- 不做 QPS/SLO 曲线。
- 不做 async server benchmark。
- 不和 vLLM 直接性能对比。
- 不要求一定达到 GPU 框架量级吞吐。

## 产出文件

- `inferlite/engine/metrics.py::RequestMetrics`  （作者手写）
- `inferlite/engine/metrics.py::StepMetrics`  （作者手写）
- `inferlite/engine/metrics.py::MetricsCollector`  （作者手写）
- `inferlite/engine/batch_core.py`  加 `metrics` 可选参数 + 各阶段埋点（作者手写）
- `scripts/bench_continuous_batching.py`  （AI 写，工程脚本）
- `tests/unit/test_metrics.py`  （AI 写，验证用）
- `bench/results/2026-xx-m3-continuous-batching-*.md`  （AI 写，结果归档）

## 算法核心

### 1. Request-level metrics

```python
@dataclass
class RequestMetrics:
    request_id: str
    arrival_ts: float
    scheduled_ts: float | None = None
    prefill_start_ts: float | None = None
    prefill_end_ts: float | None = None
    first_token_ts: float | None = None
    finished_ts: float | None = None

    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def queue_ms(self) -> float: ...

    @property
    def prefill_ms(self) -> float: ...

    @property
    def ttft_ms(self) -> float: ...

    @property
    def e2e_ms(self) -> float: ...
```

### 2. Step-level metrics

```python
@dataclass
class StepMetrics:
    step_idx: int
    batch_size: int
    max_seq_len: int
    decode_ms: float
    output_tokens: int
    running_count: int
    waiting_count: int
    occupied_slots: int
```

### 3. 汇总指标

```text
avg_batch_size = mean(step.batch_size)
slot_utilization = mean(step.occupied_slots / max_num_slots)
output_tokens_per_s = total_output_tokens / total_decode_wall_time
itl_ms = mean(inter_token_latency_per_request)
tpot_ms = total_decode_time / total_output_tokens
```

### 4. benchmark 对比组

建议至少两组：

```text
A. serial baseline: max_num_slots=1
B. continuous batching: max_num_slots=4 or 8
```

可选再加一组用于说明 static batching 问题的模拟 trace，但不需要实现完整 static batching engine。

## 接口契约（作者手写 spec）

### `inferlite/engine/metrics.py` 公开 API

```python
@dataclass
class RequestMetrics:
    """请求级时间戳 + 派生指标。所有时间戳用 time.perf_counter()，单位秒。"""

    request_id: str
    arrival_ts: float                       # submit 时记录
    scheduled_ts: float | None = None       # admit 到 running 时
    prefill_start_ts: float | None = None   # prefill forward 前
    prefill_end_ts: float | None = None      # prefill forward 后
    first_token_ts: float | None = None      # 第一个 token 采样后（只记一次）
    finished_ts: float | None = None         # 请求 finished

    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def queue_ms(self) -> float:
        """scheduled - arrival，未调度返回 0。"""
    @property
    def prefill_ms(self) -> float:
        """prefill_end - prefill_start。"""
    @property
    def ttft_ms(self) -> float:
        """first_token - arrival（包含 queue + prefill + 首 token 采样）。"""
    @property
    def decode_ms(self) -> float:
        """finished - first_token（decode 阶段总耗时）。"""
    @property
    def e2e_ms(self) -> float:
        """finished - arrival。"""
    @property
    def itl_ms(self) -> float:
        """decode_ms / (output_tokens - 1)，相邻 token 平均间隔。
        output_tokens <= 1 时返回 0（第一个 token 已计入 TTFT）。"""


@dataclass
class StepMetrics:
    """单个 decode step 的指标快照。"""
    step_idx: int
    batch_size: int          # 本轮 running 数（= forward 的 batch 维）
    max_seq_len: int         # 本轮最大 seq_len（cache_positions.max() + 1）
    decode_ms: float         # 本轮 decode forward wall clock
    output_tokens: int       # 本轮产出的 token 数（= batch_size）
    running_count: int      # = batch_size
    waiting_count: int      # 本轮 waiting 队列长度
    occupied_slots: int     # = running_count


@dataclass
class MetricsCollector:
    """采集 + 聚合。"""
    request_metrics: dict[str, RequestMetrics] = field(default_factory=dict)
    step_metrics: list[StepMetrics] = field(default_factory=list)
    max_num_slots: int = 0

    # ── 请求级采集（在 batch_generate 各阶段调用）──
    def record_arrival(self, request_id: str) -> None: ...
    def record_scheduled(self, request_id: str) -> None: ...
    def record_prefill_start(self, request_id: str) -> None: ...
    def record_prefill_end(self, request_id: str) -> None: ...
    def record_first_token(self, request_id: str) -> None:
        """只记一次：first_token_ts is None 才写。"""
    def record_finished(self, request_id: str) -> None: ...
    def record_prompt_tokens(self, request_id: str, n: int) -> None: ...
    def record_output_tokens(self, request_id: str, n: int) -> None: ...

    # ── 步级采集 ──
    def record_step(
        self,
        step_idx: int, batch_size: int, max_seq_len: int, decode_ms: float,
        output_tokens: int, running_count: int, waiting_count: int, occupied_slots: int,
    ) -> None: ...

    # ── 汇总（@property，延迟计算）──
    @property
    def avg_batch_size(self) -> float: ...          # mean(step.batch_size)
    @property
    def slot_utilization(self) -> float: ...        # mean(occupied_slots / max_num_slots)
    @property
    def total_decode_ms(self) -> float: ...         # sum(step.decode_ms)
    @property
    def total_output_tokens(self) -> int: ...       # sum(step.output_tokens)
    @property
    def output_tokens_per_s(self) -> float: ...    # total_output_tokens / (total_decode_ms/1000)
    @property
    def tpot_ms(self) -> float: ...                 # total_decode_ms / total_output_tokens
    @property
    def ttft_ms_p50(self) -> float: ...             # mean of all requests' ttft_ms（教学版用 mean 代替 p50）
    @property
    def itl_ms_p50(self) -> float: ...              # mean of all requests' itl_ms
    @property
    def prefill_ms_p50(self) -> float: ...          # mean of all requests' prefill_ms

    def summary(self) -> dict[str, float]:
        """返回所有汇总指标，benchmark 脚本用来输出。"""
```

### `batch_generate` 埋点位置

```python
def batch_generate(..., metrics: MetricsCollector | None = None) -> list[torch.Tensor]:
    # 提交阶段
    for i, prompt_ids in enumerate(prompts):
        req = RequestState(...)
        scheduler.submit(req)
        if metrics: metrics.record_arrival(req.request_id)
        if metrics: metrics.record_prompt_tokens(req.request_id, prompt_ids.shape[1])

    while scheduler.has_unfinished():
        # prefill 阶段
        admitted = scheduler.admit_until_full()
        for req in admitted:
            if metrics: metrics.record_scheduled(req.request_id)
            if metrics: metrics.record_prefill_start(req.request_id)
            logits = model(...)                    # prefill forward
            req.last_token = sampler(...)
            if metrics: metrics.record_prefill_end(req.request_id)
            if metrics: metrics.record_first_token(req.request_id)   # 只记一次

        # decode 阶段
        decode_start = time.perf_counter()
        logits = model(...)                         # batched decode forward
        decode_ms = (time.perf_counter() - decode_start) * 1000
        sampled = sampler(...)
        # 更新状态 + finish
        for req, tok in zip(running, sampled):
            ...
            if is_finished:
                if metrics: metrics.record_output_tokens(req.request_id, req.num_generated)
                if metrics: metrics.record_finished(req.request_id)
        # 记录本 step 指标
        if metrics: metrics.record_step(step_idx=..., batch_size=len(running), ...)
```

### 关键设计点

1. **时间戳用 `time.perf_counter()`**：单调递增、纳秒精度，不挡 MPS/CUDA 同步（教学版不做）。
2. **`record_first_token` 只记一次**：用 `if first_token_ts is None` 保护，后续调用不覆盖。
3. **`itl_ms` 分母是 `output_tokens - 1`**：第一个 token 已计入 TTFT，decode 阶段只产出 `n-1` 个间隔。
4. **`tpot_ms` vs `itl_ms`**：`tpot_ms = total_decode_ms / total_output_tokens`（整体每 token 成本），`itl_ms` 是请求内相邻 token 间隔（请求级）。
5. **p50 用 mean 代替**：教学版不引入 numpy/statistics 的 percentile，用 `fmean` 即可。

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|---|---|---|
| 1 | RequestMetrics 时间差计算 | 手工构造时间戳 | 精确 |
| 2 | first token 时间只记录一次 | 后续 token 不覆盖 | 精确 |
| 3 | StepMetrics batch_size | 等于本轮 running 数 | 精确 |
| 4 | avg_batch_size | 手工 mean | 精确 |
| 5 | slot_utilization | 手工 mean | 精确 |
| 6 | output_tokens_per_s | token/time 计算正确 | 浮点近似 |
| 7 | benchmark smoke | 脚本可运行并输出字段 | 精确 |
| 8 | num_requests/max_num_slots 参数 | 参数影响 trace | 精确 |

## DoD

- [ ] Request-level metrics 可记录 queue/prefill/TTFT/E2E。
- [ ] Step-level metrics 可记录 decode_ms/batch_size/slot 利用率。
- [ ] benchmark 脚本支持串行 baseline 和 M3 continuous batching 对比。
- [ ] benchmark 输出字段足以解释 M3 收益来源。
- [ ] `uv run pytest tests/unit/test_metrics.py -q` 通过。
- [ ] `uv run python scripts/bench_continuous_batching.py --num-requests 4 --max-num-slots 2` 可运行。
- [ ] commit `bench(engine): add continuous batching metrics and benchmark (M3-T6 done)`。

## 坑（按概率排序）

1. **把 prefill 时间算进 decode throughput**：M3 的收益主要在 decode batching，指标必须拆开。
2. **TTFT 定义不清**：建议从 arrival 到 first token ready。
3. **ITL 与 TPOT 混用**：ITL 是请求内相邻 token 间隔，TPOT 更偏整体 decode/token 平均成本。
4. **MPS/CUDA 同步问题**：计时前后需要必要的 device synchronize，否则时间偏小。
5. **benchmark 变成性能承诺**：M3 是教学版 fixed-slot PyTorch 实现，重点是趋势和可解释性。

## 完成总结

待完成后补：benchmark 结果表、核心指标解释、M3 相对串行的吞吐变化。
