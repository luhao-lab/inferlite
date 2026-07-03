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

- `inferlite/engine/metrics.py::RequestMetrics`
- `inferlite/engine/metrics.py::StepMetrics`
- `inferlite/engine/metrics.py::MetricsCollector`
- `benchmarks/bench_continuous_batching.py`
- `tests/unit/test_metrics.py`

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
- [ ] `uv run python benchmarks/bench_continuous_batching.py --num-requests 4 --max-num-slots 2` 可运行。
- [ ] commit `bench(engine): add continuous batching metrics and benchmark (M3-T6 done)`。

## 坑（按概率排序）

1. **把 prefill 时间算进 decode throughput**：M3 的收益主要在 decode batching，指标必须拆开。
2. **TTFT 定义不清**：建议从 arrival 到 first token ready。
3. **ITL 与 TPOT 混用**：ITL 是请求内相邻 token 间隔，TPOT 更偏整体 decode/token 平均成本。
4. **MPS/CUDA 同步问题**：计时前后需要必要的 device synchronize，否则时间偏小。
5. **benchmark 变成性能承诺**：M3 是教学版 fixed-slot PyTorch 实现，重点是趋势和可解释性。

## 完成总结

待完成后补：benchmark 结果表、核心指标解释、M3 相对串行的吞吐变化。
