# M3 — Continuous Batching 实现总结

> **完成日期**：2026-07-19
> **设计文档**：[M3.md](../plan/M3.md)
> **设计文档（详细 ADR）**：[m3-continuous-batching-design.md](m3-continuous-batching-design.md)
> **实测结果归档**：[bench/results/2026-07-18-m3-continuous-batching-mps-bf16.md](../../bench/results/2026-07-18-m3-continuous-batching-mps-bf16.md)

M3 在 `inferlite/` 中落地了完整的 continuous batching 教学版实现，共 7 个任务卡（T1–T7），全部完成。以下按任务卡逐项汇总，最后给出测试覆盖和局限性分析。

## 1. T1：调度状态机（RequestState + FCFSScheduler）

**交付**：纯 Python 调度状态机，维护请求生命周期。

- `RequestState`：封装 `request_id / prompt_ids / max_new_tokens / status / slot_id / seq_len / num_generated / last_token`。
- `RequestStatus`：`waiting / running / finished / cancelled` 四态。
- `FCFSScheduler`：维护四个集合，提供 `submit / admit_until_full / mark_finished / cancel`。

**关键设计结论**：

- `running` 用 `dict[str, RequestState]`，方便按 `request_id` 快速查找和移除。
- `admit_until_full()` 严格按 FCFS 顺序，把 `waiting` 中请求转入 `running`，直到 `len(running) == max_num_seqs`。
- 任何请求同一时刻只属于四队列之一，守恒由单测守护。

**不做**：优先级、抢占、超时、batching window、SLO-aware 策略、HTTP server、异步队列。

## 2. T2：固定槽位 KV Cache（BatchedKVCache + SlotManager）

**交付**：多请求固定槽位 KV 池，为 continuous batching 提供内存基础。

- `BatchedLayerKVCache`：单层 cache，shape `[max_num_slots, n_kv_heads, max_seq_len, head_dim]`。
- `SlotManager`：`deque` 管理 `free_slots`（popleft O(1)），`req_to_slot` 维护唯一映射。
- `BatchedKVCache`：组装多层 cache + `seq_lens` + `occupied` + `SlotManager`，提供 `from_config()` 工厂。

**关键设计结论**：

- `free` 时不清零 cache tensor，只清 `seq_lens[slot]=0` 和 `occupied[slot]=False`，下次 prefill 覆盖写入。
- slot 耗尽时 `raise RuntimeError`，`request_id` 重复时 `raise ValueError`，均为防御性检查。
- `max_num_slots` 同时是预分配槽位总数和最大 running 请求数。

**不做**：PagedAttention（留 M4）、block table / prefix cache / session cache / TTL / LRU / eviction。

## 3. T3：Batched Attention（per-row mask + cache_slots 分派）

**交付**：`GQAAttention.forward` 支持多请求 batched KV 读写。

**接口**：

```python
def forward(self, ..., layer_kv_cache, cache_position=0,
            cache_slots=None, cache_positions=None)
```

`Qwen3Model.forward` 透传 `cache_slots` / `cache_positions`。

**mask 语义**：

- M2 causal mask：`seq_len > 1` 时构建，防止看到未来。
- M3 per-row mask：`isinstance(BatchedLayerKVCache)` 时构建，每行只看自己有效 KV。

**等价性验证**：

- B=1 batched decode ≈ M2 single decode（atol=1e-4）。
- 混合 batch decode ≈ 逐条 sequential decode（atol=1e-4）。
- 165/165 全量回归通过，M2 路径不受影响。

## 4. T4：Batch Engine（batch_generate）

**交付**：`batch_generate()` 纯函数，M3 的主循环入口。

**主循环结构**：

1. **admit + prefill**：`admit_until_full()` 返回新请求，逐条 prefill（B=1 forward）。
2. **batched decode**：所有 running 请求组 batch，并行一步 decode。
3. **update + finish**：采样、更新状态、检查 max_new_tokens / EOS，finished 释放 slot。

**关键设计决策**：

| 决策 | 选择 | 理由 |
|---|---|---|
| 纯函数 vs 类 | 纯函数 | 简单，和 M2 generate 对称 |
| seq_len 语义 | next write position | 和 M2 cur_len 对齐，和 nano-vllm 一致 |
| 复用 EngineCore | 不复用 | step() 不支持 kv_cache/cache_slots，直接持有 model+sampler |
| prefill batching | 不做 | 教学范围限制，留 M10 Chunked Prefill |

## 5. T5：E2E 正确性验证

**交付**：12 个 E2E 测试，190/190 全量回归通过。

**串行 vs batch 语义等价**（`test_batch_generate.py`）：

- DeterministicModel：max_num_slots=1/2/4 三档，验证 token 级 `torch.equal`。
- 真实 Qwen3ForCausalLM：3 个不同长度 prompt，验证真实模型 token 级一致。
- 变长 prompt、EOS 早停、waiting>slots 全部完成。

**continuous batching trace**（`test_continuous_batching_trace.py`）：

- 不同 output 长度、slot 复用无 KV 污染、batch size trace。
- 非 static batching（finished 请求不锁住 wave）、waiting 不占 slot。
- EOS trace 验证 batch size 变化。

**核心结论**：M3 的所有改动（BatchedKVCache + prefill/decode 分派 + per-row mask + gather）**只有性能变化，语义完全不变**——serial generate 和 batch_generate 在 token 级别 `torch.equal`。

## 6. T6：指标体系与 Benchmark

**交付**：

- `RequestMetrics`：请求级时间戳 + 派生指标（queue_ms / prefill_ms / ttft_ms / decode_ms / itl_ms / total_ms）。
- `StepMetrics`：步级指标（batch_size / decode_ms / occupied_slots）。
- `MetricsCollector`：采集 + 聚合（avg_batch_size / slot_utilization / output_tokens_per_s / tpot_ms / ttft_ms_p50 / itl_ms_p50 / prefill_ms_p50）。
- `bench_continuous_batching.py`：对比 serial baseline 和 M3 continuous batching。
- 21 个单测覆盖所有指标 L0 项，211/211 全量回归通过。

Benchmark 结果详见 §7。

## 7. T7：文档与里程碑收口

**交付**：本文件创建、`PROGRESS.md` / `README.md` 状态同步、annotated tag `m3/continuous-batching`、所有任务卡完成总结补全。

## 8. 测试覆盖总览

| 测试文件 | 数量 | 覆盖范围 |
|---|---|---|
| `tests/unit/test_scheduler.py` | 8 | T1 四队列守恒 + admit/cancel/finish |
| `tests/unit/test_batched_kv_cache.py` | 10 | T2 slot 分配/释放/耗尽/重复 |
| `tests/unit/test_attention.py`（扩展） | — | T3 per-row mask 等价性 |
| `tests/unit/test_batch_engine.py` | 10 | T4 batch_generate 基础功能 |
| `tests/e2e/test_batch_generate.py` | 6 | T5 serial vs batch token 级等价 |
| `tests/e2e/test_continuous_batching_trace.py` | 6 | T5 continuous batching trace |
| `tests/unit/test_metrics.py` | 21 | T6 指标采集全量覆盖 |
| **全量回归** | **211/211** | M2 路径不受影响 |

## 9. 文件清单

| 文件 | 任务卡 | 类型 |
|---|---|---|
| `inferlite/scheduler/request.py` | T1 | 新建 |
| `inferlite/scheduler/fcfs.py` | T1 | 新建 |
| `inferlite/model/batched_kv_cache.py` | T2 | 新建 |
| `inferlite/model/attention.py` | T3/T5 | 修改 |
| `inferlite/model/qwen3.py` | T3/T5 | 修改 |
| `inferlite/engine/batch_core.py` | T4/T6 | 新建 |
| `inferlite/engine/protocol.py` | T4 | 修改 |
| `inferlite/engine/metrics.py` | T6 | 新建 |
| `scripts/bench_continuous_batching.py` | T6 | 新建 |
| `bench/results/2026-07-18-m3-continuous-batching-mps-bf16.md` | T6 | 新建 |

## 10. Benchmark 结果

详细结果见 `bench/results/2026-07-18-m3-continuous-batching-mps-bf16.md`，这里只保留关键结论：

- 主对比（`num_requests=4`, `max_num_slots=2`）：

  ```text
  serial throughput:  35.02 tok/s
  batch throughput:   15.29 tok/s
  speedup:            0.44x
  ```

- 消融（`num_requests=4`, `max_num_slots=1`）：

  ```text
  serial throughput:  35.40 tok/s
  batch throughput:   13.31 tok/s
  speedup:            0.38x
  ```

性能结论：

- 纯 PyTorch 教学版 M3 在 MPS 上 `batch_generate` 比 M2 serial 慢，主因不在 attention，而在 cache 读写路径。
- 分段 micro-benchmark 显示：

  - for 循环写 cache 约 63%
  - fancy index gather 约 22%
  - `.item()` 同步约 15%

- 这是"纯 PyTorch + 不调 kernel"的路线选择，不是 continuous batching 概念本身的问题。
- nano-vllm / vLLM 性能接近生产水平，是因为使用了 Triton `store_kvcache_kernel` + FlashAttention，而不是纯 PyTorch。
- 性能收益预计在 M4 PagedAttention 部分缓解，在 M9 Triton kernel 系统解决。

## 11. 已知局限性及后续解决路径

| M3 当前短板 | 根因 | 解决里程碑 | 具体方案 |
|---|---|---|---|
| cache 读写无向量化（for-loop write 63%，gather 22%） | 纯 PyTorch slice assign + fancy index，无自定义 kernel | **M9 核心算子加速**（阶段 1: cache write Triton kernel） | Triton kernel 替代 for-loop，batch 内 cache 写/读一次 kernel launch 完成 |
| prefill 没有 batch（逐条串行 prefill） | `batch_generate` 设计选择：prefill 逐条 B=1 forward | **M10 长上下文能力**（阶段 1: Chunked Prefill） | 长 prompt 切 chunk，与 decode 交替执行，支持 prefill/decode mixed batch |
| 无 PagedAttention（固定 slot 预分配，内存利用率低） | 教学简化：`[max_num_slots, n_kv_heads, max_seq_len, head_dim]` 全预分配 | **M4 PagedAttention** | block_table + 按需分配，消除内碎片 |
| 无 Prefix / Session Cache（相同前缀重复计算） | 不在 M3 范围 | **M5 Prefix Cache** | refcount + Copy-on-Write 复用公共前缀 KV |
| 无 HTTP / SSE 服务化 | M3 只解决进程内推理 | **M6 API + SSE** | FastAPI + SSE streaming + OpenAI-compatible endpoint |

## 12. 后续 M4 入口

M3 完成后的下一步是 M4 PagedAttention：

- 用 `block_table` 替代固定 slot 的连续物理 KV 分配。
- 用 refcount + Copy-on-Write 支持更灵活的 KV 复用。
- 在 M3 的 `BatchedKVCache` 与 attention 接口上扩展，而不破坏 continuous batching 语义。

后续 M5 在此基础上引入 Prefix Cache 与 reasoning 字段；M6 引入 API + SSE 服务化。
