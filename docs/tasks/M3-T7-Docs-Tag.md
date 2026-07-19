# M3-T7 Docs + Tag

> M3 第七张任务卡：收口 M3 文档、进度表、benchmark 结果和里程碑 tag。

## 元信息
- **任务 ID**: M3-T7
- **里程碑**: M3 — Continuous Batching
- **状态**: ⬜ pending
- **前置**: M3-T6
- **估时**: 2h

## 目标

**要解决什么问题**：

M3 完成后，需要把代码实现、实验结果和技术结论沉淀成项目文档，避免只留下代码而没有学习闭环。

本卡要回答：

1. M3 到底实现了什么？
2. 为什么从 M2 到 M3 选择 continuous batching？
3. M3 fixed-slot KV Cache 和 M4 PagedAttention 的边界是什么？
4. benchmark 指标说明了什么？
5. 哪些内容明确留到 M4/M5？

**做完是什么效果**：

- `docs/plan/M3.md` 与最终实现一致。
- `docs/plan/PROGRESS.md` 更新 M3 状态、日期、tag。
- README 里 M3 链接和状态正确。
- 创建有意义的 annotated tag：

```text
m3/continuous-batching
```

**不做什么（边界）**：

- 不再补新功能。
- 不再大改调度设计。
- 不引入 M4 PagedAttention 实现。
- 不写代码仓库外部的学习总结。

## 产出文件

- `docs/plan/M3.md`
- `docs/plan/PROGRESS.md`
- `README.md`
- 必要时新增：`docs/benchmarks/M3.md` 或在 `docs/plan/M3.md` 内补 benchmark 结果
- Git tag: `m3/continuous-batching`

## 算法核心

本卡不是算法实现，而是文档收口。建议按以下结构检查：

### 1. 技术结论

```text
M2: single request + static KV cache
M3: multi request + fixed-slot KV cache + decode continuous batching
M4: paged KV cache / PagedAttention
M5: prefix/session cache
```

### 2. M3 完成定义

```text
- 支持多个请求提交。
- prefill 逐条执行。
- decode 每 iteration 重新组 batch。
- finished 请求立即释放 slot。
- waiting 请求在下一轮进入。
- E2E correctness 等价串行 generate。
- 输出 metrics/benchmark。
```

### 3. benchmark 结果格式

```markdown
| 模式 | 请求数 | max_num_slots | output tokens/s | avg batch size | slot util | TTFT p50 | ITL p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| serial | 8 | 1 | ... | ... | ... | ... | ... |
| M3 continuous | 8 | 4 | ... | ... | ... | ... | ... |
```

### 4. tag message 建议

```text
M3 连续批处理完成

- RequestState + FCFS scheduler：waiting/running/finished lifecycle
- BatchedKVCache：fixed-slot multi-request KV pool
- Batched decode attention：cache_slots/cache_positions + per-row mask
- BatchEngine：prefill one-by-one, decode continuous batching
- E2E correctness：continuous batching matches serial semantics
- Metrics/benchmark：prefill/decode/TTFT/ITL/batch size/slot utilization
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|---|---|---|
| 1 | M3 任务卡状态 | T1-T7 均 done | 精确 |
| 2 | `docs/plan/M3.md` 与实现一致 | 不含过期设计 | 人工检查 |
| 3 | `PROGRESS.md` M3 行 | status/date/tag 正确 | 精确 |
| 4 | README 链接 | 不存在死链 | 精确 |
| 5 | benchmark 表 | 数字来自实际脚本输出 | 精确 |
| 6 | 全量测试 | `uv run pytest` 通过 | 精确 |
| 7 | tag 名称 | `m3/continuous-batching` | 精确 |

## DoD

- [ ] M3 所有任务卡完成总结已补齐。
- [ ] `docs/plan/M3.md` 更新最终实现与 benchmark 结果。
- [ ] `docs/plan/PROGRESS.md` 更新 M3 状态。
- [ ] README 更新 M3 状态和链接。
- [ ] 全量测试通过。
- [ ] 创建 annotated tag `m3/continuous-batching`。
- [ ] tag push 到远端。
- [ ] commit `docs: finalize M3 continuous batching milestone`。

## 坑（按概率排序）

1. **文档提前宣称 M4 能力**：M3 没有 PagedAttention、prefix cache、eviction。
2. **benchmark 数字没有来源**：必须来自实际脚本输出，不能手填预期值。
3. **README/PROGRESS tag 不一致**：M2 已用 `m2/static-kv-cache`，M3 建议用 `m3/continuous-batching`。
4. **总结文档放错位置**：项目学习总结遵守工作区规范，代码仓库内只放项目自身文档。
5. **tag 打在未 commit 的工作区**：先确认 `git status` clean。

## 完成总结

本卡完成 M3 的文档与里程碑收口：

- 任务卡总结：补齐 T1 / T2 / T7 的 `## 完成总结`，记录调度状态机、固定槽位 KV Cache 和本文档闭环的最终结论。
- 技术结论：M3 实现教学版 continuous batching，固定槽位 KV Cache，decode 每 iteration 重组 batch；PagedAttention / Prefix Cache / Chunked Prefill 明确留给 M4/M5/M10。
- Benchmark 结果：归档至 `bench/results/2026-07-18-m3-continuous-batching-mps-bf16.md`，并在 `docs/plan/M3.md` 中给出主对比、消融、瓶颈拆解和与 nano-vllm 的差异说明。
- 进度与入口：`docs/plan/PROGRESS.md` 与 `README.md` 同步 M3 完成状态，后续 M4 入口在 PLAN/M3 中明确说明。
- Tag：创建 annotated tag `m3/continuous-batching`，message 概括 M3 的 scheduler / cache / attention / engine / metrics 落地点。

待全部收口后回填：最终 commit、tag 创建时间、PROGRESS/README 中使用的具体日期与链接。
