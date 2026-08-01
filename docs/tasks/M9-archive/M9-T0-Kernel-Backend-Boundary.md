# M9-T0 — Kernel Backend 接入边界

> M9 的目标不是在 `GQAAttention.forward` 里新增一条 kernel if-else，而是把 M4 PyTorch gather backend 替换/旁路为 vLLM 风格 kernel backend。T0 先冻结 metadata、fallback 和 backend 选择边界。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M9-T0 |
| 里程碑 | M9 — 核心算子加速 |
| 状态 | ⬜ backlog |
| 前置 | M4-T7 Attention Backend Refactor ✅ |
| 后续 | M9-T1 Cache Write Kernel |
| 估时 | 2～3h |
| 核心文件 | 后续 `inferlite/model/attention.py`、kernel/backend 模块 |
| 产物 | M9 kernel backend 接入边界、metadata tensor 化清单、fallback 策略 |

## 背景

M4 的 PagedAttention 是 PyTorch 伪版：

```text
block table -> gather 成连续 K/V -> matmul attention -> Python mask
```

vLLM/nano-vLLM 的生产路径是：

```text
q + kv_cache + block_tables + seq_lens -> paged attention kernel -> output
```

M9 要做的是把底层执行 backend 换成 kernel，而不是重写模型 attention 主流程。

## 范围冻结

### 明确做

- 定义 `TorchPagedAttentionBackend` / `KernelPagedAttentionBackend` 或等价边界。
- 保留 M4 PyTorch gather backend 作为 correctness oracle 和 Mac/CPU/MPS fallback。
- 明确 kernel backend 需要的 tensor metadata：
  - `slot_mapping`
  - `block_tables`
  - `seq_lens` / `context_lens`
  - `query_start_loc` / `seq_start_loc`（如需要支持 mixed prefill/decode）
- 明确 cache write kernel 与 paged attention kernel 的输入输出合同。
- 明确 backend 选择策略：设备、可用依赖、配置开关。
- 准备 correctness 对齐：kernel backend 输出必须对齐 PyTorch fallback。

### 明确不做

- 不在 `GQAAttention.forward` 中直接写 kernel 分支。
- 不删除 PyTorch fallback。
- 不要求 Mac/MPS 跑 Triton 主线；Mac 只跑 fallback 或探索分支。
- 不在 T0 实现真实 kernel。
- 不把 M10 Chunked Prefill 混入 M9-T0。

## Metadata 清单

M9 需要把 M4/M5 的 Python 对象导出为 kernel-friendly tensor：

```text
request_ids / BlockTable / BlockPool
  ↓
slot_mapping: [num_tokens]
block_tables: [B, max_num_blocks]
seq_lens: [B]
context_lens: [B]
```

含义：

- `slot_mapping[token_idx] = physical_block_id * block_size + block_offset`，用于 cache write kernel。
- `block_tables[row, logical_block_idx] = physical_block_id`，用于 paged attention kernel 按逻辑位置间接读 KV。
- `seq_lens/context_lens` 用于 kernel 屏蔽无效 token，替代 Python `valid_lens` mask。

## Backend 关系

```text
GQAAttention.forward
  ↓
Attention backend boundary（M4-T7）
  ├── TorchPagedAttentionBackend   # M4 fallback/oracle
  └── KernelPagedAttentionBackend  # M9 主目标
```

要求：

- 两个 backend 的上层合同一致。
- 测试默认先跑 fallback，再在 GPU 环境跑 kernel 对齐。
- kernel backend 失败时能显式 fallback 或报清楚错误，不能静默给错结果。

## DoD

- [ ] M9 任务卡明确 kernel 通过 backend 接入，不污染 `GQAAttention.forward`。
- [ ] metadata tensor 化清单完整。
- [ ] PyTorch fallback 保留为 oracle。
- [ ] cache write kernel 与 paged attention kernel 的边界清楚。
- [ ] Mac/CPU/MPS fallback 与 NVIDIA GPU kernel 主线边界清楚。
