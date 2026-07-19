# inferlite M4 技术设计：PagedAttention

> **状态**：⬜ 未开始
> **作者**：luhao
> **基于**：M3 tag `m3/continuous-batching`
> **作战地图**：[M4.md](../plan/M4.md)

---

## 摘要

M3 用 fixed-slot KV Cache 跑通了 continuous batching，但每个请求独占 `max_seq_len` 连续物理空间，短请求浪费严重，也无法表达 prefix 共享。M4 引入 PagedAttention：把每个请求的逻辑 KV 切成固定大小 block，通过 block table 映射到非连续物理 block。M4 不追求 vLLM/Triton 性能，只做纯 PyTorch 伪版，目标是把 block table、refcount、Copy-on-Write 机制讲清楚、测清楚，为 M5 Prefix Cache 和 M9 kernel 打基础。

---

## 符号说明

| 符号 | 含义 | M4 典型值 |
|---|---|---|
| block_size | 每个物理 block 容纳的 token 数 | 16 / 32 |
| num_blocks | 物理 block 总数 | 可配置 |
| logical_block | 请求内部的逻辑 block 编号 | `pos // block_size` |
| physical_block | KV 池中的实际 block id | `0..num_blocks-1` |
| block_offset | token 在 block 内的偏移 | `pos % block_size` |
| block_table | logical_block -> physical_block 的映射 | `list[int]` |
| ref_count | 物理 block 被多少请求引用 | ≥0 |
| CoW | Copy-on-Write，共享 block 写入前复制 | — |

---

## 1. 调研结论

### 1.1 vLLM PagedAttention

vLLM 的核心观察：KV Cache 又大又动态，传统连续内存管理会因碎片和过度预留浪费 60%–80% 显存。PagedAttention 借鉴操作系统分页，把每个 sequence 的 KV 切成固定大小 block，连续 logical block 通过 block table 映射到非连续 physical block。

关键结论：

- 物理 block 不要求连续。
- 内存按需分配，浪费只发生在最后一个 block。
- block table 使多个 sequence 可以共享同一 physical block。
- refcount + Copy-on-Write 保证共享安全。

### 1.2 nano-vllm 实现

本地 `nano-vllm` 的关键文件：

| 文件 | 作用 | M4 借鉴点 |
|---|---|---|
| `engine/block_manager.py` | `Block` / `BlockManager` / refcount / hash | block 分配、释放、refcount、prefix hash |
| `engine/sequence.py` | `Sequence.block_table`、`num_cached_tokens` | request 内部维护 block table |
| `engine/scheduler.py` | schedule prefill/decode，调用 block manager | allocate/may_append/postprocess 的时机 |
| `engine/model_runner.py` | `slot_mapping`、`block_tables` 构造 | input token 到 KV 物理槽位的映射 |
| `layers/attention.py` | `store_kvcache` + FlashAttention block_table | M9 kernel 参考，M4 只做 PyTorch 伪版 |

M4 不直接照搬 nano-vllm，因为它依赖 CUDA/Triton/FlashAttention，并且已经包含 chunked prefill/prefix cache 逻辑。inferlite M4 只取：block manager、block table、slot mapping、refcount/CoW 的核心抽象。

---

## 2. M3 到 M4 的关键变化

| 维度 | M3 fixed-slot | M4 paged |
|---|---|---|
| 内存单位 | slot | block |
| 请求到物理内存 | `request_id -> slot_id` | `request_id -> block_table -> block_id` |
| KV layout | `[S, H, L, D]` | `[num_blocks, block_size, H, D]` |
| seq_len | `seq_lens[slot]` | `block_table.seq_len` |
| 写入 | `cache.k[slot, :, pos, :] = k` | `cache.k[block_id, offset, :, :] = k` |
| 读取 | gather slot 的 `[0:seq_len]` | 按 block_table gather 多个 block |
| 释放 | 释放整个 slot | 每个 block refcount--，为 0 才释放 |

---

## 3. ADR 决策

### ADR-01：M4 新建 `PagedKVCache`，不修改 `BatchedKVCache`

**Context**：M3 的 `BatchedKVCache` 已经稳定，用于 fixed-slot continuous batching。直接改会破坏 M3 回归。

**Decision**：新建 `inferlite/model/paged_kv_cache.py`，提供 `PagedKVCache` / `PagedLayerKVCache` / `BlockManager` / `BlockTable`。M3 代码保留，M4 通过可选参数或新入口启用 paged 路径。

**Consequences**：
- ✅ M3 fixed-slot 可作为 oracle 做正确性对比。
- ✅ 回滚简单。
- ❌ attention/batch_core 需要类型分派。

### ADR-02：M4 用 PyTorch gather 伪版，不写 Triton kernel

**Context**：vLLM/nano-vllm 使用 Triton/FlashAttention 直接按 block_table 读 KV；Mac/MPS 环境不适合。

**Decision**：M4 每步先按 block table gather 成临时连续 KV，再复用已有 attention 计算。

**Consequences**：
- ✅ 可读、可测、设备兼容。
- ✅ logits/token 等价容易验证。
- ❌ 性能可能比 M3 更慢；接受，M9 再 kernel 化。

### ADR-03：block_size 默认选 16 或 32

**Context**：nano-vllm 默认 `block_size=256`，适合生产吞吐，但不利于单测覆盖跨 block 场景。

**Decision**：M4 默认 `block_size=16` 或 `32`，让短 prompt 也能跨 block，测试更容易发现映射 bug。

**Consequences**：
- ✅ 单测更有效。
- ❌ metadata 相对开销更大；教学版接受。

### ADR-04：M4 做 refcount/CoW，但不做 prefix hash lookup

**Context**：Prefix Cache 是 M5；但没有 refcount/CoW，M5 无法安全共享 block。

**Decision**：M4 的 `BlockManager` 提供 `inc_ref` / `dec_ref` / `copy_on_write` 能力；但不做 `hash_to_block_id` prefix lookup 策略。

**Consequences**：
- ✅ M5 可直接复用。
- ✅ M4 边界清楚。
- ❌ M4 中 CoW 只能通过人工构造共享 block table 测试。

### ADR-05：M4 暂不做 chunked prefill

**Context**：nano-vllm scheduler 已有 token budget/chunked prefill；M4 目标是 L2 Memory，不是 L3 调度策略。

**Decision**：保留 M3 的逐条 prefill + batched decode，只有底层 KV 存储从 slot 改成 paged。

**Consequences**：
- ✅ 变量少，能清楚验证 paging 本身。
- ❌ 长 prompt prefill 阻塞 decode 的问题不解决，留 M10。

---

## 4. 数据流

### 4.1 prefill 写入

```text
request prompt length = 45, block_size = 16
需要 logical blocks: 0, 1, 2
block_table = [7, 3, 11]

pos 0..15   -> physical block 7,  offset 0..15
pos 16..31  -> physical block 3,  offset 0..15
pos 32..44  -> physical block 11, offset 0..12
```

### 4.2 decode 追加

```text
seq_len = 45
new token position = 45
logical_block = 45 // 16 = 2
block_offset  = 45 % 16 = 13
physical_block = block_table[2] = 11
write k/v to block 11 offset 13
seq_len += 1
```

如果 `seq_len % block_size == 0`，说明需要新 block：

```text
seq_len = 48
append token at pos 48 -> logical_block 3
block_table 还没有 index 3
=> BlockManager.allocate() 新物理 block，append 到 block_table
```

### 4.3 gather 读取

M4 伪版 attention 先 gather：

```python
full_k = gather_by_block_table(layer_cache.k, block_table, seq_len)
# full_k: [n_kv_heads, seq_len, head_dim]
```

batch 内多个请求 gather 后 padding 到同一个 `max_seq_len_in_batch`，再用 per-row mask 保证每行只看自己的有效 KV。

---

## 5. 与后续里程碑关系

| 里程碑 | M4 提供什么 |
|---|---|
| M5 Prefix Cache | block table + refcount + CoW，使公共前缀 block 可以共享 |
| M9 Triton kernel | PyTorch gather 伪版提供正确性 oracle，Triton kernel 替换 gather/read/write |
| M10 Chunked Prefill | block table 可表达长 prompt 分块写入 |
| M6 API/SSE | M4 不是硬依赖，但可降低长请求并发时的内存浪费 |

---

## 6. 踩坑预案

| 坑 | 预防 |
|---|---|
| logical block 和 physical block 混用 | 所有变量命名显式带 `logical_` / `physical_` |
| offset off-by-one | 单测覆盖 block 边界：15/16/17、31/32/33 |
| free 时重复释放共享 block | refcount 为 0 才进 free list |
| CoW 后 block_table 未替换 | `copy_on_write(table, logical_idx)` 返回新 physical id 并写回 table |
| gather padding 读到垃圾 | per-row mask 必须按 seq_len 构造 |
| M3 回归被破坏 | fixed-slot 路径不动，M4 新类型分派 |
