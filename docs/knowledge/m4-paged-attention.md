# inferlite M4 技术设计：PagedAttention

| 字段 | 内容 |
|---|---|
| 状态 | 🟡 进行中（T1 BlockPool 已完成实现验证） |
| 作者 | luhao |
| 基于 | M3 tag `m3/continuous-batching` |
| 作战地图 | [M4.md](../plan/M4.md) |

---

## 摘要

M3 用 fixed-slot KV Cache 跑通了 continuous batching，但每个请求独占 `max_seq_len` 连续物理空间，短请求浪费严重，也无法表达 prefix 共享。M4 引入 PagedAttention：把每个请求的逻辑 KV 切成固定大小 block，通过 block table 映射到非连续物理 block。M4 不追求 vLLM/Triton 性能，只做纯 PyTorch 伪版，目标是把 block pool、block table 和按需分页讲清楚、测清楚。M4 保留 `ref_count` 基础能力，但不实现 CoW；Prefix Caching、LRU 和 partial-hit CoW 统一留到 M5。

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
| ref_count | 物理 block 的生命周期引用计数；M4 建立基础不变量 | ≥0 |
| CoW | Copy-on-Write，共享 block 写入前复制；M4 不做，M5 partial hit 再引入 | — |

---

## 1. 调研结论

### 1.1 vLLM PagedAttention

vLLM 的核心观察：KV Cache 又大又动态，传统连续内存管理会因碎片和过度预留浪费 60%–80% 显存。PagedAttention 借鉴操作系统分页，把每个 sequence 的 KV 切成固定大小 block，连续 logical block 通过 block table 映射到非连续 physical block。

关键结论：

- 物理 block 不要求连续。
- 内存按需分配，浪费只发生在最后一个 block。
- block table 具备表达共享 physical block 的能力，但 M4 不主动发现或复用跨请求前缀。
- vLLM V1 通过引用计数、free queue 与 prefix cache 管理共享生命周期；需要写入共享 partial block 时由上层 cache 管理执行 CoW。

### 1.2 nano-vllm 实现

本地 `nano-vllm` 的关键文件：

| 文件 | 作用 | M4 借鉴点 |
|---|---|---|
| `engine/block_manager.py` | `Block` / `BlockManager` / refcount / hash | block 分配、释放和 prefix hash 的参考实现 |
| `engine/sequence.py` | `Sequence.block_table`、`num_cached_tokens` | request 内部维护 block table |
| `engine/scheduler.py` | schedule prefill/decode，调用 block manager | allocate/may_append/postprocess 的时机 |
| `engine/model_runner.py` | `slot_mapping`、`block_tables` 构造 | input token 到 KV 物理槽位的映射 |
| `layers/attention.py` | `store_kvcache` + FlashAttention block_table | M9 kernel 参考，M4 只做 PyTorch 伪版 |

M4 不直接照搬 nano-vLLM，因为它依赖 CUDA/Triton/FlashAttention，并且已经包含 chunked prefill/prefix cache 逻辑。inferlite M4 只取 block pool、block table、slot mapping 和引用计数的核心抽象；hash、LRU 与 CoW 留到 M5。

---

## 2. 与 nano-vLLM / vLLM 的异同

M4 的定位可以概括为：

```text
vLLM
  = 生产级 PagedAttention + serving + scheduler + kernel + prefix cache + 多 GPU

nano-vLLM
  = 小型 CUDA 教学版 vLLM，保留较多生产关键路径

inferlite M4
  = 纯 PyTorch / Mac 友好的 PagedAttention 机制教学版
    只解决 block pool + block table + paged gather 正确性
    不包含 prefix cache 与 CoW
```

### 2.1 总体相同点

三者都采用 PagedAttention 的核心抽象：

| 核心机制 | inferlite M4 | nano-vLLM | vLLM |
|---|---:|---:|---:|
| KV Cache 切成固定大小 block | ✅ | ✅ | ✅ |
| 请求维护 block table | ✅ | ✅ | ✅ |
| logical block → physical block | ✅ | ✅ | ✅ |
| physical block 非连续 | ✅ | ✅ | ✅ |
| 按需分配 block | ✅ | ✅ | ✅ |
| refcount | ✅ | ✅ | ✅ |
| Copy-on-Write | ❌（M5） | prefix cache 场景需要 | ✅ |
| 目标：减少固定 `max_seq_len` 预留浪费 | ✅ | ✅ | ✅ |

所以 M4 学的是和 vLLM/nano-vLLM 同一个核心思想：**把 KV Cache 从连续数组改成虚拟内存式分页管理**。

### 2.2 inferlite M4 vs nano-vLLM

| 维度 | inferlite M4 | nano-vLLM |
|---|---|---|
| 运行设备 | CPU/MPS 友好，纯 PyTorch | 主要 CUDA |
| Attention 实现 | PyTorch gather 伪版：先 gather 成连续 KV，再走普通 attention | `store_kvcache` Triton kernel + FlashAttention block_table |
| 目标 | 先理解 block pool / block table / paged gather | 更接近可跑的高性能 vLLM 简化版 |
| Prefix Cache | M4 不做 hash prefix lookup，留 M5 | `BlockPool` 已有 `hash_to_block_id` / `compute_hash` |
| Chunked Prefill | 不做，留 M10 | scheduler 里已有 token budget / chunked prefill 逻辑 |
| Scheduler | 沿用 M3：逐条 prefill + batched decode | 已有 prefill/decode schedule，支持 `max_num_batched_tokens` |
| block_size | 倾向 16/32，方便教学和单测跨 block | 默认更偏生产，例如 256 |
| 性能预期 | 可能比 M3 更慢，接受 | 目标是接近简化 vLLM 性能 |
| 代码改动策略 | 新建 `PagedKVCache`，不破坏 M3 `BatchedKVCache` | 原生按 paged cache 设计 |

一句话：**nano-vLLM 已经是“小 vLLM”；inferlite M4 是“把 PagedAttention 拆开讲清楚的教学实现”**。

### 2.3 inferlite M4 vs vLLM

| 维度 | inferlite M4 | vLLM |
|---|---|---|
| 工程目标 | 教学、可读、可测 | 生产 serving |
| Attention kernel | PyTorch gather 伪版 | 高性能 PagedAttention kernel / FlashAttention / CUDA/Triton |
| KV 写入 | Python/PyTorch 写入 physical block | 自定义 kernel 写入 |
| Scheduler | 简化 FCFS + M3 continuous batching | token-budget scheduler、decode-first、chunked prefill、preemption |
| Prefix Cache | 不做，留 M5 | 支持 |
| Preemption | 不做 | 支持 |
| CUDA Graph | 不做 | 支持 |
| 多 GPU / TP | 不做 | 支持 |
| Memory profiling | 简化估算 | 完整 GPU memory profiling |
| 生产 API | 不做 | OpenAI-compatible serving |
| 性能目标 | 正确性优先 | 高吞吐/低延迟 |

一句话：**vLLM 是完整系统；inferlite M4 只取其中的 PagedAttention 内存管理内核思想**。

### 2.4 M4 为什么不直接照搬 nano-vLLM

M4 的学习目标不是“尽快做一个高性能 serving engine”，而是分离出 PagedAttention 的最小闭环：

```text
物理 block 池
  ↓
block table
  ↓
logical pos → physical block + offset
  ↓
按需分配
  ↓
引用计数生命周期
  ↓
paged gather 后 attention 正确
```

nano-vLLM 已经把多个后续主题混在一起：

```text
PagedAttention
+ Triton store_kvcache
+ FlashAttention block_table
+ chunked prefill
+ prefix hash cache
+ CUDA graph
+ TP
```

这些都很有价值，但如果 M4 一次全做，会把 M4 变成多个主题交错，违背当前规划原则：**同一个里程碑只做一个能力主题**。

### 2.5 M4 明确裁剪的能力

| 裁剪项 | 为什么不做 | 放到哪个里程碑 |
|---|---|---|
| Triton cache write kernel | M4 先学机制，不做 kernel 优化 | M9 |
| FlashAttention block table 原生调用 | 依赖 CUDA/FlashAttention，不适合 Mac/MPS 教学版 | M9 |
| Prefix Cache hash lookup | 属于 KV 复用策略，不是分页机制本身 | M5 |
| Chunked Prefill | 属于长上下文调度 | M10 |
| token-budget scheduler | 依赖 chunked prefill 和 mixed batch | M10+ |
| preemption | serving 策略，非 M4 核心 | 后续 |
| CUDA Graph | 工程性能优化 | Release/M9 |
| OpenAI API/SSE | 服务层 | M6 |

### 2.6 最重要的实现差异：M4 是“PagedAttention 伪版”

M4 的 attention 路径计划是：

```text
block_table
  ↓
按 block_id gather 出 K/V
  ↓
拼成临时连续 KV
  ↓
复用现有 GQA attention + per-row mask
```

vLLM/nano-vLLM 的路径是：

```text
block_table
  ↓
kernel 内按 block table 间接读 KV
  ↓
FlashAttention / PagedAttention kernel 直接算 attention
```

因此 M4 的完成标准不是“达到 nano-vLLM 性能”，而是：

- **概念对齐 vLLM/nano-vLLM**
- **数据结构部分对齐 nano-vLLM**
- **性能路径不对齐**
- **调度复杂度不对齐**
- **prefix/chunked prefill 不对齐**

---

## 3. M3 到 M4 的关键变化

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

## 4. ADR 决策

### ADR-01：M4 新建 `PagedKVCache`，不修改 `BatchedKVCache`

**Context**：M3 的 `BatchedKVCache` 已经稳定，用于 fixed-slot continuous batching。直接改会破坏 M3 回归。

**Decision**：新建 `inferlite/cache/paged_kv_cache.py`，提供 `PagedKVCache` / `PagedLayerKVCache` / `BlockTable`；`BlockPool` 保持在 `inferlite/cache/block_pool.py`。M3 代码保留，M4 通过可选参数或新入口启用 paged 路径。

**Consequences**：
- ✅ M3 fixed-slot 可作为 oracle 做正确性对比。
- ✅ 回滚简单。
- ❌ attention/batch_core 需要类型分派。

### ADR-02：M4 用 PyTorch gather 伪版，不写 Triton kernel

**Context**：vLLM/nano-vllm 使用 Triton/FlashAttention 直接按 block_table 读 KV；Mac/MPS 环境不适合。

**Decision**：M4 每步先按 block table gather 成临时连续 KV，再复用已有 attention 计算。T3 的完整写入/读取数据流见 [M4-T3 深入：PagedKVCache 数据流](m4-paged-kv-cache.md)。

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

### ADR-04：M4 保留 refcount，但不做 CoW 和 prefix hash lookup

**Context**：vLLM V1 的 beam search 在上层以独立请求实现，公共完整前缀由 prefix caching 自动复用；M4 尚未实现 prefix cache，因此没有真实共享写入场景。

**Decision**：M4 的 `BlockPool` 只提供 allocate/free/inc_ref/dec_ref 基础能力，不提供 `copy_on_write` 或 `hash_to_block_id`。M5 增加 hash、LRU 和 partial-hit CoW。

**Consequences**：
- ✅ M4 只聚焦分页分配与映射，边界更清楚。
- ✅ refcount 不变量可直接为 M5 复用。
- ❌ M4 不支持跨请求 prefix block 自动共享。

### ADR-05：M4 暂不做 chunked prefill

**Context**：nano-vllm scheduler 已有 token budget/chunked prefill；M4 目标是 L2 Memory，不是 L3 调度策略。

**Decision**：保留 M3 的逐条 prefill + batched decode，只有底层 KV 存储从 slot 改成 paged。

**Consequences**：
- ✅ 变量少，能清楚验证 paging 本身。
- ❌ 长 prompt prefill 阻塞 decode 的问题不解决，留 M10。

---

## 5. 数据流

> 本节是 M4 总览；T3 的 batch slot mapping、scatter/gather 和 NaN 安全细节见 [M4-T3 深入：PagedKVCache 数据流](m4-paged-kv-cache.md)。

### 5.1 prefill 写入

```text
request prompt length = 45, block_size = 16
需要 logical blocks: 0, 1, 2
block_table = [7, 3, 11]

pos 0..15   -> physical block 7,  offset 0..15
pos 16..31  -> physical block 3,  offset 0..15
pos 32..44  -> physical block 11, offset 0..12
```

### 5.2 decode 追加

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
=> BlockPool.allocate() 新物理 block，append 到 block_table
```

### 5.3 gather 读取

M4 伪版 attention 先 gather：

```python
full_k = gather_by_block_table(layer_cache.k, block_table, seq_len)
# full_k: [n_kv_heads, seq_len, head_dim]
```

batch 内多个请求 gather 后 padding 到同一个块对齐长度 `L_pad = max_num_blocks_in_batch * block_size`，再用 `valid_lens` 清零无效 K/V 并构造 per-row score mask。

---

## 6. 与后续里程碑关系

| 里程碑 | M4 提供什么 |
|---|---|
| M5 Prefix Cache | block table + refcount 基础；M5 增加 hash、LRU 与 partial-hit CoW |
| M9 Triton kernel | PyTorch gather 伪版提供正确性 oracle，Triton kernel 替换 gather/read/write |
| M10 Chunked Prefill | block table 可表达长 prompt 分块写入 |
| M6 API/SSE | M4 不是硬依赖，但可降低长请求并发时的内存浪费 |

---

## 7. 踩坑预案

| 坑 | 预防 |
|---|---|
| logical block 和 physical block 混用 | 所有变量命名显式带 `logical_` / `physical_` |
| offset off-by-one | 单测覆盖 block 边界：15/16/17、31/32/33 |
| free 时重复释放 block | refcount 为 0 才进 free list；double free 显式报错 |
| M4/M5 边界混淆 | M4 不实现 hash/LRU/CoW；M5 再增加共享与淘汰 |
| gather padding 读到垃圾或 NaN | score mask 保证语义；无效 K/V 尾部清零保证数值安全 |
| M3 回归被破坏 | fixed-slot 路径不动，M4 新类型分派 |
