# inferlite M4：PagedAttention 完整设计

| 字段 | 内容 |
|---|---|
| 状态 | ✅ 完成（tag: `m4/paged-attention`，2026-08-11） |
| 前置 | M3 tag `m3/continuous-batching` |
| 后续 | M5 Prefix Caching |
| 测试 | 270 tests 全绿 |

---

## 摘要

M3 用 fixed-slot KV Cache 跑通了 continuous batching，但每个请求独占 `max_seq_len` 连续物理空间，短请求浪费严重。M4 引入 PagedAttention：把 KV Cache 从连续数组改为虚拟内存式分页管理——请求的逻辑 KV 切成固定大小 block，通过 block table 映射到非连续物理 block。

**M4 的核心收获：block pool + block table + paged scatter/gather + NaN 安全 + ForwardContext/CacheAdapter 统一架构。**

---

## 符号说明

| 符号 | 含义 | M4 典型值 |
|---|---|---|
| `block_size` | 每个物理 block 容纳的 token 数 | 16 / 32 |
| `num_blocks` | 物理 block 总数 | 可配置 |
| logical block | 请求内部的逻辑 block 编号 | `pos // block_size` |
| physical block | KV 池中的实际 block id | `0..num_blocks-1` |
| `block_table` | logical block → physical block 的映射 | `list[int]` |
| `ref_count` | 物理 block 的引用计数 | ≥0 |

---

## 1. M3 → M4 的关键变化

| 维度 | M3 fixed-slot | M4 paged |
|---|---|---|
| 内存单位 | slot | block |
| 请求→物理内存 | `request_id → slot_id` | `request_id → block_table → block_id` |
| KV layout | `[S, H, L, D]` | `[num_blocks, block_size, H, D]` |
| seq_len | `seq_lens[slot]` | `block_table.seq_len` |
| 写入 | `cache.k[slot, :, pos, :] = k` | scatter 到物理 block |
| 读取 | gather slot 的 `[0:seq_len]` | 按 block_table gather 多个 block |
| 释放 | 释放整个 slot | 每个 block refcount--，为 0 才释放 |
| 碎片 | 短请求浪费 `max_seq_len - actual_len` | 最多浪费 `block_size - 1` 个 token |

---

## 2. 与 vLLM / nano-vLLM 的异同

inferlite M4 是 **纯 PyTorch 教学版 PagedAttention**，只取核心内存管理思想：

| 维度 | inferlite M4 | nano-vLLM | vLLM |
|---|---|---|---|
| Attention 实现 | PyTorch gather 伪版 | Triton kernel + FlashAttention | 生产 kernel |
| Prefix Cache | ❌（留 M5） | ✅ hash + LRU | ✅ |
| CoW | ❌（留 M5） | ✅ | ✅ |
| Scheduler | 简化 FCFS | token budget | 完整 preemption |
| 目标 | 理解机制 | 接近可跑 | 生产 serving |

**明确不做**：Triton kernel（M9）、Prefix Cache（M5）、Chunked Prefill（M10）、OpenAI API（M6）。

---

## 3. ADR 决策

| ADR | 决策 | 理由 |
|---|---|---|
| 新建 PagedKVCache | 不改 BatchedKVCache | M3 保留做 oracle，回滚简单 |
| PyTorch gather 伪版 | 不写 Triton kernel | Mac/MPS 友好，先理解机制 |
| `block_size=16` | 不用生产的 256 | 短 prompt 跨 block，单测更有效 |
| 保留 refcount | 不做 CoW/hash | M4 只聚焦分页，prefix 留 M5 |
| 不做 chunked prefill | 保留 full prefill | 分页机制本身先验证 |

---

## 4. 三层职责

PagedKVCache 由三个组件协作：

| 组件 | 数量 | 管什么 | 不管什么 |
|---|---:|---|---|
| `BlockPool` | 全局 1 个 | 物理 block 分配/释放/ref_count | tensor、请求顺序 |
| `BlockTable` | 每请求 1 个 | logical block → physical block + `seq_len` | 空闲块、tensor |
| `PagedKVCache` | 全局 1 个 | 每层 K/V tensor、scatter/gather | 调度、采样 |

依赖方向：`BlockPool` 和 `BlockTable` 互不认识，通过 `PagedKVCache` 中转 `block_id`。

物理寻址：

```
layer.k.shape = [num_blocks, block_size, n_kv_heads, head_dim]
                          ^          ^
                      block_id     offset
```

也可以展平：`slot = block_id * block_size + offset`。

---

## 5. 请求生命周期

```
prefill:
  allocate_request(id, prompt_len)  → 分配 ceil(prompt_len/block_size) 个 block
  write_prefill(layer, ids, k, v)  → scatter K/V 到物理 block

decode:
  append_token(id)                 → 如当前 block 已满，分配新 block；seq_len += 1
  write_decode(layer, ids, k, v)  → 写 1 个 token 到 pos = seq_len - 1

finish:
  free_request(id)                 → 归还所有 block，删除 block table
```

关键约束：decode 必须先 `append_token()` 再 `write_decode()`，否则新 K/V 会覆盖历史。

---

## 6. Scatter/Gather 数据流

### 6.1 Prefill scatter 写入

以 `block_size=4`，两个请求为例：

```
request a: block_ids=[3,1], seq_len=5
request b: block_ids=[2],   seq_len=2
```

**Step 1：生成 slot_mapping**

对每个请求，用 block_ids 广播生成所有容量位置的 slot：

```
a: block_ids[:, None] * 4 + offsets[None, :]
   = [[12,13,14,15], [4,5,6,7]]
   → flatten()[:5] = [12, 13, 14, 15, 4]

b: → flatten()[:2] = [8, 9]

slot_mapping = [12, 13, 14, 15, 4, 8, 9]   # request 优先、pos 递增
```

**Step 2：从 padded batch 提取 flat K/V**

输入 `[B, n_kv, T_max, D]`（B=2, T_max=5），用 `seq_lens=[5,2]` 做 valid mask，boolean index 得到：

```
flat_k = [A0, A1, A2, A3, A4, B0, B1]   # 与 slot_mapping 一一对应
```

**Step 3：scatter**

```python
flat_cache_k[slot_mapping] = flat_k
```

连续源数据按不连续目标下标分散写入。前提：`slot_mapping` 无重复 slot（BlockPool 独占分配保证）。

### 6.2 Decode 写入

每轮每个请求只写 1 个 token，`slot_mapping` 长度为 B：

```python
slot = last_block_id * block_size + ((seq_len - 1) % block_size)
```

### 6.3 Gather 读取

**批量 gather**：block table padding 成矩阵后一次高级索引：

```python
block_table = [[7, 2],     # a: 2 blocks
               [5, 0]]     # b: 1 block, 0 是占位 padding

layer.k[block_table]       # → [B, max_blocks, block_size, n_kv, D]
  → reshape → [B, n_kv, L_pad, D]
```

`L_pad = max_blocks * block_size`，不是 `max(seq_len)`。短请求的 padding 位置含垃圾值。

### 6.4 NaN 安全

物理 tensor 用 `torch.empty` 创建，未写入位置可能含 NaN。仅做 score mask 不够：`0 × NaN = NaN`。

必须先用 `valid_lens` 清零无效 K/V：

```python
invalid = ~(positions[None, :] < valid_lens[:, None])[:, None, :, None]
k = k.masked_fill(invalid, 0)
v = v.masked_fill(invalid, 0)
```

score mask 管 attention 语义，K/V 清零管数值安全——两者缺一不可。

---

## 7. T7 vLLM V1 架构对齐

M4 实现过程中（T7）引入了 vLLM V1 的核心架构模式，消除了 M3/M4 之间 80% 的重复代码：

### 7.1 ForwardContext

cache 和 metadata 不经过模型参数传递：

```python
# 初始化：cache 绑定到每层 Attention
adapter.bind_kv_cache(model)

# 每次 forward 前：metadata 通过全局上下文设置
with set_forward_context(metadata):
    logits = model(input_ids, positions=positions)   # 只有 2 个参数
```

### 7.2 CacheAdapter Protocol

3 种 cache 实现统一接口：

```
CacheAdapter(Protocol):
  can_admit(prompt_len) → bool        # 容量检查
  allocate(req_id, prompt_len)        # 分配 cache
  free(req_id)                        # 释放 cache
  bind_kv_cache(model)                # 绑定到 Attention 层
  make_prefill_metadata(input, pos)   # 构造 AttentionMetadata
  make_decode_metadata(tokens, pos)   # 构造 AttentionMetadata
  prepare_decode(request_ids)         # decode 前同步状态
  set_seq_lens(requests)              # prefill 后同步 seq_lens
```

Engine 代码只通过这个接口操作 cache，不关心底层是 slot 还是 block。

### 7.3 统一 loop.py

M3/M4 共享 `batch_generate_loop()`：

```
while scheduler.has_unfinished():
  1. Admit:     can_admit → admit → allocate（逐条交替）
  2. Prefill:   _build_prefill_batch → make_prefill_metadata → forward → sample
  3. Decode:    prepare_decode → make_decode_metadata → forward → sample → update
  4. Finish:    完成 → free → 移出 running
```

### 7.4 Attention 两层拆分

- `Qwen3Attention`：projection → QK-norm → RoPE → 委托 Attention → o_proj
- `Attention`：cache RW（isinstance 分发 M2/M3/M4）→ GQA → causal mask → softmax

---

## 8. 最终架构

```
engine/（~800L，4 文件）
├── context.py    ForwardContext + AttentionMetadata + LLMModel Protocol
├── engine.py     M1/M2/M3/M4 generate 统一入口
├── loop.py       统一 batch 主循环（admit → prefill → decode → free）
└── metrics.py    性能指标采集

cache/（~1,250L，5 文件）
├── adapter.py    CacheAdapter Protocol + M2/M3/M4 三种 adapter
├── kv_cache.py   M2 单序列 LayerKVCache
├── batched_kv_cache.py  M3 固定 slot BatchedKVCache
├── block_pool.py        物理 block 分配池
└── paged_kv_cache.py    M4 PagedKVCache + BlockTable

model/attention.py
├── _single_cache_rw()   M2 路径
├── _batched_cache_rw()  M3 路径（prefill scatter + decode + gather）
└── _paged_cache_rw()    M4 路径（scatter + gather + NaN 安全）
```

---

## 9. 修复的隐藏 bug

| bug | 根因 | 修复 |
|-----|------|------|
| slot_mapping 始终指向最后分配的 slot | 批量 admit 后才逐个 prefill，`_current_request_ids[-1]` 总返回最后请求 | 改为逐条 admit-allocate-prefill 交替 |
| `_admit` 超额分配（3 请求占 2 slot） | `can_admit()` 只检查空闲不占用 | 同上，每条请求分配后才检查下一条 |
| sampler 3D 输入崩溃 | `logits[:, -1:, :]` 是 3D `[B,1,V]`，sampler 返回 3D | 改为 `logits[:, -1, :]`（2D `[B,V]`） |
| Metrics `total_output_tokens=0` | loop.py 未调 `record_step()` | 补充 `record_step`/`record_output_tokens`/`record_finished` |
| padded prefill cache 写入越界 | `k[i]` 是 padded 长度 T=4，但 `:plen` 切片只有 3 | 改为 `k[i, :, :plen, :]` 只写有效位置 |
| padded prefill attention 读到 padding | 不同 prompt 长度的请求 pad 到 max_len，padding 位置参与 attention | 新增 `valid_lens` mask 屏蔽 padding 位置 |

---

## 10. 与后续里程碑关系

| 里程碑 | M4 提供什么 | M4 不含什么 |
|---|---|---|
| M5 Prefix Cache | block table + refcount 基础 | hash、LRU、CoW |
| M9 Triton kernel | PyTorch gather 伪版做正确性 oracle | Triton/CUDA kernel |
| M10 Chunked Prefill | block table 可表达长 prompt 分块写入 | token budget、mixed scheduling |
| M6 API/SSE | 可降低长请求并发内存浪费 | HTTP server |

---

## 11. 测试覆盖

270 tests 全绿，关键测试：

- `test_block_pool.py` — 分配/释放/ref_count/double-free
- `test_paged_attention.py` — scatter/gather 正确性、NaN 安全、跨 block
- `test_paged_batch_engine.py` — paged batch 生命周期、EOS、block 耗尽
- `test_batch_generate.py` — serial vs batch `torch.equal` 等价
- `test_real_qwen3_batch_matches_serial` — 真实 Qwen3 模型端到端等价
