# M4-T3 深入：PagedKVCache 数据流

> T3 不是再写一个 KV Cache 容器，而是把 T1 的物理 block 编号、T2 的逻辑映射和真实 K/V tensor 连接起来。

## 元信息

| 字段 | 内容 |
|---|---|
| 所属里程碑 | M4 — PagedAttention |
| 对应任务 | [M4-T3 — PagedKVCache](../tasks/M4-archive/M4-T3-PagedKVCache.md) |
| 核心代码 | `inferlite/cache/paged_kv_cache.py` |
| 状态 | ✅ T3 done，M4 整体仍进行中 |
| 边界 | 不接 attention / engine；T4/T5 继续推进 |

## 1. 要解决的内存问题

M3 fixed-slot continuous batching 的每层 K/V shape 是：

```text
[max_num_slots, n_kv_heads, max_seq_len, head_dim]
```

这意味着每个请求一旦被 admit，就占用一整段 `max_seq_len` 空间。短请求实际只用几十个 token，也会占着几千个位置。

M4 paged cache 改为全局物理 block 池：

```text
[num_blocks, block_size, n_kv_heads, head_dim]
```

请求只按需拿 block。逻辑上连续的 token 序列，由 `BlockTable` 映射到非连续物理 block：

```text
请求逻辑位置 pos: 0, 1, 2, ...
        |
        v
BlockTable: pos -> (physical_block_id, block_offset)
        |
        v
PagedKVCache: layer.k[physical_block_id, block_offset]
```

因此 PagedKVCache 的核心不是“更大的 tensor”，而是：**把逻辑连续序列恢复到非连续物理存储上**。

## 2. 三层职责

| 组件 | 数量 | 管什么 | 不管什么 |
|---|---:|---|---|
| `BlockPool` | 全局 1 个 | 物理 block id 的分配、释放、引用计数 | tensor、请求顺序、token 位置 |
| `BlockTable` | 每请求 1 个 | 该请求的 logical block -> physical block 顺序映射，以及 `seq_len` | 空闲块、tensor、device/dtype |
| `PagedKVCache` | 全局 1 个 | 每层 K/V tensor、请求表注册、batch scatter/gather | 调度策略、采样、attention 数学 |

依赖方向是单向的：

```text
BlockPool ──────────────┐
                        v
BlockTable ───────> PagedKVCache ─────> T4 PagedAttention
                        ^
ModelConfig / torch ────┘
```

`BlockPool` 和 `BlockTable` 互不认识。它们通过 `PagedKVCache` 中转 `block_id`：

```text
pool.allocate() -> block_id -> table.append_block(block_id)
```

## 3. `block_id` 与真实 tensor 地址

`block_id` 不是裸内存地址，只是物理 block 的整数编号。真正发生寻址的是 T3：

```text
layer.k.shape = [num_blocks, block_size, n_kv_heads, head_dim]
                         ^          ^
                     block_id     offset
```

也可以把前两维展平：

```text
slot = block_id * block_size + offset

layer.k[block_id, offset, :, :]
    等价于
layer.k.view(num_blocks * block_size, n_kv_heads, head_dim)[slot, :, :]
```

例如 `block_size=4`：

| 原坐标 | 扁平 slot |
|---|---:|
| `layer.k[0, 0]` | 0 |
| `layer.k[0, 3]` | 3 |
| `layer.k[1, 0]` | 4 |
| `layer.k[3, 0]` | 12 |

这个扁平 slot 就是 batch 写入时的 `slot_mapping`。

## 4. 请求生命周期

T3 把“生命周期状态变化”和“大 tensor 写入”分开：

```text
prefill:
  allocate_request(request_id, prompt_len)
      - 分配 ceil(prompt_len / block_size) 个物理 block
      - BlockTable.seq_len = prompt_len
  write_prefill(layer_idx, request_ids, k, v)
      - 只写 K/V，不再修改 seq_len

decode:
  append_token(request_id)
      - 如当前容量已满，分配新 block
      - BlockTable.seq_len += 1
  write_decode(layer_idx, request_ids, k, v)
      - 写刚 append 的那个 token，即 pos = seq_len - 1

finish:
  free_request(request_id)
      - 归还该请求持有的所有 block
      - 删除 block table
```

为什么生命周期仍是单请求 API？因为这是 scheduler 的职责：谁进入、谁 decode、谁结束，本来就是逐请求决策。相反，K/V tensor 搬运是大数组操作，所以必须是 batch scatter / gather。

## 5. Batch Prefill 写入完整走查

设定：

```text
block_size = 4

request a:
  block_ids = [3, 1]
  seq_len = 5

request b:
  block_ids = [2]
  seq_len = 2
```

### 5.1 从 block table 到 slot mapping

对请求 a：

```text
block_ids = [3, 1]
offsets   = [0, 1, 2, 3]

block_ids[:, None] * 4 = [[12],
                          [ 4]]

offsets[None, :]       = [[0, 1, 2, 3]]

slots = block_ids[:, None] * 4 + offsets[None, :]
      = [[12, 13, 14, 15],
         [ 4,  5,  6,  7]]
```

二维结果表示每个已分配容量位置的物理 slot。请求 a 只实际写入 5 个 token，因此：

```text
slots.flatten()[:5] = [12, 13, 14, 15, 4]
```

对应表：

| 逻辑 pos | logical block | offset | physical block | slot |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 3 | 12 |
| 1 | 0 | 1 | 3 | 13 |
| 2 | 0 | 2 | 3 | 14 |
| 3 | 0 | 3 | 3 | 15 |
| 4 | 1 | 0 | 1 | 4 |

请求 b：

```text
block_ids = [2]
seq_len = 2
slots.flatten()[:2] = [8, 9]
```

按 `request_ids=["a", "b"]` 拼起来：

```text
slot_mapping = [12, 13, 14, 15, 4, 8, 9]
```

排序合同非常重要：**request 优先、request 内 pos 递增**。后面的 `flat_k/v` 必须使用完全相同的顺序。

### 5.2 从 padded batch 到 flat K/V

输入仍是 M3 风格 padded batch：

```text
k/v shape = [B, n_kv_heads, T_max, head_dim]
B = 2, T_max = 5
```

逻辑内容：

```text
a: A0 A1 A2 A3 A4
b: B0 B1 PAD PAD PAD
```

用 `seq_lens=[5, 2]` 生成有效 mask：

```text
valid = [[T, T, T, T, T],
         [T, T, F, F, F]]
```

先把 K/V 转成 `[B, T_max, n_kv, D]`，再 boolean index：

```text
flat_k = [A0, A1, A2, A3, A4, B0, B1]
```

它与 `slot_mapping` 一一对应：

| flat 行 | token | slot |
|---:|---|---:|
| 0 | A0 | 12 |
| 1 | A1 | 13 |
| 2 | A2 | 14 |
| 3 | A3 | 15 |
| 4 | A4 | 4 |
| 5 | B0 | 8 |
| 6 | B1 | 9 |

### 5.3 scatter：连续输入写到不连续物理位置

```text
flat_cache_k[slot_mapping] = flat_k
```

等价于：

```text
flat_cache_k[12] = A0
flat_cache_k[13] = A1
flat_cache_k[14] = A2
flat_cache_k[15] = A3
flat_cache_k[4]  = A4
flat_cache_k[8]  = B0
flat_cache_k[9]  = B1
```

这就是 scatter：把连续源数据按不连续目标下标分散写入。它与 gather 方向相反。

M4 能这样写的前提是 `slot_mapping` 中没有重复 slot：BlockPool 独占分配 + batch 内拒绝重复 request id 共同保证这一点。M5 prefix cache 共享 block 后，如果要写共享块，必须先 CoW。

## 6. Batch Decode 写入

Decode 与 prefill 的生命周期不同：

```text
append 前 seq_len = 5
append_token()
append 后 seq_len = 6
刚新增 token 的逻辑 pos = seq_len - 1 = 5
```

所以 decode slot 使用：

```text
slot = last_block_id * block_size + ((seq_len - 1) % block_size)
```

例子：`block_size=4`。

| 请求 | append 前 | append 后 | block 行为 | 新 token offset |
|---|---:|---:|---|---:|
| a | 4 | 5 | 原块已满，分新 block | 0 |
| b | 3 | 4 | 原块还剩一位 | 3 |

`write_decode` 的输入是：

```text
k/v shape = [B, n_kv_heads, 1, head_dim]
```

每个请求本轮只有一个新 token，因此 `slot_mapping` 长度为 B，scatter 写入 B 行。

关键约束：调用方必须先对 batch 中每个 request 调 `append_token()`，再调用 `write_decode()`。否则 decode 会把新 K/V 写到旧 token 位置，静默覆盖历史。

## 7. Gather 读取

### 7.1 单请求 oracle

如果请求的 block table 是：

```text
block_ids = [7, 2, 5]
seq_len = 10
block_size = 4
```

物理访问顺序是 `[7, 2, 5]`，不能用连续切片 `layer.k[7:10]`。`gather_kv_single` 使用高级索引：

```python
layer.k[torch.tensor([7, 2, 5])]
```

PyTorch 会按索引列表顺序取出 block：

```text
[物理 block 7, 物理 block 2, 物理 block 5]
```

再 reshape 成逻辑连续序列并截断到 `seq_len`。它不走生产路径，主要用作 batch gather 的测试 oracle。

### 7.2 批量 gather

Batch gather 要让多请求拥有相同 tensor shape，因此先把 block table padding 成矩阵：

```text
a: [7, 2]
b: [5]

block_table = [[7, 2],
               [5, 0]]
```

其中 `0` 只是占位，属于短请求的无效 padding。

一次高级索引：

```python
layer.k[block_table]
```

得到：

```text
[B, max_num_blocks, block_size, n_kv, D]
```

再合并 block 维和 offset 维，转成 attention 需要的布局：

```text
[B, max_num_blocks * block_size, n_kv, D]
  -> [B, n_kv, L_pad, D]
```

注意 `L_pad = max_num_blocks * block_size`，不是 `max(seq_len)`。T4 只能通过 `valid_lens` 判断每行真实有效长度。

## 8. NaN 数值安全闭环

PagedKVCache 用 `torch.empty` 创建物理 K/V tensor。未写入位置可能含 NaN / Inf。

Batch gather 会读出 block 对齐 padding：

```text
a: [A0, A1, A2, A3, A4, PAD, PAD, PAD]
b: [B0, B1, PAD, PAD, PAD, PAD, PAD, PAD]
```

仅做 score mask 不够。即使 padding 位置 attention probability 为 0，value 聚合中仍有：

```text
0 × NaN = NaN
```

因此 T3 返回：

```text
valid_lens = [5, 2]
```

T4 必须据此先清零无效 K/V，再做 score mask：

```python
positions = torch.arange(k.shape[2], device=k.device)
valid = positions[None, :] < valid_lens[:, None]
invalid = ~valid[:, None, :, None]
k = k.masked_fill(invalid, 0)
v = v.masked_fill(invalid, 0)
```

这延续了 [lessons L5](lessons.md#l5-score-mask-不能隔离未初始化-v-padding-中的-nan) 的结论：score mask 负责 attention 语义，K/V 清零负责数值安全。

## 9. 与 vLLM V1 的对应关系

| vLLM V1 | inferlite M4-T3 |
|---|---|
| runner 将本轮 token pack 成 `[total_tokens, ...]` | M4 从 `[B, T_max, ...]` 通过 mask 得到 `flat_k/v` |
| `slot_mapping: [total_tokens]` | `_make_prefill_slot_mapping` / `_make_decode_slot_mapping` |
| CUDA/Triton `reshape_and_cache` kernel | PyTorch `flat_cache[slot_mapping] = flat_k/v` |
| kernel 直接按 block table 读 KV | M4 先 batch gather 成连续 K/V，再复用标准 attention |
| 无 padding 物化 | M4 gather 产生 `L_pad`，靠 `valid_lens` 管理 |

M4 的目标不是复刻 vLLM kernel，而是复刻它的数据流和正确性合同。性能优化留给 M9。

## 10. 复杂度与边界

| 路径 | M4 实现 | 后续演进 |
|---|---|---|
| slot mapping 生成 | 按请求循环，张量广播生成每请求 slots | M9 kernel / 更深向量化 |
| prefill 写入 | boolean mask flatten + 一次 scatter | packed input 后可省去 padding mask |
| decode 写入 | B 个 slot + 一次 scatter | kernel scatter |
| gather 读取 | batch block table 高级索引，物化连续 K/V | PagedAttention kernel 直接读 block table |
| prefix 共享写入 | 不支持，M4 block 独占 | M5 CoW |
| chunked prefill | 不支持 full prefill 的 `seq_len == prompt_len` 假设 | M10 `write_tokens(start, len)` |

T3 完成后，M4 仍未完成：T4 需要把 `PagedKVCache.gather_kv` 接入 attention，T5 需要把 scheduler / engine 的 admission 与请求生命周期接到 paged cache。
