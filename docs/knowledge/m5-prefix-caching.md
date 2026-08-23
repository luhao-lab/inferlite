> 📖 **主线入口**：本文是 M5 的专项设计文档。如果想一条线理解 M1→M5 的 KV Cache 完整演进，请先读 [KV Cache 演进](./kv-cache-evolution.md)。

# inferlite M5：Prefix Caching 完整设计

| 字段 | 内容 |
|---|---|
| 状态 | ✅ 完成（tag: `m5/prefix-caching`，2026-08-19） |
| 前置 | M4 tag `m4/paged-attention` |
| 后续 | M6 MoE 教学版 |
| 测试 | 314 tests 全绿（+44 M5 新增） |

---

## 摘要

M4 用 block 分页解决了 KV Cache 的内存碎片问题，但每个请求的 block 都是独立分配的——即使两个请求有完全相同的 prompt 前缀，也会各自计算一遍 KV。M5 引入 prefix caching：对填满的 block 计算 chain hash，新请求到来时查 hash 命中，直接复用已缓存的 block 和对应的 KV 数据。

**M5 的核心收获：chain hash + LRU 淘汰 + cache-aware allocate + CoW + hash_blocks 注册。**

---

## 符号说明

| 符号 | 含义 | M5 典型值 |
|---|---|---|
| `block_size` | 每个物理 block 容纳的 token 数 | 4 / 16 |
| `hash` | chain hash 值（xxhash64） | int64 |
| `num_cached` | prefix cache 命中的满 block 数 | 0 ~ N |
| `ref_count` | block 的引用计数（持有该 block 的请求数） | ≥0 |
| `cached_block_lru` | ref=0 且有 hash 的 block（OrderedDict） | LRU 队列 |
| `hash_to_block_id` | chain hash → block_id 的索引 | dict[int, int] |

---

## 1. M4 → M5 的关键变化

| 维度 | M4 paged | M5 prefix caching |
|---|---|---|
| Block 元数据 | `Block(block_id, ref_count)` | `Block(block_id, ref_count, hash, token_ids)` |
| 空闲 block 管理 | `free_block_ids: deque` | `free_block_ids` + `cached_block_lru` + `hash_to_block_id` |
| 释放策略 | ref=0 → 直接归还 free pool | ref=0 且有 hash → LRU 队列；无 hash → free pool |
| 分配策略 | `allocate()` 只从 free pool 取 | `allocate()` 先 free pool，不够再淘汰 LRU |
| 新请求分配 | `allocate_request(id, prompt_len)` | `can_admit_with_cache(ids)` → `allocate_with_cache(ids, num_cached)` |
| hash 注册 | 无 | `hash_blocks()` 在 prefill/decode 后注册 chain hash |
| 共享写入 | 不存在（block 独占） | `cow_if_shared()` Copy-on-Write |

---

## 2. 与 vLLM V1 的简化对照

| 维度 | inferlite M5 | vLLM V1 |
|---|---|---|
| hash 索引 | `dict[int, int]`（1:1） | `BlockHashToBlockMap`（1:N 树结构） |
| LRU 队列 | `OrderedDict` | `FreeKVCacheBlockQueue`（双向链表） |
| hash 粒度 | 只对满 block hash | 支持 partial block fine-grained hash |
| CoW | 同步（写入前立即 clone） | 异步 `_pending_cow_copies` 批量执行 |
| 孤儿 block | 不处理（LRU 自然淘汰） | 树结构级联淘汰 |
| skip prefill | 未实现（defer） | `num_computed_tokens` 跳过已缓存 token |

---

## 3. ADR 决策

| ADR | 决策 | 理由 |
|---|---|---|
| OrderedDict LRU | `collections.OrderedDict` | O(1) touch + popitem，代码最简 |
| chain hash | `hash = xxhash64(prefix_hash ‖ token_ids)` | 位置唯一性，天然支持前缀匹配 |
| 同步 CoW | 写入前立即 clone | 教学版不需要异步批量优化 |
| CacheAdapter 接入 | prefix cache 逻辑在 BlockPool/Adapter/loop | attention 主流程不变 |
| 孤儿 block 不处理 | LRU 自然淘汰 | 不需要树结构，实现简单 |
| skip prefill defer | 先保证正确性 | 优化作为后续改进 |

---

## 4. BlockPool 三容器架构

M5 的 BlockPool 从 M4 的单队列升级为三容器：

```text
BlockPool (~308L)
├── free_block_ids: deque[int]         # 真正空闲（ref=0, hash=-1）
├── cached_block_lru: OrderedDict[int, None]  # ref=0, 有 hash（LRU 队尾=最近）
└── hash_to_block_id: dict[int, int]   # chain hash → block_id 索引
```

block 的生命周期在三容器间流转：

```text
                    allocate()              free() hash≠-1
free_pool ──────► running (ref≥1) ──────► cached_lru
    ▲                 │                      │
    │                 │ free() hash=-1       │ allocate() 淘汰
    │                 ▼                      │
    └─────────── free_pool ◄─────────────────┘
```

关键规则：
- **allocate() 优先级**：free pool → LRU 淘汰（popitem front）→ error
- **touch()**：LRU 中的 block 被命中时，ref++ + 从 LRU 移除
- **reset_hash()**：淘汰 LRU block 时清除 hash 和 hash_to_block_id 映射

---

## 5. Chain Hash 机制

chain hash 是 prefix cache 的核心：每个满 block 的 hash 依赖于前面所有 block 的 hash，形成链式结构。

```text
block 0: h₀ = xxhash64(tokens[0:bs], prefix=-1)
block 1: h₁ = xxhash64(tokens[bs:2bs], prefix=h₀)
block 2: h₂ = xxhash64(tokens[2bs:3bs], prefix=h₁)
```

**位置唯一性**：相同 token 序列在不同位置产生不同 hash。`[1,2,3,4]` 在 block 0 和 block 1 的 hash 不同，因为 prefix hash 不同。

**前缀匹配**：chain hash 天然支持前缀匹配——如果 block 0 的 hash 匹配，block 1 才有可能匹配；block 0 不匹配则停止。

**只对满 block hash**：部分填充的 block 不注册 hash，因为 token_ids 还不完整。

实现（`block_pool.py`）：

```python
def compute_hash(self, token_ids: list[int], prefix_hash: int) -> int:
    h = xxhash.xxh64()
    h.update(prefix_hash.to_bytes(8, "little", signed=True))
    h.update(struct.pack(f"{len(token_ids)}i", *token_ids))
    return h.intdigest()
```

---

## 6. LRU 淘汰策略

M5 的淘汰分两类：

### 6.1 自然淘汰（allocate 时）

当 free pool 为空但 LRU 非空时，淘汰 LRU 队首（最久未用的 cached block）：

```python
def allocate(self) -> int:
    if self.free_block_ids:
        return self.free_block_ids.popleft()
    elif self.cached_block_lru:
        block_id, _ = self.cached_block_lru.popitem(last=False)  # 淘汰 front
        self.reset_hash(block_id)  # 清除 hash + hash_to_block_id
        self.blocks[block_id].ref_count = 1
        return block_id
    raise RuntimeError("No free blocks available")
```

### 6.2 孤儿 block（ADR-05）

prefix chain 中多个 block 有前缀依赖（A→B→C）。淘汰 A 后，B、C 仍在 LRU 中但永远不会被 chain hash 命中（因为 chain hash 需要从 block 0 开始匹配）。

M5 不处理孤儿 block：等待 LRU 自然淘汰。vLLM V1 用树结构实现级联淘汰，是更优方案。

---

## 7. Prefix Cache 命中流程

完整的新请求分配流程：

```text
请求 prompt = [t0, t1, ..., t11], block_size = 4

Step 1: can_admit(prompt_len=12) → ceil(12/4) = 3 blocks needed → 容量检查

Step 2: can_admit_with_cache(token_ids):
  block 0: tokens [t0..t3] → h₀ = hash([t0..t3], -1)
    → hash_to_block_id.get(h₀) = block_5 → 命中！
  block 1: tokens [t4..t7] → h₁ = hash([t4..t7], h₀)
    → hash_to_block_id.get(h₁) = block_8 → 命中！
  block 2: tokens [t8..t11] → h₂ = hash([t8..t11], h₁)
    → hash_to_block_id.get(h₂) = None → 未命中，停止
  → return 2（命中 2 个 block）

Step 3: allocate_with_cache(token_ids, num_cached=2):
  touch(block_5) → ref++ + LRU remove
  touch(block_8) → ref++ + LRU remove
  allocate() → block_12（新 block，从 free pool 或 LRU 淘汰）
  block_table = [5, 8, 12]
```

---

## 8. CoW（Copy-on-Write）

当多个请求共享同一个 block（ref_count > 1），某个请求需要写入时必须先拷贝独占副本。

```text
请求 A 和 B 共享 block_5（ref_count=2，hash=h₀）

cow_if_shared(A, block_idx=0):
  old_bid = 5, ref=2 → 需要 CoW
  new_bid = allocate() → block_15
  for layer in layers:
    layer.k[15] = layer.k[5].clone()    # 拷贝 K tensor
    layer.v[15] = layer.v[5].clone()    # 拷贝 V tensor
  迁移 hash: hash_to_block_id[h₀] = 15
  dec_ref(5) → ref=1（B 仍持有）
  A.block_table[0] = 15               # A 独占 block_15
```

CoW 只在未满的 block 上触发（满 block 只读不写）。M5 实现中 CoW 是同步的（写入前立即 clone），vLLM V1 用异步批量 CoW 优化性能。

---

## 9. hash_blocks 注册时机

`hash_blocks()` 在两个时刻被 loop.py 调用：

### 9.1 Prefill 后

```python
# loop.py prefill 循环
for i, req in enumerate(admitted):
    if hasattr(adapter, 'cache') and hasattr(adapter.cache, 'hash_blocks'):
        adapter.cache.hash_blocks(req.request_id, req.prompt_ids.squeeze(0).tolist())
    # ... 采样 ...
```

为 prompt 中填满的 block 注册 chain hash。

### 9.2 Decode 后

```python
# loop.py decode 循环
for req, tok in zip(running, sampled):
    req.seq_len += 1
    if hasattr(adapter, 'cache') and hasattr(adapter.cache, 'hash_blocks'):
        all_ids = req.prompt_ids.squeeze(0).tolist() + [t.item() for t in req.generated_tokens]
        adapter.cache.hash_blocks(req.request_id, all_ids)
```

每步 decode 后重新检查：如果之前的部分 block 因为新 token 而填满，注册它们的 hash。

**跳过已注册**：`hash_blocks()` 内部检查 `block.hash != -1`，已注册的 block 直接用已有 hash 继续链式计算，不重复注册。

---

## 10. loop.py 的 cache-aware 改动

M5 对 loop.py 的 admit 流程做了三处改动：

```python
# 原始 M4 流程
if not adapter.can_admit(prompt_len):
    break
adapter.allocate(req.request_id, prompt_len)

# M5 cache-aware 流程
if not adapter.can_admit(prompt_len):       # ① 容量检查（M2/M3/M4 通用）
    break
prompt_ids = req.prompt_ids.squeeze(0).tolist()
num_cached = adapter.can_admit_with_cache(prompt_ids)  # ② prefix cache 查询
if num_cached == -1:
    break  # 容量不够
# ... admit ...
if num_cached > 0:
    adapter.allocate_with_cache(req.request_id, prompt_ids, num_cached)  # ③ cache-aware 分配
else:
    adapter.allocate(req.request_id, prompt_len)  # M4 路径
```

M2/M3 adapter 的 `can_admit_with_cache()` 返回 0（no-op），确保不影响已有路径。

---

## 11. 最终架构

```text
cache/（~1,467L，5 文件）
├── block_pool.py (308L)     ← M5 升级：chain hash + LRU + touch + CoW 支持
├── paged_kv_cache.py (572L) ← M5 新增：hash_blocks + cow_if_shared + allocate_request_with_cache
├── adapter.py (338L)        ← M5 新增：can_admit_with_cache + allocate_with_cache
├── kv_cache.py              M2 不变
└── batched_kv_cache.py      M3 不变

engine/（~249L loop.py）
└── loop.py (249L)           ← M5 改动：cache-aware admit + hash_blocks 注册
```

---

## 12. 修复的 bug

| bug | 根因 | 修复 |
|---|---|---|
| `can_allocate` 重复定义 | M5 新方法和 M4 旧方法同名 | 合并为单方法 + `isinstance` 分发 |
| 参数名不匹配 | 测试用 `num_cached`，实现用 `cached` | 统一为 `num_cached` |
| `num_free_blocks` 不含 LRU | 只计 `free_block_ids` | 加上 `cached_block_lru` |
| `blocks[h]` 越界写入 | hash 值当 block_id 用 | 改为 `hash_to_block_id[h]` |
| `all_token_ids` 未定义 | decode hash_blocks 缺少变量 | 构造 `prompt + generated` 完整列表 |
| M2/M3 容量检查缺失 | loop.py 直接调 `can_admit_with_cache` | 先调 `can_admit` 再调 `can_admit_with_cache` |
| 循环导入 | `adapter.py → engine → adapter.py` | 避免直接 import，用 `hasattr` 检查 |

---

## 13. 测试覆盖

314 tests 全绿，M5 新增 44 tests：

| 测试文件 | 数量 | 验证内容 |
|---|---:|---|
| `test_block_pool_m5.py` | 23 | chain hash、LRU、touch、can_allocate、allocate_with_cache、hash_blocks |
| `test_prefix_cache_m5.py` | 8 | allocate_request_with_cache、can_allocate 容量、skip prefill 正确性 |
| `test_cow_hash_m5.py` | 7 | hash 注册、CoW clone、hash 迁移、共享 block 不污染 |
| `test_prefix_cache_e2e_m5.py` | 6 | full hit、partial hit、serial 等价、M4 回归、多轮复用、spy 机制验证 |

### E2E 容量控制技巧

通过 `num_blocks` 限制并发，迫使请求串行执行：

```text
prompt_len=12, block_size=4 → ceil(12/4) = 3 blocks/req
num_blocks=5 → req-0 用 3 blocks (2 free)，req-1 需要 3 → 阻塞
req-0 完成 → 3 blocks 进 LRU (带 hash) + 2 free = 5 → req-1 命中 prefix cache
```

---

## 14. 未完成项

| 项目 | 状态 | 说明 |
|---|---|---|
| skip prefill 优化 | defer | hash 注册已完成，但命中后仍执行 full prefill（正确但非最优） |
| 孤儿 block 处理 | defer | ADR-05：不处理，LRU 自然淘汰 |
| benchmark 数据 | 可选 | TTFT 对比、hit rate 统计 |

---

## 15. 与后续里程碑关系

| 里程碑 | M5 提供什么 | M5 不含什么 |
|---|---|---|
| M6 MoE 教学版 | prefix cache 基础设施 | MoE 路由 |
| M9 Triton kernel | hash + CoW 接口不变 | Triton/CUDA kernel |
| M11 Chunked Prefill | block 级 hash 注册 | token budget 调度 |
| M14 VLM image hash | chain hash 机制可复用 | 多模态 prefix |
