# M5-T1 — BlockPool hash + LRU（对齐 vLLM V1 BlockPool）

> 在 M4 BlockPool 基础上增加 hash-based prefix caching 和 LRU 淘汰机制，对齐 vLLM V1 `BlockPool` + `FreeKVCacheBlockQueue` 的核心设计。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M5-T1 |
| 状态 | ⬜ pending |
| 前置 | M4 ✅ |
| 后续 | M5-T2 prefix cache allocate |
| 估时 | 3～4h |
| 核心文件 | `inferlite/cache/block_pool.py` |
| vLLM V1 对齐 | `vllm/v1/core/block_pool.py` BlockPool + FreeKVCacheBlockQueue |

## 背景

M4 的 `BlockPool`（85L）只维护 ref_count + free_block_ids。每个请求独占 block，即使前缀相同也各自分配。vLLM V1 在此基础上增加了：
- chain hash：对满 block 的 token 计算链式 hash
- cached_block_hash_to_block：hash → block 查找
- FreeKVCacheBlockQueue：free 时按 eviction 优先级排列（无 hash 放队首优先淘汰，有 hash 放队尾 LRU）
- touch()：cache 命中时 ref_cnt++ + 从 free queue 移除

## 范围

### 明确做

1. **Block 扩展**：加 `hash: int`、`token_ids: list[int]` 字段
2. **compute_hash(token_ids, prefix_hash)**：链式 xxhash64，保证不同位置相同 token 产生不同 hash
3. **hash_to_block_id: dict[int, int]**：hash → block_id 1:1 映射（简化 vLLM V1 的 BlockHashToBlockMap）
4. **cached_block_lru: OrderedDict[int, None]**：ref=0 且有 hash 的 block，front=最久未用，back=最近使用
5. **touch(block_id)**：cache 命中时 ref++ + 从 cached_block_lru 移除
6. **改 allocate()**：free pool 不够时从 cached_block_lru 淘汰（popitem(last=False)），淘汰时清除旧 hash
7. **改 free()/dec_ref()**：ref=0 且有 hash → 进 cached_block_lru（不归还 free pool）；ref=0 且无 hash → 归还 free pool
8. **can_allocate(token_ids) → int**：查 chain hash 返回命中 block 数（-1 表示空闲不够）
9. **allocate_with_cache(token_ids, num_cached) → list[int]**：touch cached block + 分配新 block
10. **hash_blocks(block_ids, token_ids, num_full) → None**：prefill/decode 后注册满 block hash
11. **reset_hash(block_id)**：淘汰时清除 block 的 hash + hash_to_block_id 映射

### 明确不做

- `BlockHashToBlockMap`（一个 hash → 多个 block）：简化为 1:1 dict
- kv_cache_events / metrics_collector：生产监控，教学不需要
- null_block：占位 block
- partial block hash：只对满 block 计算 hash（partial hit CoW 在 T3）
- kv_cache_group_id：单 attention 类型

## vLLM V1 对应关系

| vLLM V1 | inferlite M5-T1 |
|---|---|
| `BlockPool.__init__` | `BlockPool.__init__` + cached_block_lru |
| `FreeKVCacheBlockQueue` | `free_block_ids` (deque) + `cached_block_lru` (OrderedDict) |
| `BlockPool.touch(blocks)` | `touch(block_id)` |
| `BlockPool.get_new_blocks()` | `allocate()` + LRU 淘汰 |
| `BlockPool.free_blocks()` | `dec_ref()` / `free()` 改造 |
| `BlockPool.cache_full_blocks()` | `hash_blocks()` |
| `BlockPool._maybe_evict_cached_block()` | `_allocate_block()` 内 LRU 淘汰 |
| `BlockPool.cached_block_hash_to_block` | `hash_to_block_id` dict |
| `get_block_hash()` / chain hash | `compute_hash()` |

## LRU 淘汰策略（对齐 vLLM V1 free_blocks）

```
free(block_id):
  ref_count -= 1
  if ref_count == 0:
    if block.hash != -1:
      cached_block_lru[block_id] = None   # 有 hash → LRU 队尾（保留等复用）
    else:
      free_block_ids.append(block_id)      # 无 hash → 直接归还

allocate() 需要新 block:
  if free_block_ids:
    return free_block_ids.popleft()        # 优先用真正空闲的
  elif cached_block_lru:
    block_id, _ = cached_block_lru.popitem(last=False)  # 淘汰最久未用
    reset_hash(block_id)                   # 清除旧 hash
    return block_id
  else:
    raise RuntimeError
```

## 测试

- compute_hash 确定性（相同 token → 相同 hash）
- 链式 hash（不同位置相同 token → 不同 hash）
- can_allocate 缓存命中（相同 prompt 返回 num_cached > 0）
- allocate_with_cache 正确 touch + 分配新 block
- LRU 淘汰顺序（最久未用的先淘汰）
- touch 更新 LRU 顺序
- 淘汰后 hash_to_block_id 正确清理
- M4 回归（无 hash 时行为不变）

## DoD

- [ ] BlockPool 支持 hash chain + LRU 淘汰
- [ ] touch() / allocate_with_cache() / hash_blocks() 实现并测试
- [ ] M4 现有测试全绿
- [ ] 单测覆盖上述所有点
