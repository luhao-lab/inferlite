# M5-T2 — prefix cache allocate + 跳过 prefill（对齐 vLLM V1 find_longest_cache_hit）

> PagedCacheAdapter 支持 cache-aware allocate；引擎 loop 感知 `num_cached_tokens`，跳过已缓存部分的 prefill。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M5-T2 |
| 状态 | ⬜ pending |
| 前置 | M5-T1 ✅ |
| 后续 | M5-T3 partial hit CoW |
| 估时 | 3～4h |
| 核心文件 | `cache/adapter.py`、`cache/paged_kv_cache.py`、`engine/loop.py` |
| vLLM V1 对齐 | `SingleTypeKVCacheManager.find_longest_cache_hit()` + `num_computed_tokens` |

## 背景

T1 让 BlockPool 有了 hash 查找能力。T2 要把这个能力接入引擎：新请求到来时，先查 hash 命中多少 block，跳过已缓存部分，只 prefill 剩余 token。

vLLM V1 的做法：
1. `find_longest_cache_hit()` 逐 block 查 chain hash → 返回 `num_computed_tokens`
2. scheduler 用 `num_computed_tokens` 决定 prefill 从哪个 token 开始
3. model runner 的 `slot_mapping` 从 `num_cached_tokens` 开始生成

## 范围

### PagedCacheAdapter 改造

```python
# 新增方法
def can_admit_with_cache(self, prompt_ids: list[int]) -> int:
    """返回 prefix cache 命中的 block 数（0 = 无命中，-1 = 容量不够）。"""

def allocate_with_cache(self, request_id: str, prompt_ids: list[int],
                        num_cached_blocks: int) -> None:
    """分配 block table：touch cached block + allocate 新 block。"""
```

### PagedKVCache 改造

```python
def allocate_request_with_cache(self, request_id: str,
                                 prompt_ids: list[int],
                                 num_cached_blocks: int) -> None:
    """分配 block table，复用 cached block。"""
    # 1. touch cached blocks（ref++ + LRU remove）
    # 2. 分配新 block 给未命中部分
    # 3. 记录 request_id → block_table
```

### 引擎 loop.py 改造

```python
# 当前 M4:
admit → allocate(prompt_len) → prefill(全部 prompt)

# M5:
admit → num_cached = adapter.can_admit_with_cache(prompt_ids)
      → adapter.allocate_with_cache(request_id, prompt_ids, num_cached)
      → if num_cached > 0:
          # 跳过已缓存部分，只 prefill [num_cached*block_size:] 的 token
          skip_tokens = num_cached * block_size
          partial_prompt = prompt_ids[skip_tokens:]
          prefill(partial_prompt)
          # 注意：metadata 的 seq_lens 要包含 cached tokens
        else:
          prefill(全部 prompt)
```

### 关键约束

1. **跳过 prefill 不等于跳过 KV 写入**：cached block 的 KV 已经在物理 tensor 中，不需要重写。新 token 的 KV 需要 scatter 到新分配的 block。
2. **AttentionMetadata.seq_lens 必须包含 cached tokens**：attention 计算时需要看到完整序列长度（cached + new），causal mask 才能正确。
3. **block_table 必须包含 cached block IDs**：gather 读取时需要所有 block（cached + new）。

## vLLM V1 对应关系

| vLLM V1 | inferlite M5-T2 |
|---|---|
| `find_longest_cache_hit()` | `can_admit_with_cache()` |
| `num_computed_tokens` | `num_cached_blocks * block_size` |
| scheduler 决定 prefill 范围 | loop.py 决定 prefill 范围 |
| `slot_mapping` 从 cached offset 开始 | scatter 只写新 token 的 slot |

## 测试

- 相同 prompt 第二个请求命中 prefix cache（num_cached > 0）
- 跳过 prefill 后输出与不跳过完全相同（`torch.equal`）
- 不同 prompt 不命中（num_cached = 0）
- block_table 包含 cached + new block
- seq_lens 包含 cached tokens
- M4 回归（无 cached block 时行为不变）

## DoD

- [ ] PagedCacheAdapter 支持 cache-aware allocate
- [ ] loop.py 能跳过 cached tokens 的 prefill
- [ ] 相同 prompt 的第二个请求正确命中并跳过 prefill
- [ ] 输出与不跳过时 `torch.equal` 等价
- [ ] M4 现有测试全绿
