# M5-T3 — Partial Hit CoW + hash 注册（对齐 vLLM V1 cache_partial_block + move_block_hashes）

> 处理 prefix cache 命中在 block 中间的情况：CoW 拷贝后独占写入，以及 prefill/decode 后注册满 block hash。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M5-T3 |
| 状态 | 🟡 in_progress |
| 前置 | M5-T2 ✅ |
| 后续 | M5-T4 E2E |
| 估时 | 2～3h |
| 核心文件 | `cache/block_pool.py`、`cache/paged_kv_cache.py` |
| vLLM V1 对齐 | `cache_partial_block()` + `move_block_hashes()` + `cache_full_blocks()` |

## 背景

T1 的 chain hash 只对满 block 计算，`can_allocate` 也只匹配满 block。但 vLLM V1 支持 partial block 的 hash 注册（`cache_partial_block`）：当一个 block 从部分填充变成满时，需要更新 hash。

另外，CoW（Copy-on-Write）场景：当 prefix cache 命中了一个 shared block，请求需要写入该 block 时，必须先拷贝一份独占副本。

## 范围

### 1. hash_blocks 注册时机

prefill/decode 完成后，检查是否有 block 从部分变成满，注册 hash：

```python
def hash_blocks(self, request_id: str, token_ids: list[int]) -> None:
    """prefill/decode 后注册填满的 block 的 chain hash。"""
    table = self.block_tables[request_id]
    num_full_blocks = len(token_ids) // self.block_size
    h = -1
    for i in range(num_full_blocks):
        block_id = table.block_ids[i]
        block = self.block_pool.blocks[block_id]
        if block.hash != -1:
            # 已注册过，取已有 hash 继续链
            h = block.hash
            continue
        # 未注册：计算 hash 并注册
        start = i * self.block_size
        end = start + self.block_size
        block_tokens = token_ids[start:end]
        h = self.block_pool.compute_hash(block_tokens, h)
        block.hash = h
        block.token_ids = block_tokens
        self.block_pool.hash_to_block_id[h] = block_id
```

### 2. Partial Hit CoW（对齐 vLLM V1 move_block_hashes）

当请求 decode 追加 token 到一个 shared block 时（ref_count > 1），需要 CoW：

```python
def cow_if_shared(self, request_id: str, block_idx: int) -> int:
    """如果 block 是 shared（ref_count > 1），拷贝独占副本。
    返回（可能是新的）block_id。
    """
    table = self.block_tables[request_id]
    old_block_id = table.block_ids[block_idx]
    block = self.block_pool.blocks[old_block_id]

    if block.ref_count <= 1:
        return old_block_id  # 独占，不需要 CoW

    # 分配新 block
    new_block_id = self.block_pool.allocate()

    # 拷贝 KV tensor
    for layer in self.layers:
        layer.k[new_block_id] = layer.k[old_block_id].clone()
        layer.v[new_block_id] = layer.v[old_block_id].clone()

    # 迁移 hash（对齐 vLLM V1 move_block_hashes）
    if block.hash != -1:
        new_block = self.block_pool.blocks[new_block_id]
        new_block.hash = block.hash
        new_block.token_ids = block.token_ids
        # 更新 hash_to_block_id 指向新 block
        self.block_pool.hash_to_block_id[block.hash] = new_block_id

    # 释放旧 block 引用
    self.block_pool.dec_ref(old_block_id)

    # 替换 block_table
    table.block_ids[block_idx] = new_block_id
    return new_block_id
```

### 3. decode 时调用 CoW

在 `prepare_decode()` 或 `append_token()` 时，如果当前最后一个 block 是 shared，先 CoW 再写入。

## vLLM V1 对应关系

| vLLM V1 | inferlite M5-T3 |
|---|---|
| `cache_full_blocks()` | `hash_blocks()` |
| `cache_partial_block()` | 简化为 hash_blocks 内处理（block 满时注册） |
| `move_block_hashes(src, dst)` | `cow_if_shared()` 内 hash 迁移 |
| `_apply_cow()` | `cow_if_shared()` |
| `_pending_cow_copies` | 不需要（同步执行） |

## 测试

- hash_blocks 正确注册满 block 的 chain hash
- 连续两次 hash_blocks 不重复注册
- CoW 后新 block 独占（ref_count=1）
- CoW 后 KV 数据与原 block 一致（`torch.equal`）
- CoW 后 hash_to_block_id 正确更新
- CoW 后旧 block ref_count 正确递减
- M4 + T1 + T2 回归

## DoD

- [ ] hash_blocks 正确注册 chain hash
- [ ] CoW 正确拷贝 + hash 迁移
- [ ] shared block 写入不污染其他请求
- [ ] 所有前序测试全绿
