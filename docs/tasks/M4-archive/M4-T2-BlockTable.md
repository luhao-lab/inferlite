# M4-T2 — BlockTable

> **状态**：⬜ pending
> **里程碑**：M4 PagedAttention
> **目标**：实现 request 逻辑 token 位置到物理 block 的映射。

## 背景

PagedAttention 的核心是 block table：请求看到连续逻辑地址，底层可映射到非连续物理 block。

## 产出

文件：`inferlite/cache/paged_kv_cache.py`（与 T3 PagedKVCache 同文件）

```python
@dataclass
class BlockTable:
    """单个请求的逻辑 block 到物理 block 的映射。"""
    request_id: str
    block_ids: list[int]       # logical_block_idx -> physical_block_id
    seq_len: int = 0           # 当前有效 token 数
    block_size: int = 16       # 每个 block 的 token 容量

    @property
    def num_blocks(self) -> int: ...
    @property
    def num_full_blocks(self) -> int: ...
    @property
    def last_block_offset(self) -> int: ...
    def needs_new_block(self) -> bool: ...
    def position_to_block(self, pos: int) -> tuple[int, int]: ...
```

## 接口语义

### `position_to_block(pos) -> tuple[int, int]`

- 逻辑位置 → `(physical_block_id, block_offset)`。
- `logical_block = pos // block_size`
- `block_offset = pos % block_size`
- 返回 `(block_ids[logical_block], block_offset)`

### `needs_new_block() -> bool`

- decode 追加一个 token 时是否需要分配新 block。
- 条件：`seq_len % block_size == 0 and seq_len > 0`

### `num_full_blocks -> int`

- 已满的 block 数：`seq_len // block_size`

### `last_block_offset -> int`

- 最后一个 block 中已用的 token 数：`seq_len % block_size`

## 算法核心

```text
logical_block = pos // block_size
block_offset  = pos % block_size
physical_block = block_ids[logical_block]
physical_index = (physical_block, block_offset)
```

## 测试

建议新建 `tests/unit/test_block_table.py`。

- pos=0/15/16/17 边界映射正确。
- `needs_new_block()` 在 block 边界返回 True。
- `num_full_blocks` 和 `last_block_offset` 计算正确。
- 多 request block table 互不影响。
- seq_len 更新后 position_to_block 正确。

## DoD

- [ ] `BlockTable` 单测全过。
- [ ] 支持跨 block 边界。
- [ ] 命名区分 logical/physical。
- [ ] 不依赖 attention/model/tensor。
