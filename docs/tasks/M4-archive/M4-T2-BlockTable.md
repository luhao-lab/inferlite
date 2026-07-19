# M4-T2 — BlockTable

> **状态**：⬜ pending
> **里程碑**：M4 PagedAttention
> **目标**：实现 request 逻辑 token 位置到物理 block 的映射。

## 背景

PagedAttention 的核心是 block table：请求看到连续逻辑地址，底层可映射到非连续物理 block。

## 产出

- `BlockTable`
  - `request_id: str`
  - `block_ids: list[int]`
  - `seq_len: int`
  - `logical_block(pos) -> int`
  - `block_offset(pos) -> int`
  - `physical_location(pos) -> tuple[int, int]`
  - `append_block(block_id)`

## 算法核心

```text
logical_block = pos // block_size
block_offset  = pos % block_size
physical_block = block_ids[logical_block]
```

## 测试

- pos=0/15/16/17 边界映射。
- block table 扩展。
- 多 request block table 互不影响。
- seq_len 更新正确。

## DoD

- [ ] `BlockTable` 单测全过。
- [ ] 支持跨 block 边界。
- [ ] 命名区分 logical/physical。
