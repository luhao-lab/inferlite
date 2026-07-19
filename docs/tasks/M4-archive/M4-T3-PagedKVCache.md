# M4-T3 — PagedKVCache

> **状态**：⬜ pending
> **里程碑**：M4 PagedAttention
> **目标**：实现多层分页 KV Cache 容器，支持按 block table 写入和 gather 读取。

## 背景

M3 `BatchedKVCache` 的单层 shape 是 `[S, H_kv, L, D]`。M4 改为 `[num_blocks, block_size, H_kv, D]`。

## 产出

- `PagedLayerKVCache`
  - `k: torch.Tensor`
  - `v: torch.Tensor`
- `PagedKVCache`
  - `layers`
  - `block_manager`
  - `request_tables`
  - `allocate_request(request_id, prompt_len)`
  - `free_request(request_id)`
  - `ensure_capacity_for_append(request_id)`
  - `write(...)`
  - `gather(...)`

## 算法核心

- prefill 前按 `ceil(prompt_len / block_size)` 分配 block。
- decode 追加时如果 `seq_len % block_size == 0`，先分配新 block。
- gather 通过 block table 把非连续 block 拼成临时连续 KV。

## 测试

- 单请求 prefill 写入跨多个 block。
- decode 追加跨 block 边界。
- gather 与连续 KV 对齐。
- free_request 释放所有 block。

## DoD

- [ ] PagedKVCache 单测全过。
- [ ] 不修改 M3 `BatchedKVCache`。
- [ ] 支持 CPU/MPS。
