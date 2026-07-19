# M4-T5 — BatchEngine Integration

> **状态**：⬜ pending
> **里程碑**：M4 PagedAttention
> **目标**：让 `batch_generate` 可选择 fixed-slot 或 paged KV Cache 路径。

## 背景

M3 `batch_generate` 绑定 `BatchedKVCache`。M4 需要在不破坏 M3 的前提下支持 `PagedKVCache`。

## 产出

- `batch_generate(..., cache_type="fixed" | "paged", block_size=16, num_blocks=None)` 或新增 `batch_generate_paged()`。
- request admit 时分配 block table。
- finished 时释放 request 的 blocks。

## 关键决策

优先考虑新增 `batch_generate_paged()`，减少对 M3 稳定路径的影响；最终是否合并入口在实现时判断。

## 测试

- 同一 prompts 下 fixed-slot 与 paged 输出 token 级一致。
- waiting > capacity 时全部完成。
- finished 请求释放 blocks 后 waiting 请求进入。

## DoD

- [ ] M3 fixed path 测试不变。
- [ ] paged path E2E 通过。
- [ ] 支持 metrics 基本字段。
