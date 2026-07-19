# M4-T6 — E2E Correctness & Benchmark

> **状态**：⬜ pending
> **里程碑**：M4 PagedAttention
> **目标**：验证 PagedKVCache 正确性，并量化内存分配/碎片情况。

## 产出

- E2E 测试：fixed-slot vs paged token 级等价。
- benchmark 脚本：`scripts/bench_paged_attention.py`。
- 结果归档：`bench/results/<date>-m4-paged-attention-*.md`。

## 指标

| 指标 | 说明 |
|---|---|
| allocated_blocks | 实际分配 block 数 |
| used_tokens | 实际有效 token 数 |
| capacity_tokens | allocated_blocks × block_size |
| internal_fragmentation | capacity_tokens - used_tokens |
| throughput | tok/s，仅参考 |

## 测试场景

- 短请求多并发。
- 长短混合请求。
- prompt/output 跨 block 边界。
- block_size 扫描：8/16/32。

## DoD

- [ ] E2E token 级等价。
- [ ] benchmark 结果归档。
- [ ] 说明 M4 性能是否慢于 M3，以及原因。
