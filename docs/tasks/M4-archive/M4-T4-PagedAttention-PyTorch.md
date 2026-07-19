# M4-T4 — PagedAttention PyTorch 伪版

> **状态**：⬜ pending
> **里程碑**：M4 PagedAttention
> **目标**：让 `GQAAttention.forward` 支持 `PagedKVCache`，通过 PyTorch gather 实现教学版 PagedAttention。

## 背景

M4 不写 Triton kernel。attention 层检测到 `PagedLayerKVCache` 后，先根据 block table gather 临时连续 KV，再复用现有 attention 计算。

## 产出

- `attention.py` 增加 PagedKVCache 分派。
- per-row mask 支持 paged gather 后的不同 seq_len。
- fixed-slot 路径保持不变。

## 算法核心

1. 当前 token K/V 写入对应 physical block offset。
2. 按 request 的 block table gather full K/V。
3. batch 内 padding 到 `max_seq_len_in_batch`。
4. 用 per-row mask 避免读无效 padding。

## 测试

- B=1 paged logits 与 fixed-slot logits 对齐。
- B>1 变长请求与 fixed-slot 对齐。
- 跨 block 边界 decode 对齐。

## DoD

- [ ] 不破坏 M2/M3 attention 测试。
- [ ] Paged path 通过 logits 等价测试。
- [ ] 文档说明 PyTorch 伪版性能预期更慢。
