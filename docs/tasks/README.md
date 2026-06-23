# 任务卡索引

> spec-kit 风格：每张任务卡一个独立文件，便于 AI/人精准定位与 review。
> 进度状态汇总见 `docs/plan/M1.md` §4 任务卡总表；本目录是任务卡详细内容。
> 新建任务卡：`cp _TEMPLATE.md M1-TX-name.md` 或运行 `/work TX` 自动建。

## M1 Phase 1（数值对齐）

| ID | 文件 | 状态 |
| --- | --- | --- |
| T0 | （历史，已并入 M0） | ✅ |
| T0 | [M1-T0-ModelConfig.md](./M1-T0-ModelConfig.md) | ✅ |
| T1 | [M1-T1-RMSNorm.md](./M1-T1-RMSNorm.md) | ✅ |
| T2 | [M1-T2-SwiGLU.md](./M1-T2-SwiGLU.md) | ✅ |
| T3 | [M1-T3-RoPE.md](./M1-T3-RoPE.md) | ✅ |
| T4 | [M1-T4-GQA-Attention.md](./M1-T4-GQA-Attention.md) | ✅ |
| T5 | [M1-T5-DecoderLayer.md](./M1-T5-DecoderLayer.md) | ✅ |
| T6 | [M1-T6-Qwen3Model.md](./M1-T6-Qwen3Model.md) | ✅ |
| T7 | [M1-T7-WeightMap-load-from-hf.md](./M1-T7-WeightMap-load-from-hf.md) | ✅ |
| T8 | [M1-T8-L1-logits-alignment.md](./M1-T8-L1-logits-alignment.md) | ✅ |

## M1 Phase 2（出字）闭环

| ID | 文件 | 状态 |
| --- | --- | --- |
| T9 | [M1-T9-LLMModel-GreedySampler.md](./M1-T9-LLMModel-GreedySampler.md) | ✅ |
| T10 | [M1-T10-EngineCore-step.md](./M1-T10-EngineCore-step.md) | ✅ |
| T11 | [M1-T11-CLI-L2-e2e.md](./M1-T11-CLI-L2-e2e.md) | ✅ |
| T12-pre | [M1-T12-pre-last-token-logits.md](./M1-T12-pre-last-token-logits.md) | ✅ |
| T12 | [M1-T12-smoke-test.md](./M1-T12-smoke-test.md) | ✅ |

## M2（KV Cache + 设备支持）

| ID | 文件 | 状态 |
| --- | --- | --- |
| M2-T1 | [M2-T1-KVCache-DataStructure.md](./M2-T1-KVCache-DataStructure.md) | ⬜ |
| M2-T2 | [M2-T2-Attention-KVCache.md](./M2-T2-Attention-KVCache.md) | ⬜ |
| M2-T3 | [M2-T3-Model-Passthrough.md](./M2-T3-Model-Passthrough.md) | ⬜ |
| M2-T4 | [M2-T4-Generate-Loop.md](./M2-T4-Generate-Loop.md) | ⬜ |
| M2-T5 | [M2-T5-CLI-Device-Dtype.md](./M2-T5-CLI-Device-Dtype.md) | ⬜ |

## 任务卡模板（7 字段）

新建任务卡时复制 `_TEMPLATE.md`：
- 前置（依赖哪些 T）
- 产出文件
- 算法核心（公式/代码骨架）
- L0 测试清单
- DoD（验收标准）
- 坑（按概率排序）
- 估时
