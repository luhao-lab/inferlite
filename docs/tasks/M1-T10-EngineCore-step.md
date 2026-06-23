# M1-T10 EngineCore.step 三段式

## 元信息
- **任务 ID**: T10
- **里程碑**: M1·P2（出字闭环）
- **状态**: 🟡 in-progress
- **前置**: T9 `LLMModel Protocol + GreedySampler`
- **估时**: 3h

## 背景
T9 已经完成两个基础组件：

```text
LLMModel Protocol: model(input_ids) -> logits [B, T, V]
GreedySampler: logits [B, V] -> next_token [B, 1]
```

T10 要把它们串成最小推理 step：

```text
input_ids
  -> model(input_ids)
  -> logits [B, T, V]
  -> logits[:, -1, :]
  -> sampler
  -> next_token [B, 1]
```

这个 `step` 是后续 generate loop 的核心原子操作。

## 目标
实现 `EngineCore.step()`：

```python
next_token = engine.step(input_ids)
```

要求：

```text
input_ids [B, T]
  -> next_token [B, 1]
```

并保证只使用最后一个 token 位置的 logits。

## 非目标
- 不实现完整 generate loop。
- 不接 tokenizer。
- 不处理 EOS。
- 不处理 max_new_tokens。
- 不处理 KV cache。
- 不处理 batch request 调度。
- 不处理 streaming。

这些留到 T11+。

## 产出文件
- `inferlite/engine/core.py`
  - `EngineCore`
- `inferlite/engine/__init__.py`
  - 导出 `EngineCore`
- `tests/unit/test_engine_core.py`

## API 草案

```python
class EngineCore:
    """最小单步推理引擎。"""

    def __init__(self, model: LLMModel, sampler: GreedySampler) -> None:
        self.model = model
        self.sampler = sampler

    def step(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token = self.sampler(next_token_logits)
        return next_token
```

## shape 约定

```text
input_ids:          [B, T]
model logits:       [B, T, V]
next_token_logits:  [B, V]
next_token:         [B, 1]
```

`next_token` 保持 `[B, 1]`，是为了后续 generate loop 可以直接：

```python
input_ids = torch.cat([input_ids, next_token], dim=1)
```

## 为什么 step 放 engine，不放 sampler

Sampler 只负责：

```text
[B, V] -> [B, 1]
```

它不应该知道：

```text
logits 是 [B, T, V]
应该取最后一个位置 logits[:, -1, :]
```

“取最后位置 + 调 model + 调 sampler”是推理流程调度逻辑，属于 EngineCore。

## T12 前置优化：只计算最后 token 的 lm_head

T10 当前先使用教学版最小实现：

```python
logits = self.model(input_ids)      # [B, T, V]
next_token_logits = logits[:, -1, :]
```

这会让 `Qwen3ForCausalLM` 对整个序列的所有位置都计算 `lm_head`。但单步生成实际只需要最后一个位置：

```text
hidden_states[:, -1, :] -> lm_head -> next_token_logits [B, V]
```

因此将以下优化规划到 **T12 前置优化**，在真实 Qwen3-0.6B smoke test 前完成：

```text
Qwen3ForCausalLM.forward 支持 logits_to_keep=1
LLMModel Protocol 视需要扩展 logits_to_keep
EngineCore.step 使用 model(input_ids, logits_to_keep=1)
验证 full_logits[:, -1:, :] == optimized_logits
```

T10 不做该优化，避免把“step 语义打通”和“lm_head 性能优化”混在一张卡里。

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | `EngineCore.step` shape | `[B, T] -> [B, 1]` | exact |
| 2 | 只使用最后一个位置 logits | 手工 logits | exact |
| 3 | batch 场景逐行 next token | 手工 logits | exact |
| 4 | EngineCore 只依赖 LLMModel Protocol | FakeModel | runtime/type |
| 5 | 与 GreedySampler 组合正确 | T9 sampler | exact |

## 建议单测

### 1. 使用最后位置

```python
logits = torch.tensor([
    [
        [100.0, 0.0, 0.0],
        [0.0, 0.0, 100.0],
    ]
])
```

如果错误地取第 0 个位置，会选 token 0；正确取最后位置应选 token 2。

### 2. batch 场景

```text
sample 0 最后位置选 2
sample 1 最后位置选 0
```

验证每行独立处理。

## DoD
- [ ] `EngineCore` 实现
- [ ] `EngineCore.step(input_ids)` 返回 `[B, 1]`
- [ ] 单测覆盖只取最后一个位置 logits
- [ ] 单测覆盖 batch 场景
- [ ] `tests/unit/test_engine_core.py` 全绿
- [ ] T0-T10 回归全绿
- [ ] `docs/kb/knowledge.md` 补 engine/step 概念（如缺）
- [ ] `docs/tasks/README.md` 与 `docs/plan/M1.md` 状态更新
- [ ] `README.md` 当前进度同步（任务完成时）
- [ ] commit `feat(engine): add single-step engine core`

## 坑
1. **不要在 sampler 内取最后位置**：这是 engine 的职责。
2. **next_token shape 必须是 `[B, 1]`**：否则后续 `torch.cat(..., dim=1)` 不方便。
3. **不要提前写 generate loop**：T10 只做一个 step。
4. **不要接 tokenizer**：文本 encode/decode 放到后续任务。
5. **不要把 EngineCore 绑定到 Qwen3ForCausalLM**：类型应依赖 `LLMModel` Protocol。
