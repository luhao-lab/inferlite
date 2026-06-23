# M1-T9 LLMModel Protocol + GreedySampler

## 元信息
- **任务 ID**: T9
- **里程碑**: M1·P2（出字闭环）
- **状态**: 🟡 in-progress
- **前置**: T8 `Qwen3ForCausalLM` logits 对齐
- **估时**: 1h

## 背景
T8 已经打通：

```text
input_ids -> Qwen3ForCausalLM -> logits [B, T, vocab_size]
```

现在还不能“出字”，因为缺少两个东西：

1. 从 logits 选下一个 token 的 sampler。
2. 一个简单稳定的模型协议，让后续 EngineCore/CLI 不直接依赖具体模型类。

T9 的目标是把“单步 next-token 选择”封装出来，为 T10 的 `EngineCore.step()` 和 T11 的 CLI e2e 做准备。

## 目标
实现最小推理协议和贪心采样器：

```text
input_ids
  -> model(input_ids)
  -> logits [B, T, V]
  -> logits[:, -1, :]
  -> GreedySampler
  -> next_token_id [B, 1]
```

T9 完成后，应该能对一个 tiny/fake 模型执行单步 next-token 选择。

## 非目标
- 不接 tokenizer。
- 不写完整 generate loop。
- 不处理 KV cache。
- 不做 temperature / top-k / top-p。
- 不做 streaming。
- 不做真实 Qwen3-0.6B smoke test。

Tokenizer 建议放到 T11 CLI/e2e，因为 tokenizer 依赖真实模型目录和文本 I/O；T9 先只处理 tensor 级别的 next-token。

## 产出文件
- `inferlite/sampler/greedy.py` 或 `inferlite/sampler/__init__.py`
  - `GreedySampler`
- `inferlite/engine/protocol.py` 或等价位置
  - `LLMModel` Protocol
- `tests/unit/test_sampler.py`
- `tests/unit/test_protocol.py` 或合并到 sampler 测试

## API 草案

### 1. LLMModel Protocol

```python
from typing import Protocol

class LLMModel(Protocol):
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """返回 logits [B, T, V]。"""
```

用途：

```text
EngineCore / decode loop 只依赖“输入 input_ids，输出 logits”的协议，
不直接依赖 Qwen3ForCausalLM 的具体类名。
```

### 2. GreedySampler

```python
class GreedySampler:
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """从 logits [B, V] 选择 argmax token，返回 [B, 1]。"""
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token
```

### 3. 单步 helper（可选）

```python
def next_token(model: LLMModel, input_ids: torch.Tensor, sampler: GreedySampler) -> torch.Tensor:
    logits = model(input_ids)
    next_token_logits = logits[:, -1, :]
    return sampler(next_token_logits)
```

如果觉得 helper 边界更像 T10 的 EngineCore.step，可以 T9 只做 `GreedySampler` + Protocol。

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | GreedySampler shape | logits `[B, V]` -> token `[B, 1]` | exact |
| 2 | GreedySampler 选择最大 logit 下标 | 手工 logits | exact |
| 3 | batch 场景 | 每行独立 argmax | exact |
| 4 | LLMModel Protocol 可被 fake model 满足 | FakeModel | type/runtime |
| 5 | 单步 helper 只取最后一个位置 logits | 手工 `[B, T, V]` | exact |

## 建议单测

```python
def test_greedy_sampler_returns_argmax_keepdim():
    logits = torch.tensor([[0.1, 0.9, 0.2]])
    sampler = GreedySampler()
    assert torch.equal(sampler(logits), torch.tensor([[1]]))
```

```python
def test_next_token_uses_last_position_only():
    logits = torch.tensor([
        [[100.0, 0.0], [0.0, 100.0]],
    ])
    # 如果错误地用了第 0 个位置，会选 0；正确用最后位置应选 1。
```

## DoD
- [ ] `GreedySampler` 实现
- [ ] `LLMModel` Protocol 或等价轻量协议实现
- [ ] 单步 next-token helper 如有必要实现
- [ ] `tests/unit/test_sampler.py` 全绿
- [ ] T0-T9 回归全绿
- [ ] `docs/kb/knowledge.md` 补 greedy decoding / sampler 概念（如缺）
- [ ] `docs/tasks/README.md` 与 `docs/plan/M1.md` 状态更新
- [ ] `README.md` 当前进度同步（任务完成时）
- [ ] commit `feat(sampler): add greedy next-token sampler`

## 坑
1. **shape 约定**：sampler 输入建议是 `[B, V]`，输出 `[B, 1]`，方便后续 `torch.cat([input_ids, next_token], dim=1)`。
2. **不要在 sampler 里取 `logits[:, -1, :]`**：sampler 只负责 `[B, V] -> [B, 1]`，取最后位置是 Engine/helper 的职责。
3. **argmax 不等于采样**：GreedySampler 是确定性的，后续 temperature/top-p 再引入随机采样。
4. **dtype**：logits 是浮点，next_token_id 应是 `torch.long`。
5. **Protocol 不要过度设计**：T9 只需要 `input_ids -> logits`，不要提前塞 cache、attention_mask、streaming。
