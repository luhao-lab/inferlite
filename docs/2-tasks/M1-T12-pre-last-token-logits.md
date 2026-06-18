# M1-T12-pre Last-token logits optimization

## 元信息
- **任务 ID**: T12-pre
- **里程碑**: M1
- **状态**: 🟡 active
- **前置**: T10 `EngineCore.step` 三段式
- **估时**: 1h

## 背景
T10 的 `EngineCore.step` 采用教学版实现：先通过模型拿到全序列 logits，再取最后位置：

```python
logits = self.model(input_ids)      # [B, T, V]
next_token_logits = logits[:, -1, :]
```

对于 Qwen3-0.6B，vocab_size=151936。如果序列很长，`[B, T, V]` 会浪费大量内存，且 lm_head 的运算开销也很大。

T12（真实模型 smoke test）前的优化目标：`logits_to_keep=1`，即模型只返回最后一个位置的 logits，省去前 T-1 个位置的 lm_head。

## 目标
实现 `Qwen3ForCausalLM.forward` 支持 `logits_to_keep` 参数，`LLMModel` 协议和 `EngineCore.step` 利用该参数。

## 产出文件
- `inferlite/engine/protocol.py`（LLMModel 协议扩展）
- `inferlite/model/qwen3.py`（Qwen3ForCausalLM.forward 增加 logits_to_keep）
- `inferlite/engine/core.py`（step 使用 logits_to_keep）
- `tests/unit/test_qwen3_causal_lm_logits_to_keep.py`

## 具体改动
**inferlite/engine/protocol.py**：
- 扩展 `LLMModel`，改为 `def __call__(self, input_ids, *, logits_to_keep=None) -> Tensor: ...`

**inferlite/model/qwen3.py**：
- `Qwen3ForCausalLM.forward` 支持 `logits_to_keep`，当不为 None 时只保留最后 `logits_to_keep` 个位置的 hidden states。

**inferlite/engine/core.py**：
- `EngineCore.step` 调用 `self.model(input_ids, logits_to_keep=1)`。

**单测**：
- 确保 `logits_to_keep` 不影响原有行为。

## 非目标
- 不做 `past_key_values` 或 KV-cache 优化。

## DoD
- [ ] Qwen3ForCausalLM.forward 支持 logits_to_keep
- [ ] Protocol 和 EngineCore 使用新参数
- [ ] tests/unit 新增 logits_to_keep=1 验证
- [ ] T0-T12pre 回归全绿
- [ ] ruff / ruff format 无异常
- [ ] 提交中包含 plan/M1 和 任务索引更新
