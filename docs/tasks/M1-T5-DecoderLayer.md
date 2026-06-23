# M1-T5 DecoderLayer

## 元信息
- **任务 ID**: T5
- **里程碑**: M1·P1（数值对齐）
- **状态**: 🟡 in-progress
- **前置**: T1 `RMSNorm`，T2 `SwiGLUMLP`，T4 `GQAAttention`
- **估时**: 1h

## 目标
实现 Qwen3 decoder block 的最小数值对齐版：把 `RMSNorm`、`GQAAttention`、`SwiGLUMLP` 组合成一个标准 pre-norm decoder layer，为 T6 `Qwen3Model` 堆叠多层做准备。

## 产出文件
- `inferlite/model/qwen3.py::DecoderLayer`（推荐新文件，T6 会继续放 `Qwen3Model`）
- `tests/unit/test_decoder_layer.py`
- 如有必要：`inferlite/model/__init__.py` 导出

## 参考代码

### 主参考（数值真值）
- Transformers 固定版 `Qwen3DecoderLayer`：
  https://github.com/huggingface/transformers/blob/0dad7b822255a0ae261ec45ae937371e859ffd1a/src/transformers/models/qwen3/modeling_qwen3.py

本卡只看：
- `Qwen3DecoderLayer.__init__`
- `Qwen3DecoderLayer.forward`

### 辅助参考（只看结构）
- vLLM `Qwen3DecoderLayer`
- nano-vllm `Qwen3DecoderLayer`

不要引入 KV cache、residual dtype 优化、pipeline/tensor parallel、cache object 等工程逻辑。

## 算法核心

### 1. DecoderLayer 结构

Qwen3 decoder layer 是标准 pre-norm Transformer block：

```text
hidden_states
  -> residual = hidden_states
  -> input_layernorm
  -> self_attn
  -> residual add
  -> residual = hidden_states
  -> post_attention_layernorm
  -> mlp
  -> residual add
  -> output
```

### 2. 模块骨架

```python
class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GQAAttention(config)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

### 3. forward 骨架

```python
def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, position_ids)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | 输出 shape 保持 `[B, T, H]` | 手工断言 | exact |
| 2 | 子模块结构和命名 | `Qwen3DecoderLayer` | exact / hasattr |
| 3 | residual 两段连接顺序 | 手工 monkeypatch 子模块 | exact |
| 4 | 小尺寸 layer vs transformers | `Qwen3DecoderLayer` 同权重同输入 | fp32 `1e-5` |
| 5 | Qwen3-0.6B 可构造 | `ModelConfig.qwen3_0_6b()` | no error / shape |

## DoD
- [ ] `tests/unit/test_decoder_layer.py` 全绿
- [ ] 小尺寸 `DecoderLayer` 与 transformers `Qwen3DecoderLayer` fp32 对齐
- [ ] `uv run pytest tests/unit/test_config.py tests/unit/test_rmsnorm.py tests/unit/test_mlp.py tests/unit/test_rope.py tests/unit/test_attention.py tests/unit/test_decoder_layer.py -q` 全绿
- [ ] `docs/kb/knowledge.md` 补 DecoderLayer / pre-norm 知识卡（如缺）
- [ ] `docs/tasks/README.md` 与 `docs/plan/M1.md` 状态更新
- [ ] `README.md` 当前进度同步（若本任务完成）
- [ ] commit `feat(model): add Qwen3 decoder layer`

## 坑（按概率排序）
1. **Norm 顺序写错**：Qwen3 是 pre-norm，不是 post-norm；先 norm 再 attention/mlp，最后 residual add。
2. **第二段 residual 取错**：MLP 前要重新 `residual = hidden_states`，不能复用 attention 前的 residual。
3. **RMSNorm 维度错**：decoder layer 的两个 layernorm 是 hidden_size 维，不是 head_dim。
4. **Attention 输入 position_ids 漏传**：`self.self_attn(hidden_states, position_ids)`。
5. **测试对齐 transformers 时 position_embeddings 接口差异**：transformers `Qwen3DecoderLayer` 可能内部 attention 需要 `(cos, sin)` 或 model 层传入，测试要按本地版本签名适配。
6. **不要在 T5 引入 embedding/lm_head**：这些属于 T6/T8。
