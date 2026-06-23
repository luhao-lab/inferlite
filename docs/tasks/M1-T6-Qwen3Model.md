# M1-T6 Qwen3Model

## 元信息
- **任务 ID**: T6
- **里程碑**: M1·P1（数值对齐）
- **状态**: 🟡 in-progress
- **前置**: T1 `RMSNorm`，T5 `DecoderLayer`
- **估时**: 2h

## 目标
实现 Qwen3 主干模型的最小数值对齐版：`embed_tokens` + `ModuleList[DecoderLayer]` + final `RMSNorm`。T6 只负责把 decoder layers 堆起来并输出最后 hidden states；不实现真实权重加载、不实现 lm_head logits、不实现文本生成。

## 产出文件
- `inferlite/model/qwen3.py::Qwen3Model`
- `tests/unit/test_qwen3_model.py`
- 如有必要：`inferlite/model/__init__.py` 导出

## 参考代码

### 主参考（数值真值）
- Transformers 固定版 `Qwen3Model`：
  https://github.com/huggingface/transformers/blob/0dad7b822255a0ae261ec45ae937371e859ffd1a/src/transformers/models/qwen3/modeling_qwen3.py

本卡只看：
- `Qwen3Model.__init__`
- `Qwen3Model.forward`
- `Qwen3Model` 如何生成 `position_ids`
- `Qwen3Model` 如何把 shared position embeddings 传给每层

### 暂不看 / 暂不做
- `Qwen3ForCausalLM`
- `lm_head`
- `loss`
- `past_key_values` / KV cache
- `attention_mask` 复杂 padding 场景
- gradient checkpointing
- flash attention / sdpa 分支

## 算法核心

### 1. 模型结构

```text
input_ids [B, T]
  -> embed_tokens
  -> hidden_states [B, T, H]
  -> layers[0]
  -> layers[1]
  -> ...
  -> layers[N-1]
  -> final norm
  -> last_hidden_state [B, T, H]
```

### 2. 模块骨架

```python
class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

### 3. forward 骨架

```python
def forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

    hidden_states = self.embed_tokens(input_ids)
    for layer in self.layers:
        hidden_states = layer(hidden_states, position_ids)
    hidden_states = self.norm(hidden_states)
    return hidden_states
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | 构造后模块数量与 shape | `ModelConfig` | exact |
| 2 | `input_ids -> hidden_states` 输出 shape | 手工断言 | exact |
| 3 | `position_ids=None` 自动生成 | monkeypatch / 对比显式 position_ids | allclose |
| 4 | layer 堆叠顺序 | monkeypatch fake layers | exact |
| 5 | 小尺寸 `Qwen3Model` vs transformers | `transformers.Qwen3Model` 同权重同输入 | fp32 `1e-5` |
| 6 | Qwen3-0.6B 可构造 | `ModelConfig.qwen3_0_6b()` | no error / shape |

## DoD
- [ ] `tests/unit/test_qwen3_model.py` 全绿
- [ ] 小尺寸 `Qwen3Model` 与 transformers `Qwen3Model` fp32 对齐
- [ ] `uv run pytest tests/unit/test_config.py tests/unit/test_rmsnorm.py tests/unit/test_mlp.py tests/unit/test_rope.py tests/unit/test_attention.py tests/unit/test_decoder_layer.py tests/unit/test_qwen3_model.py -q` 全绿
- [ ] `docs/kb/knowledge.md` 补 Qwen3Model / position_ids / shared RoPE 知识卡（如缺）
- [ ] `docs/tasks/README.md` 与 `docs/plan/M1.md` 状态更新
- [ ] `README.md` 当前进度同步（若本任务完成）
- [ ] commit `feat(model): add Qwen3Model backbone`

## 坑（按概率排序）
1. **T6 不是 CausalLM**：不要加 `lm_head`；logits 属于 T8。
2. **position_ids 设备错**：`torch.arange` 必须在 `input_ids.device` 上。
3. **position_ids shape 错**：应为 `[B, T]`，不是 `[T]`。
4. **层数太多测试慢**：小尺寸对齐测试只用 `num_hidden_layers=2`；真实 28 层只做构造/shape。
5. **权重对齐路径复杂**：T6 测试只复制小尺寸 transformers 权重；真实 safetensors 映射留到 T7。
6. **transformers 返回对象不同**：`Qwen3Model.forward` 可能返回 dataclass 或 tuple，测试要按本地版本适配取 `last_hidden_state`。
7. **shared position embeddings 差异**：本项目当前让每层 attention 内部根据同一 `position_ids` 算 RoPE；transformers 可能在 model 层统一计算 `(cos, sin)` 后传入各层。T6 对齐测试需按数值等价而不是接口一致。
