# M1-T8 L1 logits 对齐

## 元信息
- **任务 ID**: T8
- **里程碑**: M1·P1（数值对齐）
- **状态**: 🟡 in-progress
- **前置**: T7 `WeightMap + load_from_hf`
- **估时**: 2h

## 背景
T6 已实现 `Qwen3Model` backbone：

```text
input_ids -> embed_tokens -> DecoderLayer stack -> final RMSNorm -> last_hidden_state
```

T7 已实现 HF / ModelScope 权重加载：

```text
model_dir -> config.json -> Qwen3Model(config) -> model.safetensors -> load_state_dict
```

但目前还没有 CausalLM 外壳，所以不能得到 vocab logits。T8 要补齐：

```text
last_hidden_state -> lm_head -> logits
```

并在小尺寸模型上与 transformers `Qwen3ForCausalLM` 做 L1 logits 对齐。

## 目标
实现最小 `Qwen3ForCausalLM` / `Qwen3CausalLM` 外壳，让 inferlite 能输出 logits：

```text
input_ids [B, T]
  -> Qwen3Model
  -> last_hidden_state [B, T, H]
  -> lm_head [H, V]
  -> logits [B, T, V]
```

T8 完成后，应能在 fp32 小尺寸配置下与 transformers `Qwen3ForCausalLM` logits 数值对齐。

## 非目标
- 不做 loss。
- 不做 sampling / generate。
- 不做 KV cache。
- 不做 tokenizer / chat template。
- 不做真实 0.6B 全量 logits 回归作为必须单测。
- 不做多卡 / tensor parallel / 量化。

## 产出文件
- `inferlite/model/qwen3.py`
  - 新增 `Qwen3ForCausalLM` 或项目内命名一致的 `Qwen3CausalLM`
- `tests/unit/test_qwen3_causal_lm.py` 或 `tests/unit/test_logits.py`
- 如有必要：
  - `inferlite/model/weights.py` 支持加载/跳过 `lm_head.weight` 的策略调整
  - `inferlite/model/__init__.py` 导出

## API 草案

### 模型类

```python
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, position_ids=position_ids)
        logits = self.lm_head(hidden_states)
        return logits
```

### 权重加载策略

T7 当前 `load_from_hf` 返回 backbone `Qwen3Model`，并跳过 `lm_head.weight`。
T8 有两种路线：

#### 路线 A：新增 CausalLM 专用加载入口（推荐）

```python
def load_causal_lm_from_hf(model_dir: str | Path, *, strict: bool = True) -> Qwen3ForCausalLM:
    config = ModelConfig.from_json(model_dir / "config.json")
    model = Qwen3ForCausalLM(config)
    load_weights_into_model(model, model_dir, strict=strict)
    return model
```

为了支持这个路线，key 映射要区分：

```text
HF key                         inferlite CausalLM key
model.embed_tokens.weight  ->  model.embed_tokens.weight
model.layers.0...          ->  model.layers.0...
model.norm.weight          ->  model.norm.weight
lm_head.weight             ->  lm_head.weight
```

注意：如果 CausalLM 外壳里字段名也叫 `model`，那 transformers HF key 和 inferlite key 可以天然一致。

#### 路线 B：继续只加载 backbone，测试里手动复制 lm_head

该路线改动更少，但不形成完整 CausalLM 加载能力。T8 目标是 logits 对齐，建议优先路线 A。

## tied embedding 注意点
Qwen3-0.6B `tie_word_embeddings=True`，表示 `lm_head.weight` 和 `embed_tokens.weight` 可能共享语义。

T8 最小策略：

```text
先按 HF checkpoint 显式加载 lm_head.weight。
如果 checkpoint 没有 lm_head.weight 且 tie_word_embeddings=True，
则可以让 lm_head.weight 复用 embed_tokens.weight。
```

但是否真的共享同一个 Parameter 对象，可以先不强求；T8 重点是 logits 数值对齐。

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | CausalLM 结构 | `model` + `lm_head` | exact |
| 2 | logits shape | `[B, T, vocab_size]` | exact |
| 3 | 小尺寸 fp32 logits 对齐 | transformers `Qwen3ForCausalLM` | `1e-5` |
| 4 | `lm_head.weight` 可加载 | fake safetensors | exact |
| 5 | tie_word_embeddings 场景 | HF config / fake checkpoint | exact |
| 6 | T0-T8 回归 | 现有单测 + 新单测 | all green |

## 建议单测设计

### 1. 结构测试

```python
cfg = _tiny_model_config()
model = Qwen3ForCausalLM(cfg)
assert isinstance(model.model, Qwen3Model)
assert tuple(model.lm_head.weight.shape) == (cfg.vocab_size, cfg.hidden_size)
```

### 2. shape 测试

```python
input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
logits = model(input_ids)
assert logits.shape == (2, 5, cfg.vocab_size)
```

### 3. transformers logits 对齐

```text
构造 tiny transformers.Qwen3ForCausalLM
复制权重到 inferlite.Qwen3ForCausalLM
固定 input_ids / position_ids
比较 logits allclose
```

### 4. fake safetensors 加载

```text
source Qwen3ForCausalLM state_dict
  -> 加 HF key 前缀 / 保存 model.safetensors
  -> load_causal_lm_from_hf
  -> 对比所有参数 exact equal
```

## DoD
- [ ] `Qwen3ForCausalLM` / `Qwen3CausalLM` 类实现
- [ ] logits 输出 shape 正确
- [ ] 小尺寸 fp32 logits 与 transformers `Qwen3ForCausalLM` 对齐
- [ ] `lm_head.weight` 权重加载逻辑清晰
- [ ] `tests/unit/test_qwen3_causal_lm.py` 或等价测试全绿
- [ ] T0-T8 回归全绿
- [ ] `docs/kb/knowledge.md` 补 logits / lm_head / tied embedding 知识点（如缺）
- [ ] `docs/tasks/README.md` 与 `docs/plan/M1.md` 状态更新
- [ ] `README.md` 当前进度同步（任务完成时）
- [ ] commit `feat(model): add Qwen3 causal LM logits`

## 坑（按概率排序）
1. **外壳字段名影响权重 key**：如果类里叫 `self.model`，key 会天然接近 HF；如果叫 `self.backbone`，就要额外映射。
2. **lm_head.weight shape**：`nn.Linear(hidden_size, vocab_size, bias=False)` 的 weight shape 是 `[vocab_size, hidden_size]`。
3. **tie_word_embeddings**：0.6B config 为 True，不要忽略 checkpoint 是否真的含 `lm_head.weight`。
4. **返回类型边界**：T8 可先返回 logits Tensor，不必模拟 HF 的 `CausalLMOutputWithPast`。
5. **测试对齐时 causal mask / position_ids**：保持和 T6 对齐测试一样，显式传 `position_ids`。
6. **不要引入 generate**：M1·P1 只做 logits，采样属于后续 P2。
