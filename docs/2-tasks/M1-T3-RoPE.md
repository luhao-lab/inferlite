# M1-T3 RoPE

## 元信息
- **任务 ID**: T3
- **里程碑**: M1·P1（数值对齐）
- **状态**: 🟡 in-progress
- **前置**: T0 `ModelConfig`
- **估时**: 2h

## 目标
实现 Qwen3 使用的 RoPE（Rotary Position Embedding）基础算子：根据 `position_ids` 生成 `cos/sin`，并把旋转位置编码应用到 attention 的 `q/k` 上，为 T4 GQA Attention 做准备。

## 产出文件
- `inferlite/model/layers.py::RotaryEmbedding`
- `inferlite/model/layers.py::rotate_half`
- `inferlite/model/layers.py::apply_rotary_pos_emb`
- `tests/unit/test_rope.py`

## 算法核心

### 1. 生成频率
Qwen3 默认 RoPE 参数：

```text
base = rope_theta = 1_000_000
head_dim = 128  # 独立超参，不是 hidden_size / num_heads
inv_freq[i] = 1 / base ** (i / head_dim), i = 0, 2, 4, ...
```

对 `position_ids: [B, T]`：

```text
freqs = position_ids @ inv_freq
emb = concat(freqs, freqs)  # [..., head_dim]
cos = cos(emb)
sin = sin(emb)
```

### 2. 旋转 q/k
Transformers Qwen3 ground truth：

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

T3 只做 default RoPE，不实现 dynamic/yarn/longrope 等扩展。

## 参考代码

- **主参考（数值真值）**：Transformers 固定版 `modeling_qwen3.py`
  - https://github.com/huggingface/transformers/blob/0dad7b822255a0ae261ec45ae937371e859ffd1a/src/transformers/models/qwen3/modeling_qwen3.py
  - 本卡只看 `Qwen3RotaryEmbedding`、`rotate_half`、`apply_rotary_pos_emb`。
- **辅助参考（暂不实现）**：vLLM `get_rope` / `Qwen3Attention`
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py
  - 只用于理解推理框架如何把 RoPE 接到 Attention；T3 不照搬 vLLM cache/TP/quant 逻辑。

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | `rotate_half` 输出 | transformers `rotate_half` | exact / allclose |
| 2 | `RotaryEmbedding` 生成 `cos/sin` | `Qwen3RotaryEmbedding` | fp32 `1e-6` |
| 3 | `apply_rotary_pos_emb` 旋转 q/k | transformers `apply_rotary_pos_emb` | fp32 `1e-6`，低精度放宽 |
| 4 | shape/broadcast | 手工断言 | shape exact |
| 5 | Qwen3-0.6B 参数 | config shape | `inv_freq.shape == [64]`，`cos/sin == [B,T,128]` |

## DoD
- [ ] `tests/unit/test_rope.py` 全绿
- [ ] 已确认 `head_dim=128` 路径，不误用 `hidden_size / num_attention_heads`
- [ ] `uv run pytest tests/unit/test_config.py tests/unit/test_rmsnorm.py tests/unit/test_mlp.py tests/unit/test_rope.py -q` 全绿
- [ ] `docs/3-kb/knowledge.md` 补 RoPE 知识卡
- [ ] `docs/2-tasks/README.md` 与 `docs/1-plan/M1.md` 状态更新
- [ ] commit `feat(model): add RoPE aligned with Qwen3RotaryEmbedding`

## 坑（按概率排序）
1. **`head_dim` 用错**：Qwen3-0.6B 是 128，不是 `1024 / 16 = 64`。
2. **`rotate_half` 切分方式错**：transformers Qwen3 是前半/后半切分，不是 even/odd interleave。
3. **广播维度错**：T4 attention 的 q/k 预计是 `[B, num_heads, T, head_dim]`，因此 `unsqueeze_dim=1`。
4. **dtype 错**：生成频率时应 fp32 计算 `cos/sin`，最后 cast 回输入 dtype。
5. **RoPE 只作用于 q/k，不作用于 v**。
