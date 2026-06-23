# M1-T4 GQA Attention

## 元信息
- **任务 ID**: T4
- **里程碑**: M1·P1（数值对齐）
- **状态**: 🟡 in-progress
- **前置**: T1 `RMSNorm`，T3 `RoPE`
- **估时**: 3h

## 目标
实现 Qwen3 decoder self-attention 的最小数值对齐版：包含 q/k/v/o 四个无 bias 线性层、QK-norm、RoPE、GQA 的 KV head repeat、causal mask 和 scaled dot-product attention，为 T5 `DecoderLayer` 做准备。

## 产出文件
- `inferlite/model/attention.py::GQAAttention`（推荐新文件，避免 `layers.py` 继续膨胀）
- `tests/unit/test_attention.py`
- 如有必要：`inferlite/model/__init__.py` 导出

## 参考代码

### 主参考（数值真值）
- Transformers 固定版 `Qwen3Attention`：
  https://github.com/huggingface/transformers/blob/0dad7b822255a0ae261ec45ae937371e859ffd1a/src/transformers/models/qwen3/modeling_qwen3.py

本卡只看：
- `Qwen3Attention.__init__`
- `Qwen3Attention.forward`
- `repeat_kv`
- `eager_attention_forward` / attention interface 调用方式

### 辅助参考（只看工程组织）
- vLLM `Qwen3Attention`：
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py
- nano-vllm `Qwen3Attention`：
  `nano-vllm/nanovllm/models/qwen3.py`

只借鉴组织方式，不照搬 TP/quant/cache/kernel。

## 算法核心

### 1. 参数与形状
Qwen3-0.6B 关键参数：

```text
hidden_size = 1024
num_attention_heads = 16      # query heads
num_key_value_heads = 8       # key/value heads
head_dim = 128                # 独立超参
num_key_value_groups = 16 / 8 = 2
```

T4 中张量推荐形状：

```text
hidden_states: [B, T, H]
q:             [B, n_q,  T, D]
k/v:           [B, n_kv, T, D]
cos/sin:       [B, T, D]
attn_scores:   [B, n_q, T, T]
output:        [B, T, H]
```

### 2. 模块结构

```python
class GQAAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)
```

### 3. forward 骨架

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # [B, n_kv, T, D] -> [B, n_q, T, D]
    if n_rep == 1:
        return hidden_states
    b, n_kv, t, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, t, d)
    return hidden_states.reshape(b, n_kv * n_rep, t, d)


def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.shape

    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    q = self.q_norm(q)
    k = self.k_norm(k)

    cos, sin = self.rotary_emb(q, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
    causal_mask = torch.triu(
        torch.ones(q_len, q_len, dtype=torch.bool, device=hidden_states.device),
        diagonal=1,
    )
    attn_weights = attn_weights.masked_fill(causal_mask, torch.finfo(attn_weights.dtype).min)
    attn_weights = torch.softmax(attn_weights, dim=-1)

    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
    return self.o_proj(attn_output)
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | `repeat_kv` shape 与内容 | transformers `repeat_kv` | exact |
| 2 | Linear 权重 shape | Qwen3Config / ModelConfig | exact |
| 3 | `GQAAttention` 输出 shape | 手工断言 | exact |
| 4 | QK-norm 是否存在且 shape 正确 | transformers `Qwen3Attention` | exact / state_dict |
| 5 | 小尺寸 attention vs transformers | `Qwen3Attention` 同权重同输入 | fp32 `1e-5`；低精度暂可不测 |
| 6 | causal mask 生效 | 手工构造输入，未来 token 不影响过去输出 | allclose |
| 7 | Qwen3-0.6B 参数形状 | `ModelConfig.qwen3_0_6b()` | exact |

## DoD
- [ ] `tests/unit/test_attention.py` 全绿
- [ ] `repeat_kv` 与 transformers 行为一致
- [ ] `GQAAttention` 可与 transformers 小尺寸 `Qwen3Attention` fp32 对齐
- [ ] `uv run pytest tests/unit/test_config.py tests/unit/test_rmsnorm.py tests/unit/test_mlp.py tests/unit/test_rope.py tests/unit/test_attention.py -q` 全绿
- [ ] `docs/kb/knowledge.md` 补 GQA/QK-norm 知识卡
- [ ] `docs/tasks/README.md` 与 `docs/plan/M1.md` 状态更新
- [ ] `README.md` 当前进度同步（若本任务完成）
- [ ] commit `feat(model): add GQA attention aligned with Qwen3Attention`

## 坑（按概率排序）
1. **head_dim 用错**：Qwen3-0.6B 是 `128`，不要用 `hidden_size / num_heads = 64`。
2. **QK-norm 漏掉**：Qwen3 Attention 在 RoPE 前有 `q_norm/k_norm`，这是 Qwen3 相比部分 LLaMA-like 模型的重要差异。
3. **GQA repeat 方向错**：是把 `n_kv` repeat 到 `n_q`，不是减少 query head。
4. **RoPE 位置错**：顺序是 `q/k projection -> q/k norm -> RoPE -> repeat_kv -> attention`。
5. **causal mask dtype/设备错**：mask 必须在同 device；masked value 对 fp16/bf16 要用 `torch.finfo(dtype).min`。
6. **softmax upcast 差异**：transformers attention 可能内部 fp32 softmax；若 fp32 对齐先过，低精度差异后续再收敛。
7. **输出 reshape 漏 contiguous**：`transpose` 后 `.view` 前需要 `.contiguous()` 或用 `.reshape()`。
8. **不要引入 KV cache**：T4 是 prefill/full attention；KV cache 留到 M2。
