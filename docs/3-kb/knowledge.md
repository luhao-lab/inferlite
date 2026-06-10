# Knowledge Cards

> 项目用到的论文 / 库 / 概念 / 工具，原子卡片，可被任务卡/lessons 引用。
> 单文件多 H2，按主题查找，搜索友好。
> 新增卡片直接在对应章节末尾追加。

---

## 📊 索引摘要

> 自动维护：每次新增/删除卡片时同步更新本段。最后更新：2026-06-09。

**总览**：4 类共 **23 张** knowledge 卡片 + 4 条 lessons + 3 个 ADR。

| 章节 | 卡片数 | 列表（点击跳转） |
| --- | --- | --- |
| **Papers** | 5 | [RMSNorm](#rmsnorm-zhang-sennrich-neurips-2019) · [Qwen3 Tech Report](#qwen3-tech-report) · [SwiGLU](#swiglu-shazeer-2020) · [RoPE](#rope-su-et-al-2021) · [GQA](#gqa-ainslie-et-al-2023) |
| **Libraries** | 6 | [transformers Qwen3 模块](#transformers-qwen3-ground-truth) · [vLLM Qwen3 weight loading](#vllm-qwen3-weight-loading) · [transformers 5 核心对象](#transformers-hf) · [pytest](#pytest-api) · [modelscope](#modelscope-snapshot_download) · [huggingface_hub 1.x](#huggingface_hub-1x) |
| **Concepts** | 7 | [upcast fp32](#upcast-to-fp32) · [数值对齐策略](#numeric-alignment-strategy) · [tie_word_embeddings](#tie_word_embeddings) · [形状速查](#shape-cheatsheet) · [推理上下文](#inference-context) · [Python dataclass](#python-dataclass) · [Factory pattern](#factory-pattern) |
| **Tools** | 5 | [uv](#uvpython) · [make](#make) · [ruff](#rufflint-format) · [pre-commit](#pre-commitcommit) · [pytest-mark / CI](#pytest-mark-ci-matrix) |

**已沉淀的 lessons**（[详见 lessons.md](./lessons.md)）：

- **L1** RMSNorm 必须 upcast fp32 算方差（→ Concepts#upcast）
- **L2** 国内拉模型用 ModelScope 替代 HF mirror（→ Libraries#modelscope, #huggingface_hub）
- **L3** 地基 vs 算法是两个频道，不要混着切（→ 协作节奏）
- **L4** GQA 的 `head_dim` 是独立超参，不能从 KV 头数推导（→ Papers#Qwen3, Concepts#shape）

**已落地的 ADR**（[详见 decisions.md](./decisions.md)）：

- **ADR-001** spec-driven 工作流 + 双文件知识库
- **ADR-002** 知识库与代码同仓（R1 重构）
- **ADR-003** 分组目录 + MkDocs Material 可视化

**任务卡进度**（[详见 PROGRESS](../1-plan/PROGRESS.md)）：M1-T0 ModelConfig ✅ · M1-T1 RMSNorm ✅ · M1-T2 SwiGLUMLP ✅ · M1-T3 RoPE ✅ · T4-T6 ⬜

**已知缺口**（开工时回填）：

- ⚠️ `Concepts#KV Cache 结构` — 等 M2 任务卡触发
- ⚠️ `Concepts#Continuous Batching 调度` — 等 M3 任务卡触发

---

## Papers

### RMSNorm (Zhang & Sennrich, NeurIPS 2019)

**一句话**：LayerNorm 的简化版（去中心化 + 去偏置 β），LLaMA/Qwen/Mistral 标配。

**公式**：

LayerNorm:  `(x - μ) / σ · γ + β`（2 stats + 2 params）

RMSNorm:    `x / sqrt(mean(x²) + ε) · γ`（1 stat + 1 param）

**关键论点**：
1. 去中心化对训练几乎不影响（"激活分布形状不变"）
2. 参数减半（无 β）
3. 速度比 LN 快 7%-64%（看 framework）
4. NLU/MT 任务上效果持平甚至略好

**本项目对应**：
- 文件：`inferlite/model/layers.py::RMSNorm` (T1)
- ε：Qwen3 用 1e-6（LLaMA-2 同；LLaMA-1 是 1e-5）
- 实现注意：必须 upcast fp32 算 var（见下方 Concepts）

**外部参考**：
- 论文：https://arxiv.org/abs/1910.07467
- transformers：`transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm`
- nano-vllm：`nanovllm/layers/layernorm.py::RMSNorm`

### Qwen3 Tech Report

> 论文：[Qwen3 Technical Report (2025-05)](https://arxiv.org/abs/2505.09388)
> 我们做的是 **Qwen3-0.6B (dense, base)**，所以下面只摘和这个尺寸相关的事实。

#### 1. 架构家谱：Llama-style + GQA + Q/K-norm 微调

Qwen3 整体仍是 Llama 同构体（pre-norm + RMSNorm + RoPE + SwiGLU + GQA），相比 Llama-3 关键差别：

| 维度 | Llama-3 | **Qwen3-0.6B** | 备注 |
| --- | --- | --- | --- |
| Norm | RMSNorm | **RMSNorm** | 一致 |
| 位置编码 | RoPE (θ=500K) | **RoPE (θ=1M)** | Qwen3 训练上下文 32K → θ 加大 |
| 注意力 | GQA | **GQA + Q/K LayerNorm** | Qwen3 在 Q/K 投影后再 norm 一道，稳数值 |
| FFN | SwiGLU | **SwiGLU** | 一致 |
| MoE | 无（dense） | 无（0.6B 是 dense） | 只有 30B/235B 是 MoE |
| Tie embed | 无 | **有（0.6B 特例）** | ≥1.7B 不 tie |
| Attention bias | False | **False** | Llama 一脉相承 |

**M1·P1 数值对齐时**：Q/K-norm 这条会让我们的 `GQAAttention` 比纯 Llama 多两个 RMSNorm（`q_norm` / `k_norm`），不要漏掉。

#### 2. Forward dataflow（推理 path，单序列）

```
input_ids  [B, T]                     ← B=batch, T=seq_len
   │
   ▼ embed_tokens (V × H)
hidden_states  [B, T, H=1024]
   │
   ▼ × 28 层 DecoderLayer：
   │     residual = h
   │     h = input_layernorm(h)              # RMSNorm, [B,T,H]
   │     h = self_attn(h, position_ids)      # GQA + RoPE + Q/K-norm
   │     h = h + residual
   │     residual = h
   │     h = post_attention_layernorm(h)     # RMSNorm
   │     h = mlp(h)                          # SwiGLU: down(silu(gate(h)) * up(h))
   │     h = h + residual
   │
   ▼ norm (final RMSNorm)
hidden_states  [B, T, H]
   │
   ▼ lm_head (= embed_tokens.weight^T，因 tie_word_embeddings=True)
logits  [B, T, V=151936]
   │
   ▼ argmax / sampler
next_token_id
```

#### 3. Self-Attention 内部（GQA + Q/K-norm + RoPE）

```
h  [B, T, H=1024]
 ├──► q_proj (H × n_q·d) → Q  [B, T, n_q=16, d=128]   ← d != H/n_q 见下
 ├──► k_proj (H × n_kv·d) → K  [B, T, n_kv=8,  d=128]
 └──► v_proj (H × n_kv·d) → V  [B, T, n_kv=8,  d=128]

Q = q_norm(Q)        # ⚠️ Qwen3 特色：投影后还要 RMSNorm 一次
K = k_norm(K)

(Q, K) = apply_rope(Q, K, position_ids, base=1e6)

K_full = repeat_kv(K, group=n_q/n_kv=2)   # [B, T, 16, 128]
V_full = repeat_kv(V, group=2)

attn = softmax(Q @ K_full^T / √d, dim=-1) @ V_full   # [B, T, 16, 128]
out  = o_proj(attn.flatten(-2))                       # [B, T, H]
```

**关键形状事实**：
- `head_dim=128`，但 `H/n_q = 1024/16 = 64` —— **head_dim 是独立超参**（Qwen3 选 128 是为了 RoPE 频谱够大 + 数值稳；不是手滑）
- `q_proj` 输出维度 = `n_q × d = 2048`，**不是** `H = 1024`；`k_proj/v_proj` 输出 = `n_kv × d = 1024`
- GQA 比 = `n_q : n_kv = 2 : 1`，每 2 个 Q 头共用 1 组 KV → KV cache 直接砍半

#### 4. ModelConfig 11 字段 × 含义/值/为什么（**T0 直接对照表**）

| # | 字段 | 类型 | Qwen3-0.6B 值 | 含义 / 为什么是这个值 |
| --- | --- | --- | --- | --- |
| 1 | `hidden_size` | int | **1024** | residual stream 维度 H；所有 layer 输入输出都是 [B,T,H] |
| 2 | `num_hidden_layers` | int | **28** | 堆 28 个 DecoderLayer；T6 写 `Qwen3Model` 时 `nn.ModuleList(...)` |
| 3 | `num_attention_heads` | int | **16** | Query 头数 n_q；softmax 在每个头独立做 |
| 4 | `num_key_value_heads` | int | **8** | KV 头数 n_kv；GQA 把 n_q 个 Q 分成 n_kv 组共用 KV |
| 5 | `head_dim` | int | **128** | 每个 head 的维度 d；**独立参数**，≠ H/n_q (=64) |
| 6 | `intermediate_size` | int | **3072** | SwiGLU 中间层 I；MLP = down(silu(gate(h)) * up(h))，gate/up 都是 H→I |
| 7 | `vocab_size` | int | **151936** | tokenizer 词表大小；embed/lm_head 行数 |
| 8 | `max_position_embeddings` | int | **40960** | 训练时见过的最大 ctx；M11 YaRN 之前不要超 |
| 9 | `rope_theta` | float | **1e6** | RoPE 基频 base；Qwen3 加大到 1M（Llama-3=500K，Llama-2=10K）→ 长 ctx 外推友好 |
| 10 | `rms_norm_eps` | float | **1e-6** | RMSNorm ε；Qwen 系列统一 1e-6（≠ Llama-1 的 1e-5） |
| 11 | `tie_word_embeddings` | bool | **True** | 0.6B 特有：lm_head.weight 复用 embed_tokens.weight；省 V·H ≈ 156M 参数 |

**为啥就这 11 个？** 写完 T0-T8 用到的恰好就是这 11 个；Qwen3 完整 config 还有 `sliding_window` / `attention_bias` / `rope_scaling` / `attention_dropout` 等十几个字段，**不在 M1 路径上**所以 T0 不收（M11 YaRN 时回头加 `rope_scaling`；M2 prefix cache 复杂场景时再考虑 `sliding_window`）。

#### 5. 0.6B 参数量校验（写 T7 `load_from_hf` 时核对）

```
embed:           V × H              = 151936 × 1024 ≈ 156M
each layer:
  q_proj:        H × (n_q·d)        = 1024 × 2048   = 2.1M
  k_proj:        H × (n_kv·d)       = 1024 × 1024   = 1.0M
  v_proj:        H × (n_kv·d)       = 1024 × 1024   = 1.0M
  o_proj:        (n_q·d) × H        = 2048 × 1024   = 2.1M
  gate_proj:     H × I              = 1024 × 3072   = 3.1M
  up_proj:       H × I              = 1024 × 3072   = 3.1M
  down_proj:     I × H              = 3072 × 1024   = 3.1M
  q_norm/k_norm + 2 RMSNorm:                         ≈ 4·d (negligible)
  → 每层 ≈ 15.5M × 28 层 ≈ 434M
final norm:                                          negligible
lm_head:        tied with embed (0)
total ≈ 156M + 434M ≈ 590M  ✅
```

报告里 "Qwen3-0.6B" 实际 ~600M 参数（取整命名），对得上。

#### 6. 必读源码（写代码时打开对照）

- `transformers/models/qwen3/modeling_qwen3.py`
  - `Qwen3RMSNorm` — T1（已对齐）
  - `Qwen3MLP` — T2 SwiGLU
  - `Qwen3RotaryEmbedding` + `apply_rotary_pos_emb` — T3
  - `Qwen3Attention` — T4（注意 `q_norm` / `k_norm`）
  - `Qwen3DecoderLayer` — T5
  - `Qwen3Model.forward` — T6 整体 dataflow
- `transformers/models/qwen3/configuration_qwen3.py::Qwen3Config` — **T0 11 字段名字的真实来源**

#### 7. 外部参考

- 论文：<https://arxiv.org/abs/2505.09388>
- 模型卡：<https://huggingface.co/Qwen/Qwen3-0.6B>
- HF 源码：<https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py>
- ModelScope 镜像：<https://www.modelscope.cn/models/Qwen/Qwen3-0.6B>


### SwiGLU (Shazeer 2020)

> 论文：Noam Shazeer, *GLU Variants Improve Transformer*, arXiv:2002.05202.

**一句话**：SwiGLU 把 Transformer FFN 的“单路激活”换成“双路线性投影相乘”：一条 gate 路过 Swish/SiLU，一条 up 路保留线性信号，再逐元素相乘后 down 回 hidden size。

#### 1. 从 FFN 到 SwiGLU

经典无 bias FFN：

```text
FFN_ReLU(x) = ReLU(x W1) W2
```

GLU 族：

```text
GLU(x)    = sigmoid(x W) ⊙ (x V)
GEGLU(x)  = GELU(x W)    ⊙ (x V)
SwiGLU(x) = Swish(x W)   ⊙ (x V)
```

Transformer FFN 版 SwiGLU：

```text
FFN_SwiGLU(x) = (Swish(x W_gate) ⊙ (x W_up)) W_down
```

PyTorch / Qwen3 写法：

```python
y = down_proj(F.silu(gate_proj(x)) * up_proj(x))
```

#### 2. 为什么是 3 个矩阵

| 路径 | 形状 | 作用 |
| --- | --- | --- |
| `gate_proj` | H → I | 产生门控信号，过 `silu` |
| `up_proj` | H → I | 产生被门控的内容信号 |
| `down_proj` | I → H | 回到 residual stream 维度 |

相比普通 FFN 的 2 个矩阵，SwiGLU 多一条 gate 路。论文为了参数量/计算量对齐，会把中间维度降到原 FFN 的约 2/3；Qwen3-0.6B 已经在 config 里给出最终取值 `intermediate_size=3072`，实现时直接用 config，不再自行按比例推导。

#### 3. Qwen3-0.6B 对应参数

```text
H = hidden_size = 1024
I = intermediate_size = 3072
bias = False
hidden_act = "silu"
```

对应权重形状：

```text
gate_proj.weight: [I, H] = [3072, 1024]
up_proj.weight:   [I, H] = [3072, 1024]
down_proj.weight: [H, I] = [1024, 3072]
```

#### 4. transformers ground truth

`transformers.models.qwen3.modeling_qwen3.Qwen3MLP` 核心逻辑：

```python
self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
self.act_fn = ACT2FN[config.hidden_act]

def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

#### 5. T2 易错点

1. `nn.Linear` 默认 `bias=True`，必须显式 `bias=False`。
2. 只对 gate 路激活：`F.silu(gate) * up`，不是 `F.silu(gate * up)`。
3. `gate_proj` 与 `up_proj` 数学上形式近似，但权重加载 key 不同，不能交换命名。
4. T2 不需要 upcast；主要是 Linear + element-wise + SiLU，先按 transformers dtype 行为对齐。
5. L0 测试必须同步权重：`mine.load_state_dict(ref.state_dict())` 后再比输出。

**本项目应用**：M1-T2 `inferlite/model/layers.py::SwiGLUMLP`。

### RoPE (Su et al. 2021)

> 论文：Jianlin Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864.

**一句话**：RoPE 不把位置向量加到 hidden state 上，而是在 attention 里对 `q/k` 做按位置变化的二维旋转；这样点积天然只依赖相对位置差。

#### 1. T3 实现边界

本项目 M1-T3 只实现 Qwen3 默认 RoPE：

```text
head_dim = 128
rope_theta = 1_000_000
rope_type = "default"
```

不实现 dynamic/yarn/longrope 等扩展。

#### 2. 核心公式

```text
inv_freq[i] = 1 / rope_theta ** (i / head_dim), i = 0, 2, 4, ...
freqs = position_ids × inv_freq
emb = concat(freqs, freqs)
cos = cos(emb)
sin = sin(emb)
```

旋转：

```python
q_rot = q * cos + rotate_half(q) * sin
k_rot = k * cos + rotate_half(k) * sin
```

其中 Qwen3 的 `rotate_half` 是前半/后半切分：

```python
x1 = x[..., : x.shape[-1] // 2]
x2 = x[..., x.shape[-1] // 2 :]
return torch.cat((-x2, x1), dim=-1)
```

#### 3. T3 易错点

1. `head_dim=128` 是独立超参，不是 `hidden_size / num_attention_heads`。
2. Qwen3/transformers 的 `rotate_half` 不是 even/odd 交错切分。
3. RoPE 只旋转 `q/k`，不旋转 `v`。
4. 对 `[B, heads, T, head_dim]` 的 q/k，`cos/sin` 从 `[B, T, head_dim]` 需要 `unsqueeze(1)`。
5. 生成 `cos/sin` 时用 fp32 算三角函数，最后 cast 回输入 dtype。

**本项目应用**：M1-T3 `inferlite/model/layers.py::RotaryEmbedding`、`rotate_half`、`apply_rotary_pos_emb`。


### GQA (Ainslie et al. 2023)

（待补 —— 开 T4 时生成）

---

## Libraries

### transformers — Qwen3 模块（ground truth）

**项目锁定版本**：`transformers==5.10.2`（已 lock 在 `pyproject.toml`）；切勿随意升级。

**关键类清单**：

| 类名 | 模块 | 对应 inferlite |
| --- | --- | --- |
| `Qwen3Config` | `configuration_qwen3` | `inferlite/config.py::ModelConfig` |
| `Qwen3RMSNorm` | `modeling_qwen3` | `inferlite/model/layers.py::RMSNorm` |
| `Qwen3MLP` | 同上 | `SwiGLUMLP` |
| `Qwen3RotaryEmbedding` | 同上 | `RotaryEmbedding` |
| `Qwen3Attention` | 同上 | `GQAAttention` |
| `Qwen3DecoderLayer` | 同上 | `Qwen3DecoderLayer` |
| `Qwen3Model` | 同上 | `Qwen3Model` |
| `Qwen3ForCausalLM` | 同上 | 不实现（直接 tie embed） |

**标准对齐测试模式**：

```python
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from inferlite.model.layers import RMSNorm

def test_vs_ref():
    H = 1024
    ref = Qwen3RMSNorm(H, eps=1e-6).eval()
    mine = RMSNorm(H, eps=1e-6).eval()
    mine.load_state_dict(ref.state_dict())   # 同步权重
    x = torch.randn(2, 8, H)
    with torch.no_grad():
        assert torch.allclose(mine(x), ref(x), atol=1e-5)
```

**易错点**：
1. `.eval()` 必调（关 dropout）
2. `.load_state_dict()` 必同步（否则 weight 随机 vs ones）
3. `with torch.no_grad():` 节省内存
4. 创建 ref 时传 `Qwen3Config` 实例，避免字段不一致

**Qwen3-0.6B 关键参数**：
- H=1024, I=3072, N=28, num_heads=16, num_kv_heads=8（GQA 2:1）
- head_dim=128, V=151936, rope_theta=1e6, rms_norm_eps=1e-6
- tie_word_embeddings=True, attention_bias=False

**外部参考**：
- 固定源码：https://github.com/huggingface/transformers/blob/0dad7b822255a0ae261ec45ae937371e859ffd1a/src/transformers/models/qwen3/modeling_qwen3.py
- 官方文档：https://huggingface.co/docs/transformers/main/en/model_doc/qwen3

### vLLM — Qwen3 weight loading { #vllm-qwen3-weight-loading }

**一句话**：T7 `WeightMap + load_from_hf` 不应该只看 transformers；vLLM 的 Qwen3 模型是“推理框架如何兼容 HF 权重”的更好参考。

#### 1. 参考文件

固定到当前 `main` commit（2026-06-09 查询）：

- vLLM Qwen3 模型源码：
  <https://github.com/vllm-project/vllm/blob/3f627ebef757e5d575fc33c64250adbc2f2973b4/vllm/model_executor/models/qwen3.py>
- raw：
  <https://raw.githubusercontent.com/vllm-project/vllm/3f627ebef757e5d575fc33c64250adbc2f2973b4/vllm/model_executor/models/qwen3.py>

核心入口：

```python
class Qwen3ForCausalLM(nn.Module, ...):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def load_weights(self, weights):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
```

#### 2. vLLM 做了什么

| vLLM 设计点 | 含义 | inferlite M1 取舍 |
| --- | --- | --- |
| `Qwen3ForCausalLM.load_weights(weights)` | 模型类自己暴露权重加载入口 | T7 可写 `load_from_hf(model, model_dir)`，先函数化，别急着塞进 class |
| `AutoWeightsLoader` | 根据 module/parameter 名自动匹配外部权重 | M1 写显式 `WeightMap`，可读性优先 |
| `skip_prefixes=["lm_head."]` when tied | `tie_word_embeddings=True` 时跳过独立 `lm_head.weight` | T7 必须处理：Qwen3-0.6B 不要求独立 lm_head |
| `packed_modules_mapping` | 工业版把 q/k/v 和 gate/up 合并成 packed projection | M1 不合并，保留 `q_proj/k_proj/v_proj` 和 `gate_proj/up_proj`，T7 只理解它为什么存在 |
| `embedding_modules` | 标注 input/output embedding 的语义映射 | T7 可用来理解 `embed_tokens` 与 `lm_head` 的关系 |
| `QKVParallelLinear` / `RowParallelLinear` | tensor parallel + fused linear | M1 禁用，不引入 TP/quant/kernel |

#### 3. T7 应该借鉴的最小思想

不是照搬 vLLM，而是借鉴三条：

1. **权重加载是模型边界的一部分**
   - checkpoint key 属于外部格式
   - inferlite module key 属于内部格式
   - T7 要显式写清这两者如何映射

2. **tied embedding 是加载策略，不只是 forward 策略**
   - 如果 `tie_word_embeddings=True`，`lm_head.weight` 可以跳过
   - forward 时用 `F.linear(hidden, embed_tokens.weight)` 或把 `lm_head` 指到 `embed_tokens`

3. **工业框架为了性能会 pack 权重，但教学版先不 pack**
   - vLLM `qkv_proj` 对应 HF `q_proj/k_proj/v_proj`
   - vLLM `gate_up_proj` 对应 HF `gate_proj/up_proj`
   - inferlite M1 保持拆开，T8 logits 对齐更直观

#### 4. T7 推荐实现轮廓

```python
def iter_safetensors(model_dir: Path) -> Iterator[tuple[str, torch.Tensor]]:
    # 读 model.safetensors 或 index.json 指向的 shards
    ...

WEIGHT_MAP = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.attn.q_proj.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.attn.k_proj.weight",
    "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.attn.v_proj.weight",
    "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.attn.o_proj.weight",
    "model.layers.{i}.self_attn.q_norm.weight": "layers.{i}.attn.q_norm.weight",
    "model.layers.{i}.self_attn.k_norm.weight": "layers.{i}.attn.k_norm.weight",
    "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate_proj.weight",
    "model.layers.{i}.mlp.up_proj.weight": "layers.{i}.mlp.up_proj.weight",
    "model.layers.{i}.mlp.down_proj.weight": "layers.{i}.mlp.down_proj.weight",
    "model.layers.{i}.input_layernorm.weight": "layers.{i}.input_layernorm.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm.weight",
    "model.norm.weight": "norm.weight",
}
```

M1 先做：

- 显式映射
- 打印 missing/unexpected key
- `tie_word_embeddings=True` 时跳过 `lm_head.weight`
- 不做 tensor parallel / quant / qkv packing / gate-up packing

#### 5. 双参考体系

| 任务段 | 正确性参考 | 系统/加载参考 |
| --- | --- | --- |
| T2-T6 模型 forward | transformers `modeling_qwen3.py` | vLLM 只看结构差异 |
| T7 权重加载 | HF `state_dict` key | **vLLM `load_weights()`** |
| T8 logits 对齐 | transformers full model logits | vLLM 不作为数值 reference |
| T9-T11 出字闭环 | transformers generation 行为 | nano-vllm / vLLM engine |

**本项目应用**：T7 `WeightMap + load_from_hf`。

### Reference code 分层纪律

| 参考项目 | 当前定位 | inferlite 使用阶段 | 读取边界 |
| --- | --- | --- | --- |
| `huggingface/transformers` | 数学真值 / L1 ground truth | T1-T8 | 只看 `modeling_qwen3.py` 对应类，测试以它为准 |
| `vllm-project/vllm` | 推理工程 / 权重加载 / engine 组织 | T4/T7/T9-T11/M2+ | 只看 `qwen3.py::Qwen3Attention/load_weights` 和 engine/scheduler 指定入口 |
| `QwenLM/Qwen3` | 官方使用说明 | 全阶段按需 | 查 tokenizer/chat template/thinking/deployment，不当算子源码 |
| `ggml-org/llama.cpp` | 本地推理 / GGUF / 量化 / Metal | M5+ | M1 不读，避免 C++/GGML graph 干扰 PyTorch 对齐 |
| `sgl-project/sglang` | serving runtime / prefix cache / speculative decoding | M5+ | 出字闭环后再看 runtime，不参与 T1-T8 数值对齐 |
| `huggingface/text-generation-inference` | 生产 serving/router/batching | M5+ | 看服务架构，不看 Qwen3 算子 |
| `NVIDIA/TensorRT-LLM` | GPU 高性能后端 / fused op | M8+ | 只在 kernel/benchmark 阶段看 |

一句话：**Transformers 定义“算得对不对”，vLLM 定义“推理系统怎么组织”，QwenLM 定义“官方怎么用”，其他项目延后到 serving/后端阶段。**

### transformers — 五个核心对象（HF 推理最小闭环）

| 对象 | 作用 | inferlite 是否手写 |
| --- | --- | --- |
| `AutoTokenizer.from_pretrained(id)` | 字符串 → token id（BPE） | 否，永远复用 |
| `AutoModelForCausalLM.from_pretrained(id, dtype, device_map)` | 下载权重 + 实例化 | **是**：`Qwen3Model.load_from_modelscope()` |
| `model.eval()` | 切推理模式 | 是，`__init__` 末尾自动调 |
| `tokenizer.decode(ids, skip_special_tokens=True)` | token id → 字符串 | 否 |
| `model.generate(...)` | 自动循环 forward | **手撕**：CLI 自己写循环+采样 |

`generate()` 的本质（M1 要重写）：
```python
while not done:
    out  = model(input_ids)
    next = sampler(out.logits[:, -1, :])
    input_ids = torch.cat([input_ids, next], dim=1)
```

**from_pretrained 关键参数**：
- repo id：`"Qwen/Qwen3-0.6B"`
- dtype：`torch.float32`（M1-M5 阶段固定 fp32）
- device_map：`"mps" / "cuda" / "cpu"`

**注意**：transformers 5.x 用 `dtype=`，4.x 是 `torch_dtype=`，二者等价。

### pytest — 单测核心 API

**4 个核心特性**（本项目几乎只用这些）：

#### 1. 函数即测试
```python
def test_xxx():
    assert ...
```
- 文件 `test_*.py`，函数 `test_*`
- 运行：`uv run pytest tests/unit/test_rmsnorm.py -v`

#### 2. `@pytest.mark.parametrize`
本项目最常用，N shape × M dtype 笛卡尔积：

```python
@pytest.mark.parametrize("shape", [(2, 8), (4, 16), (1, 32, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_vs_ref(shape, dtype):
    ...
```
- 自动展开 3×3=9 case
- 失败时显示出错参数组

#### 3. `pytest.fixture`
```python
@pytest.fixture
def qwen3_config():
    return Qwen3Config(hidden_size=1024, ...)
```
- scope：`function`（默认）/ `module` / `session`

#### 4. `pytest.mark.slow` + 选择性运行
配合 pyproject.toml 注册 marker；CI 跑 `-m "not slow"` 跳过。

**常用参数**：
- `-v` 详细 / `-x` 第一败停 / `-k <expr>` 名字过滤 / `--lf` 上次失败
- `--tb=short` 简短 traceback

**易错点**：
1. **不用 `torch.equal`**（fp16/bf16 必然不等）→ 用 `torch.allclose(..., atol=...)`
2. **atol 按 dtype 设**：fp32 1e-5，fp16/bf16 5e-3
3. **parametrize 装饰器堆叠**：靠近函数的最先变化（笛卡尔积外层→内层）

**外部参考**：https://docs.pytest.org/

### modelscope — snapshot_download

国内环境下载 Qwen3 等 HF repo：

```python
from modelscope import snapshot_download
local_dir = snapshot_download("Qwen/Qwen3-0.6B")
# 然后正常用 transformers.from_pretrained(local_dir)
```

- Qwen 系列在 ModelScope 的 repo id 与 HF 完全一致
- 权重 sha256 一致
- 第一次下完留 `.lock` 文件，进程 kill 后不释放 → preflight 应加 lock 清理逻辑

### huggingface_hub 1.x — 域名硬校验

**重要陷阱**：1.0 起客户端硬校验 URL 必须 `huggingface.co`。即使设 `HF_ENDPOINT=hf-mirror.com`，部分代码路径（auth / metadata）仍 hard-code 官方域名 → 国内不可用。

**解法**：国内一律走 `modelscope.snapshot_download`，不要依赖 HF_ENDPOINT。

---

## Concepts

### 数值升精度（upcast to fp32）

**一句话**：fp16/bf16 模型中，涉及 reduce + sqrt + softmax / exp / log 的算子，必须临时升精度到 fp32 计算，再降回原 dtype 输出。

**数学/直觉**：
- fp16 尾数 10 位，范围上限 65504：`x²` 若 |x|>256 即溢出
- bf16 尾数 7 位，范围与 fp32 同，但精度差，累加误差严重
- fp32 尾数 23 位，足够

reduce 会**累加**误差。hidden_size=1024 时 fp16 的 mean(x²) 几乎必然显著偏差。

**何时需要 upcast**：

| 场景 | 必要 | 备注 |
| --- | --- | --- |
| RMSNorm / LayerNorm | **必须** | mean(x²) + sqrt |
| Softmax | **必须** | exp 溢出 |
| log_softmax / NLLLoss | **必须** | log 数值敏感 |
| matmul（Linear） | 不必 | tensor core 已优化 |
| element-wise add/mul | 不必 | 不放大误差 |

**标准模式**：
```python
def numerically_sensitive_op(x):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    # ... fp32 运算 ...
    return result.to(input_dtype)   # 最后一步降回
```

**本项目应用**：T1 RMSNorm / T4 Attention softmax / T9 Sampler log_softmax

### 数值对齐策略 { #numeric-alignment-strategy }

**一句话**：数值对齐不是“跑通就行”，而是用一个可信 reference（通常是 transformers）把输入、权重、dtype、模式固定住，然后逐层比较输出误差，定位第一个分叉点。

#### 1. 三层验证

| 层级 | 目标 | 例子 |
| --- | --- | --- |
| L0 单元测试 | 单个算子逻辑正确 | `RMSNorm` / `ModelConfig` |
| L1 模块对齐 | 单个模型模块和 transformers 输出接近 | `Qwen3MLP` / `Qwen3Attention` |
| L2 e2e | 单序列 forward / greedy 结果一致 | logits top-k / next token |

M1·P1 的主线是 L0 → L1：先把 `ModelConfig` / `RMSNorm` / `SwiGLU` / `RoPE` / `GQA` / `DecoderLayer` 都钉住，再拼 `Qwen3Model`。

#### 2. 标准对齐流程

```python
ref = TransformersModule(...).eval()
mine = InferliteModule(...).eval()
mine.load_state_dict(ref.state_dict(), strict=False)

x = fixed_input(seed=0, dtype=torch.float32)
with torch.no_grad():
    y_ref = ref(x)
    y_mine = mine(x)

assert torch.allclose(y_mine, y_ref, atol=..., rtol=...)
```

关键控制变量：

1. **同输入**：固定 seed；shape 覆盖小/中/边界。
2. **同权重**：`load_state_dict()` 或手动复制权重。
3. **同模式**：`.eval()`；`torch.no_grad()` / `torch.inference_mode()`。
4. **同 dtype 策略**：fp32 先对齐；bf16/fp16 再单独测。
5. **同容差**：fp32 用更严 `1e-5` 级；bf16/fp16 放宽。

#### 3. T0 的角色

`ModelConfig` 是数值对齐的地基：后续所有模块的 shape 都从同一个 config 读，避免在 `RMSNorm(1024)`、`GQAAttention(16, 8, 128)` 等地方散落 magic number。

尤其是 Qwen3-0.6B：

```text
head_dim = 128
hidden_size / num_attention_heads = 1024 / 16 = 64
```

如果把 `head_dim` 错推成 64，T4 Attention 的 projection shape 会整体错位，L1 对齐不可能通过。

#### 4. 常见失败模式

| 失败模式 | 症状 | 排查方向 |
| --- | --- | --- |
| 忘 `.eval()` | dropout / cache 行为不同 | 先关训练态 |
| 忘同步权重 | 输出完全不同 | 检查 `state_dict` key |
| dtype 策略不同 | fp32 近、bf16 偏 | 查 upcast / cast 回原 dtype |
| shape 参数硬编码错 | matmul / reshape 爆炸或输出维度不对 | 回到 `ModelConfig` |
| reference 版本漂移 | 本地/链接看到的源码不一致 | 用固定 commit 链接 |

**本项目应用**：M1·P1 全部模块。

### tie_word_embeddings

**一句话**：embed 矩阵 `[V, H]` 既当 input embedding 也当 output projection (lm_head)，省一份权重。

**Qwen3-0.6B**：tied = True；>0.6B 是 False。所以不要硬编码。

**实现**：
```python
logits = F.linear(hidden, self.embed_tokens.weight)   # 复用 embed.weight 作为 lm_head
```

不需要单独 `self.lm_head` 层。

### 形状速查（背下来） { #shape-cheatsheet }

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `input_ids` | `[B, T]` | int64 token ID |
| `inputs_embeds` | `[B, T, H]` | 跳过 embed 直接喂浮点 |
| 隐藏态 | `[B, T, H]` | 层间 |
| logits | `[B, T, V]` | 词表分数 |
| 下一 token | `[B, 1]` | argmax |

**字母**：
- B = batch（M1=1，M3 引入 batching）
- T = seq length
- H = hidden_size（Qwen3-0.6B=1024）
- I = intermediate_size（=3072，仅 SwiGLU 中间）
- V = vocab_size（=151936）
- N = layers（=28）

### 推理上下文管理 { #inference-context }

```python
with torch.no_grad():
    out = model(...)
# 等价 (更彻底):
with torch.inference_mode():
    out = model(...)
```
推理路径必须包，否则 MPS 显存翻倍。

### Python dataclass

**一句话**：`dataclass` 是 Python 给“主要用来存数据的类”准备的语法糖；它根据字段声明自动生成 `__init__` / `__repr__` / `__eq__`，让 `ModelConfig` 这种超参容器少写样板代码。

#### 1. 不用 dataclass 会怎样

```python
class ModelConfig:
    def __init__(self, hidden_size: int, num_hidden_layers: int, rms_norm_eps: float):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
```

字段一多（T0 有 11 个）就容易漏赋值、顺序写错、比较不方便。

#### 2. 用 dataclass

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
```

Python 自动生成：

- `__init__`：可以 `ModelConfig(hidden_size=1024, ...)`
- `__repr__`：打印时显示字段和值，便于 debug
- `__eq__`：按字段内容比较，而不是按对象地址比较

所以测试里可以直接写：

```python
assert ModelConfig.from_json(path) == ModelConfig.qwen3_0_6b()
```

这句成立的前提就是 `dataclass` 自动生成了字段级 `__eq__`。

#### 3. `frozen=True`

```python
@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
```

表示对象创建后不可修改：

```python
cfg = ModelConfig(hidden_size=1024)
cfg.hidden_size = 2048   # dataclasses.FrozenInstanceError
```

**为什么 config 要 frozen**：模型一旦按 `cfg.hidden_size=1024` 初始化，权重 shape 已经固定；如果运行中偷偷改成 2048，config 与真实模型结构会分裂，后续会出现很难排查的 shape bug。

#### 4. 适用边界

`dataclass` 适合：字段驱动、主要存数据、行为少的对象。

典型例子：

- `ModelConfig`
- `SamplingParams`
- `RequestState`
- `BlockMeta`

不适合：大量行为 + 复杂生命周期的核心执行对象，例如 `EngineCore` / `Scheduler` / `Qwen3Model(nn.Module)`。

**本项目应用**：T0 `inferlite.config.ModelConfig`。

### Factory pattern (经典工厂)

**一句话**：Factory pattern 是“把对象创建逻辑封装成一个方法/函数”，调用方不用知道对象怎么组装，只拿到成品。

T0 里有两个轻量工厂：

```python
cfg = ModelConfig.qwen3_0_6b()      # 从硬编码事实创建
cfg = ModelConfig.from_json(path)   # 从 config.json 创建
```

#### 为什么不用调用方自己拼

如果每个调用方都这么写：

```python
cfg = ModelConfig(
    hidden_size=1024,
    num_hidden_layers=28,
    # ... 11 个字段 ...
)
```

问题：

1. magic number 到处散落
2. 字段一变要全局改
3. JSON 过滤 / dtype cast / head_dim fallback 逻辑会复制多份

Factory 把这三件事收到一个地方：

- `qwen3_0_6b()`：测试用，不依赖磁盘
- `from_json()`：真实模型加载用，负责白名单过滤、`head_dim` 兜底、`rope_theta` cast float

#### 本质

Factory 不是高级设计模式，T0 里就是：**给“怎么创建一个合法 config”起一个名字**。

**本项目应用**：

- T0：`ModelConfig.qwen3_0_6b()` / `ModelConfig.from_json()`
- T7：`load_from_hf(path)` 会先调用 `ModelConfig.from_json(path / "config.json")`


---

## Tools

### uv（Python 包管理器）

Astral 出品，Rust 写。取代 pyenv + venv + pip + pip-tools。

- 装 PyTorch + transformers：pip 60s，uv 5s
- 一站式：Python 版本 + venv + 依赖锁定
- lockfile 默认开（`uv.lock` commit 进 git）
- 原生支持 PyTorch 索引源切换（cpu / cu121 / cu124）

**常用命令**：
```bash
uv sync                    # 装依赖
uv sync --frozen           # 严格按 lock
uv run pytest              # 在 venv 跑命令
uv add "torch>=2.5"        # 加依赖
uv add --dev pytest-xdist  # dev 依赖
uv lock --upgrade <pkg>    # 升级单包
```

**`uv.lock` 重要性**：
- `pyproject.toml` 说"我要 torch>=2.4"
- `uv.lock` 锁定"实际装的是 torch==2.7.0 + 全部传递依赖 + sha256"
- 必须 commit（保证跨机器/时间复现）

### make（任务运行器）

Unix 自带（1976）。`Makefile` 定义"任务名 → shell 命令"。

```makefile
setup:
\tbash scripts/setup.sh
test:
\tuv run pytest
```

**关键**：执行行**必须 Tab 缩进**，不能空格。

### ruff（lint + format）

取代 flake8 + black + isort。`make lint` / `make fmt`。配置在 `pyproject.toml [tool.ruff]`。

### pre-commit（commit 前自动检查）

`.pre-commit-config.yaml` 配置，含 trailing-whitespace / EOF / yaml/toml check / large-file guard / ruff lint+format。`scripts/setup.sh` 自动注册 hook。

### pytest-mark / CI matrix

CI 跑 `-m "not slow"` 跳过慢测试。matrix: ubuntu + macos, python 3.12。

---

## 维护规则

- **新增卡片**：在对应章节末尾追加 `### <Title>` 子段，不开新文件
- **删除卡片**：直接删段，更新引用
- **跨段引用**：用 markdown 锚点 `[upcast](#数值升精度upcast-to-fp32)`
- **完整精读论文**：留链接到 `docs/papers/...`（走 `paper-deep-read` skill），本文件只放项目视角摘要
