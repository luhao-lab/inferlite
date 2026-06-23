# Module Blueprints

> **这是什么**：每个核心模块的"契约卡片"，解决"越写越复杂、后面看不懂"的问题。
>
> **和 knowledge.md / lessons.md 的区别**：
> - `knowledge.md` — 概念/论文/工具，事实性，独立读
> - `lessons.md` — 踩坑记录，叙事性，按时间追加
> - `blueprints.md`（本文件）— **模块维度**，回答"这个模块做什么 / 契约是什么 / 踩过什么坑 / 和谁有依赖"
>
> **灵感来源**：MetaInfer 的 `inference_blueprint.json`，用 Markdown 实现，更轻量，对人可读。
>
> **维护规则**：
> - 每个模块一个 H2 卡片，格式固定（见下方模板）
> - 每次任务卡归档时（`/archive task T<n>`）更新对应 blueprint
> - M 里程碑归档时更新"跨 M 依赖"字段
> - 不要删旧内容，只追加 `[M2 更新]` 等标注

---

## 📊 索引

| 模块 | 文件 | 完成于 | 跨 M 稳定性 |
|------|------|--------|------------|
| [ModelConfig](#modelconfig) | `inferlite/config.py` | M1-T0 | M1-M4 稳定（M2 加 KV cache 字段） |
| [RMSNorm](#rmsnorm) | `inferlite/model/layers.py` | M1-T1 | 全程稳定，不需修改 |
| [SwiGLUMLP](#swiglumlp) | `inferlite/model/layers.py` | M1-T2 | M1 稳定，M2 可能加 TP 切分 |
| [RotaryEmbedding](#rotaryembedding) | `inferlite/model/layers.py` | M1-T3 | M1 稳定，M2 可能改为预计算 |
| [GQAAttention](#gqaattention) | `inferlite/model/attention.py` | M1-T4 | M2 重大改动（加 KV Cache 槽） |
| [Qwen3DecoderLayer](#qwen3decoderlayer) | `inferlite/model/qwen3.py` | M1-T5 | M2 透传 KV cache 参数 |
| [Qwen3ForCausalLM](#qwen3forcausallm) | `inferlite/model/qwen3.py` | M1-T5 | M2 透传 KV cache 参数 |
| [WeightLoader](#weightloader) | `inferlite/model/weights.py` | M1-T6 | M1 稳定，M2 需加量化分支 |
| [GreedySampler](#greedysampler) | `inferlite/sampler/greedy.py` | M1-T7 | M1 稳定，M4 替换为 Top-p/k |
| [EngineCore](#enginecore) | `inferlite/engine/core.py` | M1-T8 | M2 大改（连续批处理） |

---

## 模板

```markdown
## ModuleName

**职责**：一句话说清楚这个模块做什么。

**接口契约**：
- 输入：shape / dtype / 含义
- 输出：shape / dtype / 含义
- 前置条件：调用方必须保证的事情

**在推理链路中的位置**：
调用关系图（谁调用我，我调用谁）

**关键设计决策**：
- 为什么这样设计，不那样设计（ADR 级别的事情放 decisions.md，这里放模块内的选择）

**踩坑记录**：
- L<n>: 简要描述（详见 lessons.md）

**跨 M 依赖**：
- 哪些 M 会改动这个模块，改什么

**数值对齐 ground truth**：
对应 transformers 里的类/函数（如有）
```

---

## ModelConfig

**职责**：从 `config.json` 反序列化模型超参，作为整个推理链路的唯一配置数据源。

**接口契约**：
- 输入：`config.json` 路径（字符串）或已解析的 dict
- 输出：`ModelConfig` dataclass 实例，所有字段都有明确类型
- 前置条件：`config.json` 必须包含 Qwen3 必填字段（`hidden_size`, `num_hidden_layers`, etc.）

**在推理链路中的位置**：
```
load_causal_lm_from_hf()
  └─ ModelConfig.from_json(model_dir / "config.json")
       └─ 传入 Qwen3ForCausalLM(config)
            └─ 传入所有子模块（RMSNorm eps, Attention head_dim, ...）
```

**关键设计决策**：
- `head_dim` 从 config.json 优先读取，兜底用 `hidden_size // num_attention_heads`（GQA 中 KV 减少的是 group 数，不是 head 维度，见 L4）
- 纯 dataclass，无方法副作用，所有字段 post_init 校验
- 不持有模型权重或设备引用（只存超参）

**踩坑记录**：
- L4: `head_dim` 不等于 `hidden_size // num_attention_heads`，必须优先读 JSON 字段（详见 lessons.md#L4）

**跨 M 依赖**：
- M2 加 KV cache 相关字段（`max_seq_len`, `block_size`）
- M1-M4 不需修改已有字段

**数值对齐 ground truth**：
`transformers.models.qwen3.configuration_qwen3.Qwen3Config`

---

## RMSNorm

**职责**：均方根归一化，Pre-norm 模式，每个 DecoderLayer 入口和 Attention→MLP 之间调用。

**接口契约**：
- 输入：`[B, T, H]`，任意 dtype（fp16 / bf16 / fp32）
- 输出：`[B, T, H]`，dtype 与输入相同
- 参数：`weight [H]`（可学习 scale），`eps`（Qwen3 = 1e-6）
- 前置条件：无（任意 float tensor 均可）

**在推理链路中的位置**：
```
DecoderLayer.forward(x)
  ├─ residual = x
  ├─ x = input_layernorm(x)           # RMSNorm ← 这里
  ├─ x = self_attn(x)
  ├─ x = residual + x
  ├─ residual = x
  ├─ x = post_attention_layernorm(x)  # RMSNorm ← 这里
  └─ x = mlp(x)
  └─ return residual + x
```

**关键设计决策**：
- 内部 upcast fp32 算 var，最后 cast 回原 dtype（精度关键，见 L1）
- `weight * x` 在 fp32 下完成，然后一次性 cast，不能"先 cast x 再乘 weight"
- 不含 bias（RMSNorm 定义如此，区别于 LayerNorm）

**踩坑记录**：
- L1: fp16/bf16 下不 upcast 则 x² 累加溢出或精度损失严重，28 层累计后 logits 偏差超容差（详见 lessons.md#L1）

**跨 M 依赖**：
- 全程稳定，M2~M4 不需修改
- M2 的 KV Cache 机制不触碰 Norm 层

**数值对齐 ground truth**：
`transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm`

---

## SwiGLUMLP

**职责**：Transformer MLP 层，用 SwiGLU 激活函数（gate 控制）替代传统 FFN。

**接口契约**：
- 输入：`[B, T, H]`（H = hidden_size）
- 输出：`[B, T, H]`（shape 不变）
- 参数：`gate_proj [I, H]`、`up_proj [I, H]`、`down_proj [H, I]`（I = intermediate_size）
- 前置条件：无

**在推理链路中的位置**：
```
DecoderLayer
  └─ mlp(post_attention_layernorm(x))
       ├─ gate = gate_proj(x)      # [B,T,I]
       ├─ up   = up_proj(x)        # [B,T,I]
       ├─ gate = silu(gate) * up   # SwiGLU gate
       └─ return down_proj(gate)   # [B,T,H]
```

**关键设计决策**：
- Qwen3 用 SiLU（`F.silu`）作为激活，不是 GELU
- `intermediate_size` 独立超参（Qwen3-0.6B = 2816），不是 `4 * hidden_size`
- gate 和 up 是两条独立路径，合并成 element-wise 乘法

**踩坑记录**：
- 无（截至 M1）

**跨 M 依赖**：
- M1 稳定
- M2 TP 并行时可能需要列并行切分 gate/up/down

**数值对齐 ground truth**：
`transformers.models.qwen3.modeling_qwen3.Qwen3MLP`

---

## RotaryEmbedding

**职责**：给 Q、K 注入位置信息，旋转编码（RoPE），替代绝对位置编码。

**接口契约**：
- 输入：
  - `q [B, T, n_q,  d]`（n_q = num_attention_heads）
  - `k [B, T, n_kv, d]`（n_kv = num_key_value_heads）
  - `position_ids [B, T]` 或隐含 `[0..T-1]`
- 输出：`q_rot [B, T, n_q, d]`、`k_rot [B, T, n_kv, d]`（shape 不变）
- 前置条件：`d` 必须是 head_dim（RoPE 只旋转前 `rotary_dim` 个维度，Qwen3 全旋转）

**在推理链路中的位置**：
```
GQAAttention.forward()
  ├─ q = q_norm(q)  # QK-Norm（Qwen3 特有）
  ├─ k = k_norm(k)
  ├─ q, k = rotary_emb(q, k, position_ids)  # RoPE ← 这里（Norm 之后）
  └─ ... attention 计算
```

**关键设计决策**：
- Qwen3 的 RoPE 在 QK-Norm **之后**施加（区别于部分旧模型）
- Neox-style 旋转（前半维度 cos 旋转，后半维度对应）
- `base`（theta）= 1,000,000（Qwen3 扩展上下文）
- M2 引入 KV Cache 后，position_ids 需传入 decode 位置（不再是 `0..T-1`）

**踩坑记录**：
- 无（截至 M1）

**跨 M 依赖**：
- M2 decode 阶段 position_ids 从 `[0,T-1]` 改为 decode 步对应的绝对位置
- M3/M4 引入 long-context 时 base 可能动态调整（YaRN/ABF）

**数值对齐 ground truth**：
`transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding`

---

## GQAAttention

**职责**：分组查询注意力（GQA），Q 多头、KV 少头，用 repeat 扩展后计算 scaled dot-product attention。

**接口契约**：
- 输入：`hidden_states [B, T, H]`，`attention_mask [B, 1, T, T]`（可选）
- 输出：`[B, T, H]`
- 前置条件：`num_attention_heads % num_key_value_heads == 0`

**在推理链路中的位置**：
```
DecoderLayer
  └─ self_attn(input_layernorm(x))
       ├─ q_proj → [B,T,n_q,d]
       ├─ k_proj → [B,T,n_kv,d]
       ├─ v_proj → [B,T,n_kv,d]
       ├─ q_norm, k_norm (QK-Norm)
       ├─ rotary_emb (RoPE)
       ├─ k = repeat_kv(k, n_q//n_kv)  # GQA expand
       ├─ v = repeat_kv(v, n_q//n_kv)
       ├─ attn = scaled_dot_product_attention(q, k, v, mask)
       └─ o_proj → [B,T,H]
```

**关键设计决策**：
- Qwen3 有 QK-Norm（每个 head 内部 RMSNorm），Norm 在 RoPE **之前**施加
- GQA groups = `num_attention_heads // num_key_value_heads`（Qwen3-0.6B = 16/8 = 2）
- M1 使用 `torch.nn.functional.scaled_dot_product_attention`（标准实现）
- M2 引入 KV Cache 后需重构：decode 步只输入 1 个 token，但 attend 到整个历史

**踩坑记录**：
- 无（截至 M1；M2 KV Cache 改造时此处高风险）

**跨 M 依赖**：
- **M2 重大改动**：增加 KV cache 槽参数，decode 步复用历史 KV
- M1 的 prefill 路径在 M2 保持不变，decode 路径新增

**数值对齐 ground truth**：
`transformers.models.qwen3.modeling_qwen3.Qwen3Attention`

---

## Qwen3DecoderLayer

**职责**：单层 Transformer decoder block，组合 Norm + Attention + Norm + MLP + 残差连接。

**接口契约**：
- 输入：`hidden_states [B, T, H]`，`attention_mask`（可选）
- 输出：`[B, T, H]`（shape 不变）
- 参数：来自 `ModelConfig`（共享 eps、head 数等）

**在推理链路中的位置**：
```
Qwen3ForCausalLM
  └─ model.layers[0..27]（28 层 DecoderLayer）
       每层：
       ├─ residual = x
       ├─ x = input_layernorm(x)       # RMSNorm
       ├─ x = self_attn(x, mask)       # GQAAttention
       ├─ x = residual + x             # 残差 1
       ├─ residual = x
       ├─ x = post_attention_layernorm(x)  # RMSNorm
       ├─ x = mlp(x)                   # SwiGLUMLP
       └─ return residual + x          # 残差 2
```

**关键设计决策**：
- Pre-norm 模式（Norm 在子层之前，区别于 Post-norm）
- 残差连接在每个子层之后（不在 Norm 之后）
- 层数 = `num_hidden_layers`（Qwen3-0.6B = 28）

**踩坑记录**：
- 无（截至 M1）

**跨 M 依赖**：
- M2 透传 KV cache 参数到 GQAAttention（接口变更但逻辑不变）

**数值对齐 ground truth**：
`transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer`

---

## Qwen3ForCausalLM

**职责**：完整推理模型，embed → 28 层 decoder → final norm → lm_head → logits。

**接口契约**：
- 输入：`input_ids [B, T]`（token id），`logits_to_keep=1`（只取最后 N 步的 logits）
- 输出：`logits [B, logits_to_keep, V]`（V = vocab_size = 151936）
- 前置条件：`input_ids` 中值在 `[0, vocab_size)` 范围内

**在推理链路中的位置**：
```
EngineCore.step(input_ids)
  └─ model(input_ids, logits_to_keep=1)
       ├─ embed_tokens → [B,T,H]
       ├─ × 28 DecoderLayer
       ├─ norm (final RMSNorm)
       └─ lm_head → [B,1,V]（只取最后 1 步）
```

**关键设计决策**：
- `tie_word_embeddings=False`（Qwen3-0.6B），lm_head 和 embed_tokens 权重不共享
- `logits_to_keep` 参数避免计算所有 T 步的 logits（节省内存）
- M1 只支持 prefill（输入 T 个 token），M2 加 decode（输入 1 个 token）

**踩坑记录**：
- 无（截至 M1）

**跨 M 依赖**：
- M2 forward 接口增加 KV cache 相关参数（past_key_values 或 block_table）

**数值对齐 ground truth**：
`transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM`

---

## WeightLoader

**职责**：从 HuggingFace safetensors 文件加载权重，映射到 inferlite 模型的参数名。

**接口契约**：
- 输入：`model_dir`（含 `config.json` + `*.safetensors` 的目录）
- 输出：已填充权重的 `Qwen3ForCausalLM` 实例（eval 模式，不在计算图中）
- 前置条件：safetensors 文件格式，HF 标准参数名前缀

**在推理链路中的位置**：
```
main()
  └─ load_causal_lm_from_hf(model_dir)
       ├─ ModelConfig.from_json()
       ├─ Qwen3ForCausalLM(config)
       └─ load_weights_into_model(model, model_dir)
            ├─ safetensors.torch.load_file()
            └─ 名称映射 + model.load_state_dict(strict=False)
```

**关键设计决策**：
- HF 参数名前缀 `model.` → inferlite 直接对应（或需 strip）
- `tie_word_embeddings` 为 False 时 lm_head 有独立权重，需单独处理
- 量化支持留 M2（M1 只支持 bf16/fp16）

**踩坑记录**：
- `tie_word_embeddings` 缺失时 lm_head 会报 missing key（见 knowledge.md#tie_word_embeddings）

**跨 M 依赖**：
- M2 加量化时需修改权重 dtype 转换逻辑
- M1 稳定

**数值对齐 ground truth**：
HF `from_pretrained` 内部的 `load_state_dict` 逻辑

---

## GreedySampler

**职责**：贪婪采样，从 logits 取 argmax 得到下一个 token id。

**接口契约**：
- 输入：`logits [B, 1, V]`（lm_head 输出）
- 输出：`next_token_ids [B, 1]`（token id）
- 前置条件：logits 是 float tensor（不需要归一化，只取最大值）

**在推理链路中的位置**：
```
EngineCore.step()
  ├─ logits = model(input_ids, logits_to_keep=1)
  └─ next_tokens = sampler(logits)  # GreedySampler ← 这里
```

**关键设计决策**：
- 实现 `SamplerProtocol`（`engine/protocol.py`），使 EngineCore 不依赖具体采样策略
- M1 只有贪婪，M4 替换为 Top-p / Top-k 时只需换实现，EngineCore 不变

**踩坑记录**：
- 无（截至 M1）

**跨 M 依赖**：
- M4 替换为 TopPSampler / TopKSampler，本模块退役
- M1~M3 稳定

**数值对齐 ground truth**：
`torch.argmax`（无对应 transformers 类）

---

## EngineCore

**职责**：推理引擎主循环，管理自回归 decode 迭代（step by step），连接模型和采样器。

**接口契约**：
- 输入（`generate`）：`input_ids [B, T]`（prefill），`max_new_tokens`，`eos_token_id`
- 输出：`output_ids [B, T+N]`（输入 + 新生成的 token）
- 前置条件：`model` 实现 `forward(input_ids, logits_to_keep)`，`sampler` 实现 `SamplerProtocol`

**在推理链路中的位置**：
```
main()
  └─ engine.generate(input_ids, max_new_tokens)
       └─ for step in range(max_new_tokens):
             ├─ next_token = engine.step(current_ids)
             │    ├─ logits = model(current_ids, logits_to_keep=1)
             │    └─ return sampler(logits)
             ├─ current_ids = append(current_ids, next_token)
             └─ if next_token == eos: break
```

**关键设计决策**：
- M1 简单追加 token（无 KV cache，每步重新走全序列）
- 通过 `ModelProtocol` 和 `SamplerProtocol` 解耦，不直接持有 `Qwen3ForCausalLM`
- **M2 重大重构**：引入 KV Cache 后不再重新走全序列，需要管理 block_table、kv_pool

**踩坑记录**：
- 无（截至 M1；M2 重构此处是最高风险点）

**跨 M 依赖**：
- **M2 大改**：引入连续批处理 + KV Cache 管理（block_table、kv_pool、调度器）
- M1 的 generate 逻辑会被 M2 完全替换

**数值对齐 ground truth**：
无（调度逻辑，无对应 transformers 类）

---

## 维护规则

1. **新模块**：任务卡归档时 `/archive task T<n>` → 在本文件追加 blueprint
2. **更新字段**：里程碑归档 `/archive milestone M<n>` → 更新"跨 M 依赖"字段，标注 `[M<n> 更新]`
3. **不删旧内容**：标注 `[已变更 M<n>]` 后保留，作为演化历史
4. **格式一致**：所有卡片使用相同 6 个字段（职责/接口契约/推理链路位置/设计决策/踩坑记录/跨 M 依赖）
