# 概念速查（M1 期间常翻）

> 配合 `scripts/preflight.py` 看，preflight 的 5 行代码就是 "transformers 推理最小闭环"。
> 本文件不含任务拆解，纯查阅型。任务拆解见 `docs/M1.md`。

## 1. 五个核心对象

| 对象 | 作用 | inferlite 是否手写 |
| --- | --- | --- |
| `AutoTokenizer.from_pretrained(id)` | 字符串 → token id 列表（BPE 分词） | 否，永远复用 transformers |
| `AutoModelForCausalLM.from_pretrained(id, dtype, device_map)` | 下载权重 + 实例化网络 | **是**，叫 `Qwen3Model.load_from_modelscope(...)` |
| `model.eval()` | 切推理模式（关 dropout） | 是，`__init__` 末尾自动调 |
| `tokenizer.decode(ids, skip_special_tokens=True)` | token id → 字符串 | 否 |
| `model.generate(...)` | 自动循环 forward 直到 max_new_tokens 或 EOS | **要手撕**：CLI 自己写循环 + 采样 |

`generate(...)` 的本质（M1 你 CLI 要重写的）：

```python
while not done:
    out  = model(input_ids)                 # forward 一次
    next = sampler(out.logits[:, -1, :])    # 取最后位置 logits → 采样
    input_ids = torch.cat([input_ids, next], dim=1)
```

## 2. 关键超参

**generate 超参**：

| 超参 | 含义 |
| --- | --- |
| `max_new_tokens` | 新生成 token 上限（不含 prompt） |
| `do_sample=False` | 贪心解码（M1 唯一支持） |
| `do_sample=True` + `temperature/top_p/top_k` | 随机采样（M5） |
| `num_beams` | beam search（inferlite 不做） |
| `pad_token_id` | padding 用，单 prompt 单 batch 不需要 |

**from_pretrained 超参**：

| 超参 | 含义 | M1 用值 |
| --- | --- | --- |
| `<owner>/<name>` | HF Hub repo id | `"Qwen/Qwen3-0.6B"` |
| `dtype` | 权重精度 | `torch.float32` |
| `device_map` | 权重搬到哪 | `"mps"` / `"cuda"` / `"cpu"` |

> transformers 5.x 用 `dtype=`；4.x 老教程是 `torch_dtype=`，二者等价。

**张量转换两件事**：

```python
inputs = tokenizer("你好", return_tensors="pt").to(device)
#                                  ^^^                ^^^
#                          返回 PyTorch tensor   搬到模型所在设备
```

## 3. 推理优化常用上下文

```python
with torch.no_grad():       # 不算梯度，省 ~50% 显存 + 加速 5–30%
    out = model(...)

# 等价（更彻底）：
with torch.inference_mode():
    out = model(...)
```

M1 推理路径必须包一层，否则 MPS 上显存翻倍。

## 4. 形状速查（背下来）

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `input_ids` | `[B, T]` | int64 token ID |
| `inputs_embeds` (M13) | `[B, T, H]` | 跳过 embed 直接喂浮点 |
| 隐藏态 | `[B, T, H]` | 每层 attention/MLP 之间 |
| logits | `[B, T, V]` | 每个位置在词表上的"分数" |
| 下一 token | `[B, 1]` | argmax 出来的 ID |

**字母含义**：

- **B** = batch size（M1 = 1；M3 引入连续 batching）
- **T** = sequence length（token 数）
- **H** = hidden_size（Qwen3-0.6B = 1024）
- **I** = intermediate_size（Qwen3-0.6B = 3072，仅 SwiGLU 中间维度）
- **V** = vocab_size（Qwen3 = 151,936）
- **N** = layers（Qwen3-0.6B = 28）

## 5. Qwen3-0.6B 超参速查

见 `docs/M1.md` §2.2。

## 6. 一句话贯穿前向

```
[B, T] int → embedding → [B, T, H] →
  ┌────────────── 28 × Transformer Block ──────────────┐
  │  RMSNorm → Attention(GQA + RoPE + QK-norm) + 残差  │
  │  RMSNorm → SwiGLU MLP                     + 残差  │
  └────────────────────────────────────────────────────┘
→ Final RMSNorm → lm_head(= embed_tokens.weight) → [B, T, V]
```
