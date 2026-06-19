# 从零手写 LLM 推理引擎（一）：朴素前向推理

本文是"从零手写 LLM 推理引擎"系列的第一篇，对应里程碑 M1（`m1/naive-forward`）。文章分两条主线：
- **ML 主线**：推理引擎的分层架构、Qwen3 模型的结构实现（GQA、QK-Norm、RoPE 等）
- **工程主线**：让项目"可用"所必须的工程技术——Python 类型系统、测试策略、打包、开发工作流

两条主线的关系：ML 主线回答"推理引擎的逻辑是什么"，工程主线回答"怎么把这个逻辑写成一个可维护、可测试、可发布的 Python 项目"。

---

## 项目概览

| | |
|---|---|
| **项目地址** | [inferlite](https://github.com/luhao-lab/inferlite) · Tag `m1/naive-forward` |
| **模型** | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)（HuggingFace） |
| **代码规模** | ~800 行核心实现 + 90 个单元测试，mypy / ruff 全通过 |

```bash
git clone https://github.com/luhao-lab/inferlite.git
cd inferlite && uv sync

# 运行全部单元测试
uv run pytest tests/unit/ -q

# 用真实 Qwen3-0.6B 生成
uv run inferlite-generate \
  --model-dir ~/huggingface/Qwen3-0.6B \
  --prompt "请解释 Transformer 中 Attention 的作用" \
  --max-new-tokens 100 \
  --chat-template
```

Qwen3-0.6B 模型下载：[HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B) / [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)

---

## 为什么要重复造轮子

> 已有 vLLM、TGI、SGLang、nano-vllm……为什么还要再写一个推理框架？

### 0.1 开源框架解决的是部署问题，不是认知问题

vLLM、TGI 这类工业级框架的设计目标是**高吞吐生产部署**：PagedAttention、连续批处理、多 GPU 张量并行，代码量数万行，抽象层次极高。读懂它们需要先掌握你想学的那些基础知识；而调用它们，对个人的认知提升几乎没有帮助——你只是多学会了一个 API。

nano-vllm、picoGPT、llama2.c 这类"nano 系列"项目在社区里长期有人在做，原因也在这里：**没有自己实现过的项目，都是没有掌握的知识。** 用一个框架 run 起来一个模型，和真正理解这个模型的每一层计算是两件完全不同的事。

### 0.2 AI 时代的新问题：LLM 强到代码可以一键生成

这个时代出现了一个新的认知陷阱：LLM 已经强到可以一键生成任何模块的代码，甚至包括 GQA attention、RoPE、KV Cache 实现。这带来了一个幻觉——"我让 AI 写出来了，说明我掌握了"。

但认知的建立需要**主动构建**，不是被动接收。让 AI 生成一段 RoPE 代码，和自己动手推导旋转矩阵的形状变换、亲手调试 `cos/sin` 的 broadcast 维度、跑通数值对齐测试，是完全不同的两条路径。前者快，后者留下认知。

inferlite 不是纯手敲的项目——核心逻辑由人主导，AI 负责样板代码、测试骨架、文档草稿。这是一种**人机协作的学习方式**：把 AI 当搭档而不是外包方，自己始终握着核心逻辑的主导权。

### 0.3 个人 LLM 学习框架的定位

inferlite 的定位是**个人 LLM 学习框架**，而不是生产推理引擎。

这个定位意味着：

- **新模型出来，手动实现加入**。比如 Qwen3 加入了 QK-Norm，LLaMA 4 引入了 Mixture-of-Experts，下一代模型可能用 SSM 替代 Attention 的一部分——每次有新结构，把它实现到自己的框架里、跑通测试、验证数值对齐，才算真正掌握了这个结构。
- **每个里程碑解决一个核心问题**。M1 是朴素前向推理（正确性基线），M2 是 KV Cache（效率），M3 是并发调度……每个 milestone 都是一个清晰的认知台阶。
- **框架的所有权在自己手上**。不是 fork 别人的项目加几行代码，而是从第一行开始，每个设计决策都有自己的理解在里面。

在 AI 能力爆炸的时代，工具的使用门槛越来越低，但**对底层原理的理解**反而变得更有价值——因为理解原理的人才能判断 AI 输出的对错，才能在出问题时知道往哪里排查。这正是 nano 系列项目在 2024-2025 年反而越来越多的原因。

---

## 推理引擎做什么，主流框架怎么组织

> **本章要回答的问题**：在动手写任何代码之前，LLM 推理引擎究竟要解决什么问题？工业级方案是如何分层的？M1 在这张大图里处于什么位置？

### 1.1 推理引擎的核心职责

LLM 推理的逻辑可以拆成五个关键问题：

| 问题 | 对应组件 |
|------|---------|
| 用户文本如何变成 token？ | Tokenizer + Chat Template |
| token 如何经过神经网络得到下一个 token 的概率分布？ | Model Forward Pass |
| 如何从概率分布里选 token？ | Sampler（greedy / top-p / top-k） |
| 如何避免重复计算已有 token 的 Attention？ | KV Cache |
| 如何调度多个并发请求，高效利用 GPU？ | Scheduler + Batch Manager |

### 1.2 vLLM 的分层架构

以 vLLM（当前工业界最主流的开源 LLM 推理引擎）为参照：

```
用户 / HTTP 请求
    |
    v
[Entrypoint]  LLM class / OpenAI-compatible API Server
    |
    v
[LLMEngine]   Input Processing · Scheduler · Output Processing
    |
    v
[Worker]      一个 GPU → 一个 Worker 进程
    |
    v
[ModelRunner] 准备输入 tensor，管理 KV cache，调用 forward
    |
    v
[Model]       torch.nn.Module：真正的神经网络计算
```

这五层中，**越靠上越关注调度和并发，越靠下越关注数值计算**。vLLM 的调度器（Scheduler）负责 PagedAttention 的 KV Cache 内存管理；ModelRunner 负责把 token ids 组装成 batch tensor 并调用 forward；Model 层只是一个 `nn.Module`，不知道自己在被谁调用。

### 1.3 M1 在这张图里的位置

inferlite M1 实现了整个链路的**最小可运行子集**：

```
                  vLLM 完整架构          inferlite M1
                  ─────────────         ──────────────
Entrypoint        LLM / API Server  →   CLI (cli.py)
LLMEngine         Scheduler+调度    →   generate loop (engine/core.py)
Worker            多进程 GPU Worker →   单进程 CPU
ModelRunner       KV cache 管理     →   无 KV cache，每步 full forward
Model             nn.Module         →   Qwen3ForCausalLM (model/)
Sampler           top-p/top-k/beam  →   greedy argmax (sampler/)
Tokenizer         自动 chat template →   可选 --chat-template
```

**M1 有意省略的部分**：KV Cache、批调度、并行、量化。这些都是正确性之上的效率优化，M1 只保证**数值正确**，后续里程碑逐步叠加。

---

## 模块设计：目录职责与依赖方向

> **本章要回答的问题**：为什么这样划分目录？每个 package 的职责边界在哪里？依赖方向为什么要严格控制？

### 2.1 目录结构与职责

```
inferlite/              # 主包（Python package）
│
├── config.py           # 模型超参配置：从 HF config.json 反序列化，贯穿所有层
│
├── engine/             # 调度层：知道"怎么跑"，不知道"模型长什么样"
│   ├── protocol.py     #   LLMModel Protocol：engine 对模型的最小接口约定
│   └── core.py         #   EngineCore（step 方法）+ generate loop
│
├── model/              # 数值计算层：Qwen3 的神经网络实现
│   ├── layers.py       #   叶节点：RMSNorm / SwiGLUMLP / RotaryEmbedding
│   ├── attention.py    #   GQAAttention（含 QK-Norm、RoPE、causal mask）
│   ├── qwen3.py        #   组合节点：DecoderLayer → Qwen3Model → Qwen3ForCausalLM
│   └── weights.py      #   safetensors 加载 + HF key 映射
│
├── sampler/            # 采样层：从 logits 选 next token 的策略
│   └── greedy.py       #   GreedySampler（argmax）
│
├── server/             # HTTP API 层（M2+ 实现，当前预留占位）
│   └── __init__.py     #   FastAPI / uvicorn 入口将在此实现
│
├── utils/              # 通用工具函数（当前预留占位）
│   └── __init__.py     #   未来放 logging、metric、helper 等跨层工具
│
└── cli.py              # 命令行入口：把用户参数组装成一次完整的推理调用
```

各目录职责一句话总结：

| 目录 / 文件 | 职责 | 知道什么 | 不知道什么 |
|------------|------|---------|-----------|
| `config.py` | 超参的唯一事实源 | 11 个 Qwen3 超参 | 任何 tensor 操作 |
| `engine/` | 推理调度，模型无关 | generate 循环、Protocol 接口 | Qwen3 的结构细节 |
| `model/` | Qwen3 神经网络计算 | tensor shape、注意力计算 | 它被谁调用 |
| `sampler/` | token 采样策略 | logits [B,V] → next_token [B,1] | 模型结构、引擎调度 |
| `server/` | HTTP API 入口（预留） | 请求解析、响应格式 | 模型实现细节 |
| `utils/` | 跨层通用工具（预留） | 调用方按需使用 | — |
| `cli.py` | 组装所有组件，暴露命令行 | 各模块的公开 API | 模块内部实现 |

### 2.2 依赖方向

```
cli.py
  └─> engine/core.py    (调度层：不知道模型结构)
        └─> engine/protocol.py  (LLMModel Protocol)
        └─> sampler/greedy.py   (采样层：只看 logits [B,V])
  └─> model/weights.py  (加载层：不知道引擎)
        └─> model/qwen3.py
              └─> model/attention.py
              └─> model/layers.py
  └─> config.py         (被所有层引用，但自身不依赖任何层)
```

关键设计原则：**engine 层只依赖 Protocol，不依赖具体模型类**。`EngineCore` 接受任何满足 `LLMModel` Protocol（即 `(input_ids) -> logits`）的对象，便于后续替换模型实现（如加 KV Cache 版本）而不改动引擎逻辑。`config.py` 处于依赖树底端，被所有层读取，但自身不导入任何项目模块。

---

## 代码导读：一次推理请求的完整旅程

> 推荐的阅读方式：先跑起来（见"项目概览"的快速上手命令），再对照这张调用图读代码。

### 入口：从命令行到 `main()`

```bash
uv run inferlite-generate --model-dir ~/huggingface/Qwen3-0.6B --prompt "你好" --chat-template
```

`pyproject.toml` 的 `[project.scripts]` 把这条命令映射到：

```
inferlite-generate  →  inferlite/cli.py :: main()
```

### 完整调用链

下面是一次推理请求从命令行到输出 token 的完整函数调用路径，括号内标注所在文件：

```
main()                                   cli.py
 │
 ├─ parse_args()                         cli.py
 │     解析 --model-dir / --prompt / --max-new-tokens / --chat-template
 │
 ├─ AutoTokenizer.from_pretrained()      transformers（外部库）
 │     加载 tokenizer，text ↔ token ids
 │
 ├─ tokenizer.apply_chat_template()      仅 --chat-template 时
 │     将裸 prompt 包装成 ChatML 格式
 │
 ├─ load_causal_lm_from_hf()             model/weights.py
 │    ├─ ModelConfig.from_json()         config.py
 │    │     读取 config.json，解析 11 个超参
 │    ├─ Qwen3ForCausalLM(config)        model/qwen3.py
 │    │     构建神经网络（embed + 28 层 + lm_head）
 │    └─ load_weights_into_model()       model/weights.py
 │          safetensors → key 映射 → model.load_state_dict()
 │
 ├─ model.eval()                         切换到推理模式
 ├─ GreedySampler()                      sampler/greedy.py
 ├─ EngineCore(model, sampler)           engine/core.py
 │
 ├─ tokenizer.encode(prompt_text)        → input_ids [1, T]
 │
 └─ torch.no_grad() + generate()         engine/core.py
       │
       └─ for step in range(max_new_tokens):
             │
             ├─ engine.step(input_ids)
             │    ├─ model(input_ids, logits_to_keep=1)   Qwen3ForCausalLM.forward()
             │    │    ├─ embed_tokens(input_ids)          → [B,T,H]
             │    │    ├─ for layer in layers:             × 28 次
             │    │    │    └─ DecoderLayer.forward()      → [B,T,H]
             │    │    │         ├─ RMSNorm                   layers.py
             │    │    │         ├─ GQAAttention.forward()    attention.py
             │    │    │         │    ├─ q/k/v proj
             │    │    │         │    ├─ q_norm, k_norm   QK-Norm
             │    │    │         │    ├─ apply_rotary_pos_emb  RoPE
             │    │    │         │    ├─ repeat_kv        GQA 扩展 KV heads
             │    │    │         │    ├─ attn_scores + causal_mask
             │    │    │         │    └─ softmax → matmul(v) → o_proj
             │    │    │         ├─ RMSNorm
             │    │    │         └─ SwiGLUMLP.forward()       layers.py
             │    │    ├─ final RMSNorm
             │    │    └─ lm_head(hidden[:, -1:, :])      → [B,1,V]
             │    └─ sampler(logits[:, -1, :])            → [B,1]  argmax
             │
             ├─ input_ids = cat([input_ids, next_token], dim=1)
             └─ if next_token == eos_token_id: break
```

### 关键数据的形状变化

理解 tensor shape 的变化是读懂代码的捷径：

```
input_ids                  [1, T]           整数，token id 序列（T 随每步递增）
  ↓ embed_tokens
hidden_states              [1, T, 1024]     浮点嵌入向量，H=hidden_size
  ↓ × 28 DecoderLayer
hidden_states              [1, T, 1024]     形状不变，经 28 层 Attention+MLP 变换
  ↓ lm_head 前 [:, -1:, :]（logits_to_keep=1 截断）
hidden_states              [1, 1, 1024]     只取最后一个位置
  ↓ lm_head (Linear, no bias)
logits                     [1, 1, 151936]   词表分数，V=vocab_size
  ↓ [:, -1, :]
logits                     [1, 151936]      squeeze time 维
  ↓ argmax(dim=-1, keepdim=True)
next_token                 [1, 1]           下一个 token id
```

### 读代码的推荐顺序

从简单到复杂，依次理解每个文件：

| 顺序 | 文件 | 理由 |
|------|------|------|
| 1 | `config.py` | 纯数据类，11 个超参，无 tensor 操作，最容易入手 |
| 2 | `sampler/greedy.py` | 最短，一行 argmax，理解采样层的接口契约 |
| 3 | `engine/protocol.py` | 理解 Protocol 是什么，为什么 engine 不绑定具体模型类 |
| 4 | `engine/core.py` | step() 和 generate loop，核心调度逻辑 |
| 5 | `model/layers.py` | 叶节点：RMSNorm、SwiGLUMLP、RotaryEmbedding |
| 6 | `model/attention.py` | 最复杂：GQA + QK-Norm + RoPE + causal mask |
| 7 | `model/qwen3.py` | 组合：DecoderLayer → Qwen3Model → Qwen3ForCausalLM |
| 8 | `model/weights.py` | HF checkpoint 加载和 state_dict key 映射 |
| 9 | `cli.py` | 最后看入口，理解所有组件如何被组装在一起 |

> 测试是另一条阅读路径：`tests/unit/` 下每个文件对应一个模块。测试代码比实现代码更短、更聚焦，往往更容易看清一个函数的输入/输出契约。从 `test_sampler.py` 或 `test_engine_core.py` 开始是个好选择。

---

## 权重加载：从 safetensors 到 PyTorch 模型

> **本章要回答的问题**：HF 下载的 checkpoint 是什么格式？如何把它的权重映射到 inferlite 的模型类里？`tie_word_embeddings` 的语义是什么？

### 3.1 HF checkpoint 的结构

Qwen3-0.6B 下载后目录结构：

```
Qwen3-0.6B/
├── config.json            超参配置（hidden_size、num_heads 等）
├── model.safetensors      模型权重（单文件，约 1.1GB）
├── tokenizer.json         词表
└── tokenizer_config.json  包含 chat_template
```

`model.safetensors` 本质是一个 `dict[str, Tensor]`，key 为 PyTorch 风格的模块路径：

```
model.embed_tokens.weight              [151936, 1024]
model.layers.0.self_attn.q_proj.weight [ 2048,  1024]
model.layers.0.self_attn.k_proj.weight [ 1024,  1024]
model.layers.0.self_attn.v_proj.weight [ 1024,  1024]
model.layers.0.self_attn.o_proj.weight [ 1024,  2048]
model.layers.0.self_attn.q_norm.weight [  128]
model.layers.0.self_attn.k_norm.weight [  128]
model.layers.0.mlp.gate_proj.weight    [2816,  1024]
model.layers.0.mlp.up_proj.weight      [2816,  1024]
model.layers.0.mlp.down_proj.weight    [1024,  2816]
model.layers.0.input_layernorm.weight  [1024]
...  (28 层)
model.norm.weight                      [1024]
lm_head.weight                         [151936, 1024]
```

### 3.2 两种加载目标与 key 映射

inferlite 支持两种加载目标，对应不同的 state_dict key 前缀处理：

| target | 模型类 | HF key `model.xxx` → inferlite key | `lm_head.weight` |
|--------|--------|--------------------------------------|-----------------|
| `backbone` | `Qwen3Model` | 去掉 `model.` 前缀 → `xxx` | 跳过 |
| `causal_lm` | `Qwen3ForCausalLM` | 原样保留 → `model.xxx` | 加载 |

原因：`Qwen3Model` 是裸 backbone，其 state_dict key 形如 `layers.0...`；而 `Qwen3ForCausalLM` 内部字段命名刻意对齐 HF（`self.model`、`self.lm_head`），state_dict key 与 HF checkpoint 天然一致。

```python
# inferlite/model/weights.py L48-L85
def map_hf_key_to_inferlite_key(hf_key: str, target: str) -> str | None:
    if target == "backbone":
        if hf_key in {"lm_head.weight"}:
            return None                          # 跳过
        if hf_key.startswith("model."):
            return hf_key.removeprefix("model.") # 去掉 model. 前缀
    elif target == "causal_lm":
        return hf_key                            # 原样
```

> 代码定位：[weights.py L48–L85](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/model/weights.py#L48-L85)

加载流程还包含 shape 预检验（在 `load_state_dict` 之前提前暴露 mismatch），报错时同时显示 HF key 和 inferlite key，定位更直接。

### 3.3 tie_word_embeddings：权重共享的语义

Qwen3-0.6B 的 `config.json` 中 `tie_word_embeddings: true`。其含义是：`lm_head.weight`（将 hidden_state 投影回 vocab 分数）与 `embed_tokens.weight`（将 token id 映射成向量）共享同一个 tensor。

语义上，两者互为逆操作：embedding 是 token→向量，lm_head 是向量→token。共享权重意味着同一个词的"嵌入方向"和"解码方向"完全一致，同时减少约 151936×1024×4 bytes ≈ 590MB 参数量。

实现：**引用赋值，不是值拷贝**。

```python
# inferlite/model/qwen3.py L285-L286
if config.tie_word_embeddings:
    self.lm_head.weight = self.model.embed_tokens.weight
    # 之后：self.lm_head.weight is self.model.embed_tokens.weight → True
```

这个赋值在 `__init__` 里完成。后续 `load_state_dict` 把权重数据写入 `embed_tokens.weight` 时，`lm_head` 通过同一个引用自动看到更新，无需额外处理。

> 代码定位：[qwen3.py L250–L286](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/model/qwen3.py#L250-L286)

---

## 模型前向：从 token id 到 logits

> **本章要回答的问题**：Qwen3 的神经网络是什么结构？输入 token ids 如何一步步变成 logits？pre-norm 与 post-norm 有什么区别？

### 4.1 三层结构

```
Qwen3ForCausalLM
    └── Qwen3Model (backbone)
            ├── embed_tokens: nn.Embedding [V, H]
            ├── layers: nn.ModuleList [28 × DecoderLayer]
            │       └── DecoderLayer
            │               ├── input_layernorm:        RMSNorm [H]
            │               ├── self_attn:              GQAAttention
            │               ├── post_attention_layernorm: RMSNorm [H]
            │               └── mlp:                    SwiGLUMLP
            └── norm: RMSNorm [H]
    └── lm_head: nn.Linear [H → V, no bias]
```

Qwen3-0.6B 参数：`H=1024`（hidden_size），`V=151936`（vocab_size），28 层，16 Q heads，8 KV heads，head_dim=128。

### 4.2 Pre-norm 结构

现代 decoder-only 大模型（LLaMA / Qwen / Mistral）几乎全用 **pre-norm**，原始 Transformer 论文的 post-norm 在深层网络中梯度不稳定问题较严重。

两种结构对比：

```
post-norm（原始 Transformer）:  x → sublayer(x) → x + residual → Norm
pre-norm（现代主流）:            x → Norm → sublayer(x) → x + residual
```

pre-norm 的 residual 主干是"干净"的——归一化只作用于送入子层的分支，不修改主干信号。梯度反传时，主干提供了一条无衰减的直接通路。

```python
# inferlite/model/qwen3.py L106-L132
# Attention 子层
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)      # pre-norm
hidden_states = self.self_attn(hidden_states, position_ids)
hidden_states = residual + hidden_states                  # residual add

# MLP 子层
residual = hidden_states   # ← 必须重新赋值，起点是 attention 之后的状态
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

注意第二段 `residual = hidden_states` 必须重新赋值。复用 attention 前的 residual 会跳过 attention 子层的贡献——这是实现 pre-norm 时的一个常见错误。

> 代码定位：[qwen3.py L89–L133](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/model/qwen3.py#L89-L133)

### 4.3 lm_head 与 logits_to_keep 优化

朴素实现中，`lm_head` 对所有 T 个 token 位置做 `[B,T,H] → [B,T,V]` 的矩阵乘。但自回归推理只需要**最后一个位置**的 logits 用于预测下一个 token，前 T-1 个位置的计算纯属浪费。

inferlite 引入 `logits_to_keep` 参数，在进 lm_head 之前截取：

```python
# inferlite/model/qwen3.py L307-L313
hidden_states = self.model(input_ids, position_ids=position_ids)
if logits_to_keep is not None:
    hidden_states = hidden_states[:, -logits_to_keep:, :]  # [B, 1, H]
logits = self.lm_head(hidden_states)                        # [B, 1, V]
```

`EngineCore.step()` 固定传 `logits_to_keep=1`，每步节省 (T-1)/T 的 lm_head 投影计算。序列越长，收益越大。

> 代码定位：[qwen3.py L288–L314](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/model/qwen3.py#L288-L314)，[core.py L51–L53](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/engine/core.py#L51-L53)

---

## GQA Attention：从 MHA 到分组查询注意力

> **本章要回答的问题**：Qwen3 为什么不用标准 MHA？GQA 在计算层面如何实现？QK-Norm 和 RoPE 各起什么作用，为什么顺序不能反？

### 5.1 MHA → GQA 的演变动机

标准 **Multi-Head Attention（MHA）** 中 Q/K/V head 数量相同（如 GPT-3 的 96 heads）。推理时，每个 token 的 K、V 向量需要缓存（即 KV Cache），内存占用为：

```
KV Cache 大小 = 2 × num_layers × seq_len × num_heads × head_dim × dtype_bytes
```

对于 GPT-3（96 layers，96 heads，head_dim=128，fp16）：每 token 约 37.7 KB。长序列 + 大 batch 时 KV Cache 成为主要内存瓶颈。

**Grouped Query Attention（GQA）**（Ainslie et al., 2023）：减少 K/V head 数，多个 Q head 共享一对 K/V head：

```
Qwen3-0.6B:
  num_heads (Q)          = 16
  num_key_value_heads    = 8
  num_key_value_groups   = 2   (每 2 个 Q head 共享 1 个 KV head)
```

KV Cache 内存缩减为 MHA 的 `n_kv / n_q = 8/16 = 50%`，但 attention 计算质量接近 MHA（远优于 Multi-Query Attention 的极端情况 n_kv=1）。

### 5.2 repeat_kv：将 KV 扩展到 Q head 数

做 `q @ k^T` 之前，需要将 k/v 从 8 heads "复制"到 16 heads。`expand` 不分配新内存，只创建广播视图；`reshape` 合并维度：

```python
# inferlite/model/attention.py L44-L88
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 输入: [B, 8, T, D]
    # 目标: [B, 16, T, D]
    b, n_kv, t, d = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]         # [B, 8, 1, T, D]
        .expand(b, n_kv, n_rep, t, d)           # [B, 8, 2, T, D]  ← 广播，无内存拷贝
        .reshape(b, n_kv * n_rep, t, d)         # [B, 16, T, D]
    )
```

> 代码定位：[attention.py L44–L88](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/model/attention.py#L44-L88)

### 5.3 QK-Norm 与 RoPE 的顺序依赖

Qwen3 相比 LLaMA 2 增加了 **QK-Norm**：在每个 head 的 Q/K 向量上做 RMSNorm，作用是稳定 attention score 的数值范围，防止 q·k 点积在深层模型中数值爆炸。

关键顺序约束：**必须先 QK-Norm，再 RoPE**。

原因：RoPE 通过旋转矩阵将位置信息编码进 Q/K 向量。若先做 RoPE 再做 RMSNorm，归一化会消除向量的模长信息，而模长在旋转后已经编码了位置差异——这会破坏 RoPE 的位置编码。

```python
# QK-Norm 在 RoPE 之前
q = q.view(B, T, n_q, D).transpose(1, 2)      # [B, n_q, T, D]
k = k.view(B, T, n_kv, D).transpose(1, 2)     # [B, n_kv, T, D]

q = self.q_norm(q)                             # RMSNorm on head_dim
k = self.k_norm(k)

cos, sin = self.rotary_emb(position_ids)
q, k = apply_rotary_pos_emb(q, k, cos, sin)   # RoPE 注入位置信息
```

### 5.4 Causal Mask 与 Scaled Dot-Product Attention

```python
# attention score 计算
attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
# scaling = head_dim ** -0.5 = 128 ** -0.5 ≈ 0.0884
# 防止 q·k 点积随 head_dim 增大而方差爆炸，导致 softmax 梯度消失

# causal mask：token i 不能看到 j > i 的位置
# mask 值为 0（保留）或 -inf（屏蔽），加到 attn_scores 上
attn_scores = attn_scores + causal_mask

attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
# softmax 在 float32 下计算（数值稳定性），再 cast 回 q 的 dtype

output = torch.matmul(attn_weights, v)  # [B, n_q, T, D]
output = output.transpose(1, 2).reshape(B, T, n_q * D)
output = self.o_proj(output)            # [B, T, H]
```

Causal mask 的构造：上三角矩阵（不含对角线）置为 `-inf`，使 softmax 后对应权重趋于 0。这保证了自回归生成的因果性——位置 t 的 attention 只能依赖 t 及之前的 token。

---

## 采样与生成：GreedySampler 和自回归循环

> **本章要回答的问题**：采样层和引擎层如何解耦？generate loop 的每一步做了什么？EOS 停止条件为什么是 `.all()` 而不是 `.any()`？

### 6.1 Sampler 的接口设计

Sampler 是推理引擎的**策略层**，与模型结构、引擎调度完全解耦。接口约定：

```
sampler(logits: [B, V]) -> next_token: [B, 1]
```

Greedy 实现：

```python
# inferlite/sampler/greedy.py L21-L23
def __call__(self, logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1, keepdim=True)
    # keepdim=True 保持 [B, 1] 形状，便于后续 torch.cat([input_ids, next_token], dim=1)
```

M1 只实现 greedy，后续 milestone 可以在不改动 engine 的前提下替换为 temperature sampling / top-k / top-p / beam search。

> 代码定位：[greedy.py L21–L23](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/sampler/greedy.py#L21-L23)

### 6.2 EngineCore：调度与模型的解耦点

```python
# inferlite/engine/core.py L31-L59
class EngineCore:
    def __init__(self, model: LLMModel, sampler: GreedySampler) -> None:
        self.model = model      # 只依赖 Protocol，不绑定 Qwen3ForCausalLM
        self.sampler = sampler

    def step(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_ids, logits_to_keep=1)  # [B, 1, V]
        next_token_logits = logits[:, -1, :]              # [B, V]
        return self.sampler(next_token_logits)             # [B, 1]
```

`EngineCore` 只依赖 `LLMModel` Protocol（定义在 `engine/protocol.py`），而非具体的 `Qwen3ForCausalLM` 类。这使得未来替换为带 KV Cache 的模型版本时，`EngineCore` 代码无需改动。

### 6.3 generate loop：token 级自回归循环

```python
# inferlite/engine/core.py L85-L92
for _ in range(max_new_tokens):
    next_token = engine.step(input_ids)
    input_ids = torch.cat([input_ids, next_token], dim=1)
    if eos_token_id is not None and (next_token == eos_token_id).all():
        break
return input_ids
```

EOS 检查使用 `.all()` 而非 `.any()`：batch 内**所有序列**都生成 EOS 时才停止，避免因单条序列先达到 EOS 而截断 batch 中仍未完成的序列。

**当前实现的根本局限**：无 KV Cache，每步 `engine.step(input_ids)` 将完整的 `input_ids`（含所有历史 token）重新过 28 层 Attention。计算量随序列长度 T 线性增长，每步的 Attention 计算是 O(T²)，整体生成 N 个 token 是 O(N·T²)。M2 的 KV Cache 会把单步 Attention 降到 O(T)。

> 代码定位：[core.py L62–L93](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/engine/core.py#L62-L93)

---

## Python 类型系统：Protocol、dataclass 与 argparse

> **本章要回答的问题**：inferlite 用了哪些 Python 类型特性？它们解决了什么具体的工程问题？这些特性本身是如何工作的？

### 7.1 `@dataclass(frozen=True)`：配置的不可变契约

`config.py` 用 `@dataclass(frozen=True)` 定义 `ModelConfig`：

```python
@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int        # H = 1024
    num_hidden_layers: int  # N = 28
    num_attention_heads: int
    num_key_value_heads: int
    # ... 共 11 个字段
```

`@dataclass` 自动生成 `__init__`、`__repr__`、`__eq__`，避免手写样板代码。`frozen=True` 使所有字段在创建后不可修改（类似 immutable tuple），尝试赋值会抛 `FrozenInstanceError`。

**为什么推理配置需要不可变**：`ModelConfig` 在 `__init__` 里传入 Qwen3 的所有层，28 个 `DecoderLayer` 共享同一个 config 引用。如果 config 可变，某处修改 `config.hidden_size` 会影响所有已创建的层，造成难以排查的 shape mismatch。`frozen=True` 把"配置是只读契约"的意图写进了类型系统。

`__post_init__` 在 `__init__` 之后自动调用，用于早期校验：

```python
def __post_init__(self):
    assert self.num_attention_heads % self.num_key_value_heads == 0, \
        f"GQA: n_q={self.num_attention_heads} 必须能被 n_kv={self.num_key_value_heads} 整除"
```

"早 fail"的价值：报错在构造期（`ModelConfig(...)` 这行），而不是运行到 `repeat_kv` 时的 tensor reshape 失败——后者的 traceback 更难定位根因。

> 代码定位：[config.py L41–L75](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/config.py#L41-L75)

### 7.2 `typing.Protocol`：结构化接口约定

Python 的 `typing.Protocol`（PEP 544）实现**结构子类型（structural subtyping）**，即"鸭子类型的静态检查版本"：只要一个对象有指定的方法签名，它就被认为满足该协议，无需继承。

`engine/protocol.py` 定义了 `LLMModel`：

```python
from typing import Protocol

class LLMModel(Protocol):
    def __call__(
        self, input_ids: torch.Tensor, *, logits_to_keep: int | None = None
    ) -> torch.Tensor:
        ...
```

`EngineCore` 的构造函数接受 `model: LLMModel`，而非 `model: Qwen3ForCausalLM`：

```python
class EngineCore:
    def __init__(self, model: LLMModel, sampler: GreedySampler) -> None:
        self.model = model
```

**与 ABC（Abstract Base Class）的区别**：

| 方式 | 满足方式 | 适用场景 |
|------|---------|---------|
| `ABC` + `abstractmethod` | 必须显式继承 `ABC` | 希望强制子类继承关系 |
| `typing.Protocol` | 只要方法签名匹配即可（无需继承） | 解耦：调用方定义接口，实现方不感知 |

使用 Protocol 的好处：测试里的 `FakeModel` 不需要 `from inferlite.engine.protocol import LLMModel`，只要实现了 `__call__(input_ids, *, logits_to_keep=...)` 就自动满足协议，mypy 也能静态验证。

> 代码定位：[protocol.py L27–L58](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/engine/protocol.py#L27-L58)

### 7.3 `argparse`：CLI 参数解析

`cli.py` 用 `argparse` 定义命令行接口：

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal greedy generation.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--chat-template", action="store_true")
    return parser.parse_args(argv)
```

`argv=None` 时使用真实命令行参数（`sys.argv[1:]`）；测试时可传入显式列表，避免修改全局 `sys.argv`：

```python
# tests/unit/test_cli.py
def test_parse_args_defaults():
    args = parse_args(["--model-dir", "/tmp/model", "--prompt", "hello"])
    assert args.max_new_tokens == 8
    assert args.chat_template is False
```

这个设计使 CLI 的参数解析逻辑可以被独立单测，无需真正启动推理。

> 代码定位：[cli.py L33–L57](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/inferlite/cli.py#L33-L57)

### 7.4 `model.eval()` 与 `torch.no_grad()`：推理模式的两把锁

这两个 API 经常一起出现，但作用层次不同：

**`model.eval()`** 切换模型的"行为模式"，影响：
- `Dropout`：训练时随机丢弃神经元，推理时全部保留（`eval()` 后生效）
- `BatchNorm`：训练时用 batch 统计量，推理时用运行时累积的均值/方差

Qwen3 当前实现不包含 Dropout，但 `model.eval()` 是推理代码的必要规范——如果未来加入任何训练/推理行为不同的层，忘记调用会造成输出偏差。

**`torch.no_grad()`** 禁用 autograd 计算图构建。PyTorch 默认对每个 tensor 操作维护反向传播所需的元数据（梯度函数、输入引用等）。推理时不需要反传：

```python
model.eval()                          # 切换行为模式
with torch.no_grad():                 # 禁用梯度图
    output_ids = generate(engine, input_ids, ...)
```

两者要同时使用，不可省略其中一个。

---

## 测试策略：分层隔离与数值对齐

> **本章要回答的问题**：如何在没有 GPU 和模型文件的 CI 环境中跑测试？如何区分"纯逻辑正确性"和"数值对齐"两种测试目标？pytest 的哪些机制支撑了这种分层？

### 8.1 分层测试设计

```
tests/
├── unit/               # 单元测试：无 GPU，无模型文件，CI 全量运行
│   ├── test_attention.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_decoder_layer.py
│   ├── test_engine_core.py
│   ├── test_generate.py
│   ├── test_mlp.py
│   ├── test_qwen3_causal_lm.py
│   ├── test_qwen3_model.py
│   ├── test_rmsnorm.py
│   ├── test_rope.py
│   ├── test_sampler.py
│   └── test_weights.py
└── integration/
    └── test_real_qwen3_smoke.py   # 需要本地 Qwen3-0.6B 文件，打 @local_model marker
```

| 层级 | 路径 | 运行方式 | 依赖 | 目的 |
|------|------|---------|------|------|
| Unit Tests | `tests/unit/` | CI 全量运行 | 无真实模型文件 | 逻辑正确性，快速反馈 |
| Integration Smoke | `tests/integration/` | 本地 `-m local_model` | 需要 Qwen3-0.6B 文件 | 端到端数值对齐 |

### 8.2 FakeModel：隔离外部依赖的核心技术

Unit tests 不能依赖真实模型文件（1.1GB，不应该进 CI）。解决方案是 `FakeModel`——一个确定性的假模型：

```python
# 示例：测试 EOS 停止逻辑，与真实模型无关
class FakeModel:
    def __init__(self, always_returns_token: int):
        self.always_returns_token = always_returns_token

    def __call__(self, input_ids, *, logits_to_keep=None):
        B, T = input_ids.shape
        # 构造确定性 logits：always_returns_token 位置设极大值
        logits = torch.zeros(B, 1, VOCAB_SIZE)
        logits[:, :, self.always_returns_token] = 1e9
        return logits

def test_generate_stops_early_at_eos():
    model = FakeModel(always_returns_token=EOS_ID)
    engine = EngineCore(model, GreedySampler())
    result = generate(engine, input_ids, max_new_tokens=10, eos_token_id=EOS_ID)
    assert result.shape[1] == input_ids.shape[1] + 1  # 只生成了 1 步
```

`FakeModel` 不需要继承任何类，只要实现 `__call__` 签名就满足 `LLMModel` Protocol。这正是 Protocol 与 FakeModel 配合的价值：Protocol 定义了接口边界，测试可以不依赖真实实现。

### 8.3 pytest marker：自定义测试标签

`@pytest.mark` 是 pytest 的标签系统，用于给测试贴上分类标记，再通过 `-m` 选择性运行：

```python
# tests/integration/test_real_qwen3_smoke.py
import pytest

@pytest.mark.local_model
def test_real_qwen3_smoke():
    # 需要本地 Qwen3-0.6B 文件
    ...
```

自定义 marker 需要在 `pyproject.toml` 注册（否则会有 warning）：

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "local_model: tests that require a locally downloaded model (skipped in CI by default)",
]
```

使用方式：

```bash
# CI：跳过需要真实模型的测试（默认）
uv run pytest tests/unit/ -q

# 本地验证数值对齐（需要 Qwen3-0.6B 已下载）
uv run pytest -m local_model -v
```

### 8.4 数值对齐验证

核心数值测试（`test_qwen3_causal_lm.py`）用 HuggingFace transformers 的 `Qwen3ForCausalLM` 作为参照，验证 inferlite 实现的 logits 输出与官方实现在 atol=1e-3 以内完全一致。这是 M1 正确性的最强保证：

```python
# 思路：相同权重 + 相同输入 → logits 差异在浮点误差范围内
hf_logits = hf_model(**inputs).logits
our_logits = our_model(input_ids, logits_to_keep=T)
assert torch.allclose(hf_logits, our_logits, atol=1e-3)
```

---

## 打包与开发工作流：pyproject.toml、uv 与 Makefile

> **本章要回答的问题**：`pyproject.toml` 怎么把 Python 代码变成可安装的命令行工具？`uv` 和 `Makefile` 如何组合成一个完整的开发工作流？

### 9.1 `pyproject.toml`：现代 Python 打包

`pyproject.toml` 是 Python 打包的统一配置文件（PEP 517/518/621），取代了过去的 `setup.py` + `setup.cfg` + `requirements.txt` 碎片化方案。

inferlite 的关键配置：

```toml
[project]
name = "inferlite"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4,<3",
    "transformers>=5.10,<6",
    "safetensors>=0.4,<1",
    "fastapi>=0.110,<1",    # server/ 预留
    "uvicorn>=0.30,<1",     # server/ 预留
]

[project.scripts]
inferlite-generate = "inferlite.cli:main"   # ← 这行把函数变成命令行工具
```

`[project.scripts]` 定义 **entry points**：`uv sync` / `pip install -e .` 后，会在 PATH 里创建 `inferlite-generate` 可执行文件，直接映射到 `inferlite/cli.py` 的 `main()` 函数。这是 Python 项目发布命令行工具的标准方式。

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["inferlite"]
```

`build-system` 指定构建后端（这里是 `hatchling`），`uv build` 会用它打包成 wheel 文件。

**依赖版本策略**：使用 `>=major.minor,<next_major` 的松散下界 + 严格上界，兼顾灵活升级与避免大版本破坏性变更。精确版本锁定在 `uv.lock` 里（机器生成，作为事实源）。

> 代码定位：[pyproject.toml](https://github.com/luhao-lab/inferlite/blob/m1%2Fnaive-forward/pyproject.toml)

### 9.2 `uv`：现代 Python 依赖管理

`uv` 是 Rust 编写的超快 Python 包管理器，本项目用它代替 `pip` + `venv`：

```bash
uv sync          # 根据 uv.lock 精确还原依赖（等价于 pip install -r requirements.txt）
uv run pytest    # 在 venv 里运行命令（等价于 . .venv/bin/activate && pytest）
uv run inferlite-generate --model-dir ...  # 运行已注册的命令行工具
```

`uv.lock` 是 `uv sync` 生成的精确版本锁定文件，记录每个包的确切版本和 hash，保证不同机器、不同时间的依赖环境完全一致。

### 9.3 `Makefile`：开发任务自动化

`Makefile` 是项目的"任务书"，记录所有常用开发命令，避免每次查文档：

```makefile
test:          ## run all tests
	uv run pytest

test-fast:     ## run only fast tests (skip slow markers)
	uv run pytest -m "not slow"

lint:          ## ruff check
	uv run ruff check .

fmt:           ## ruff format
	uv run ruff format .

typecheck:     ## mypy
	uv run mypy inferlite

docs-serve:    ## serve docs locally at http://localhost:8000
	uv run mkdocs serve
```

使用 `make help` 可以打印所有可用命令（通过 `##` 注释自动提取）。

**ruff** 是 Rust 编写的 Python linter + formatter，速度比 flake8/black 快数十倍，本项目在 `pyproject.toml` 里配置：

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]  # pycodestyle + pyflakes + isort + bugbear + upgrade
ignore = ["E501"]  # 行长度由 fmt 控制
```

`UP`（pyupgrade）规则会自动把旧写法升级到新语法，如 `Union[int, str]` → `int | str`（Python 3.10+）。

---

## M1 的局限与 M2 展望

> **本章要回答的问题**：M1 的根本性能瓶颈在哪里？KV Cache 的原理是什么？M2 要解决什么问题？

| 维度 | M1 现状 | M2 目标 |
|------|---------|---------|
| KV Cache | 无，每步 full forward O(T²) | Paged / Static KV Cache，单步 O(T) |
| 硬件 | CPU fp32 | MPS / CUDA，dtype 控制 |
| 并发 | 单请求 | Continuous batching |
| 采样 | 仅 greedy | Temperature / top-k / top-p |

KV Cache 是推理效率的核心。自回归推理每步计算中，已生成 token 的 Key/Value 向量实际上不会改变（因为 causal mask 保证它们的 attention 不受新 token 影响）。将这些 K/V 向量缓存起来，下一步只计算新 token 的 Q，并与缓存中的 K/V 做 attention，将每步计算量从 O(T²) 降至 O(T)，生成 N 个 token 的总复杂度从 O(N·T²) 降至 O(N·T)。
