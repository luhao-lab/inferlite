# inferlite — 从零手写 LLM 推理引擎 · 里程碑路线

> 目标：在 `luhao-lab/inferlite` 仓库里，**手敲** 一个可读、可跑、可解释的 LLM 推理框架，覆盖 vLLM 的核心思想（KV cache / PagedAttention / Continuous Batching / Prefix Cache），并通过持续迭代不断扩充（MoE / Spec Decoding / Triton kernel / 量化 / VLM …）。
>
> 节奏不绑定时间，按**里程碑驱动**：M1–M5 完成核心 demo（"跑得起来"），M6+ 在同仓库长期扩充，每个里程碑配一篇知乎文章。

<!-- anchor:positioning -->
## 0. 定位与不变量

- **仓库**：`luhao-lab/inferlite`（公开 / MIT）
- **代码全手敲**：Agent 不写 `inferlite/*.py`，只做：研究、计划、资料检索、原理讲解、Review、文章草稿
- **里程碑闭环**：每个 M 完成 = ① 代码 push ② 文章发知乎 ③ PROGRESS 更新
- **学习 > 性能**：优先可读性；性能优化作为后续里程碑慢慢加
- **对照参照**：
  - `MinivLLM` — 最小教学版，模块映射底图
  - `nano-vllm`（GeeeekExplorer）— 千行版结构最干净
  - `vLLM v1`（`vllm/v1/`）— 工程级对照，**只读不抄**
  - `SGLang` runtime — RadixAttention / 调度策略对照

### 0.1 协作原则（基于 M0 + T1 经验沉淀，2026-06-07）

> 这五条是 R2 体检后从前期实操中抽取的高频失败模式，写在 PLAN 顶部用于约束后续每个 M。

1. **地基与算法是两个频道**（→ Lessons L3）
   同一会话 / 同一 commit 内，只 push **基础设施改动**（CI / Makefile / docs）或**算法实现**（layers / engine）之一，不要混。混在一起会让 review 失焦、回滚成本高。

2. **任务卡 7 字段写完再开工**
   `前置 / 产出 / 算法核心 / L0 测试 / DoD / 易踩坑 / 估时` 全部填好（即使只填一句"待补"也比空着强）。开工后任何"哦还有一个事"先回填到对应字段，不直接动代码。

3. **每张完成的任务卡 → 至少回填 1 处知识/教训**
   - 完成总结段是最低门槛
   - 如果踩坑了 → `lessons.md` 加 L 编号
   - 如果用到了新论文/库/概念 → `knowledge.md` 对应章节加卡片
   - 如果产生方法论决策 → `knowledge.md → 架构决策章节` 加 ADR

4. **`make doctor` 是 commit 的零成本守卫**
   PLAN/PROGRESS/任务卡 改动后 push 前手动跑一次。doctor 没绿不 push。

5. **`[P]` 标记并行机会，但单线推进默认**
   M1 §4 的 `[P] ✓` 表示**可以**并行，不表示**必须**并行。学习导向项目里，单线深读比并行铺面更高效；只在等模型下载等 IO 阻塞时考虑切到并行卡。

<!-- anchor:architecture -->
## 1. 框架的 4 层抽象（贯穿所有里程碑）

```
┌─────────────────────────────────────────┐
│  L4  Server 层    协议入口（同步 / SSE / OpenAI 兼容）
├─────────────────────────────────────────┤
│  L3  Engine 层    调度器（continuous batching / 采样 / spec）
├─────────────────────────────────────────┤
│  L2  Memory 层    KV 存储与复用（cache → 分页 → 前缀复用）
├─────────────────────────────────────────┤
│  L1  Model 层     一次前向（dense / MoE / attention kernel）
└─────────────────────────────────────────┘
```

| 层 | 本质问题 | 解决方案 |
| --- | --- | --- |
| L1 Model | tokens → logits | Transformer 前向 |
| L2 Memory | KV 怎么存才能既不浪费又能复用 | 分页 + 引用计数 |
| L3 Engine | 多请求怎么挤进同一次前向 | 按 step 重组 batch |
| L4 Server | 外部怎么调用 | HTTP/SSE 协议 |

> 一句话心智模型：**用一个调度器（L3），把多个请求复用同一份模型权重（L1）和同一块显存（L2），通过统一协议（L4）对外服务。**

#### 里程碑 × 4 层抽象映射表

每个里程碑明确徽章一到两层主战场，避免"一个 M 什么都动":

| 里程碑 | L1 Model | L2 Memory | L3 Engine | L4 Server | 主战场 |
| --- | :---: | :---: | :---: | :---: | --- |
| **M1·P1** Qwen3 数值对齐 | ⚫主 | | | | L1 算子 + 整模型 logits allclose |
| **M1·P2** 单序列前向 | ⚫ | | ⚫主 | | 最小 Engine + CLI 出字 |
| **M2** KV Cache | ⚫ | ⚫主 | | | L2 首实现 `ContiguousKVCache` |
| **M3** Continuous Batching | | | ⚫主 | | L3 调度器诞生 |
| **M4** PagedAttention（PyTorch） | | ⚫主 | ⚫ | | L2 升级为 `PagedKVCache` |
| **M5a** API 闭环 | | | ⚫（sampler） | ⚫主 | OpenAI API + SSE |
| **M5b** 复用能力 | | ⚫主（prefix） | | ⚫ | Prefix cache + reasoning 字段 |
| **M5c** 工程收口 | | | | ⚫ | Benchmark + CI + v1.0 |
| **M6** MoE for-loop | ⚫主 | | | | L1 添加二路模型、验证 Registry |
| **M7** Spec Decoding (n-gram) | | | ⚫主 | | L3 加 `Drafter` Plugin |
| **M8** Triton PagedAttention | ⚫ | ⚫主 | | | L1↔L2 边界 kernel |
| **M9** MoE grouped GEMM | ⚫主 | | | | L1 优化 |
| **M10** EAGLE-1 spec | ⚫ | | ⚫主 | | L3 插件升级 |
| **M11** Long context (YaRN) | ⚫主 | ⚫ | | | L1 RoPE 重映射 + L2 长度 |
| **M12** Chunked Prefill | | | ⚫主 | | L3 调度策略 |
| **M13** VLM 教学版 | ⚫主 | | | ⚫ | L1 + L4 multipart 请求 |
| **M14** VLM 工程化 | ⚫ | ⚫主 | ⚫ | | L2 image hash prefix cache |
| **M15+** | 取决于选题 | | | | 量化/MLA ⚫ L1 · TP/PP ⚫ L3 · Disagg P/D ⚫ L4 |

> 读法：**主**表示主战场（该 M 可调用、可讲验、可写文章的核心层），不带标记的⚫表示"有轻微修改但不是重点"。同一讲里程碑只让 1–2 层点亮，其他层保持不变量，你才能在文章里准确表达"这个 M 在学什么"。

<!-- anchor:model-extension -->
### 1.1 模型扩展机制（如何支持多模型）

让 L2/L3/L4 对"具体是哪个模型"无感知，靠三件套：**Protocol + Registry + WeightMap**。

#### A. 统一接口 `LLMModel` Protocol（M1 就定）

```python
class LLMModel(Protocol):
    config: ModelConfig                       # num_layers/num_kv_heads/head_dim/hidden/vocab/dtype/max_pos

    def forward(
        self,
        input_ids: Optional[Tensor] = None,   # 纯文本走这里 [batch, seq]
        inputs_embeds: Optional[Tensor] = None,  # 多模态走这里（VLM 把图 token embedding 塞进来）
        positions: Tensor = ...,              # 位置（PagedAttention 需要）
        kv_cache: KVCache = ...,              # 框架统一的 cache 对象
    ) -> Tensor:                              # logits [batch, seq, vocab]
        ...

    @classmethod
    def load_from_hf(cls, hf_path: str) -> "LLMModel": ...
```

调度器 / PagedAttention / Server **只调 `model.forward(...)`，永不导入具体模型类**。

> **M1 设计纪律**：哪怕只支持文本，`forward` 签名也要保留 `inputs_embeds` 参数（默认 None），M13 上 VLM 时零返工。

#### B. 模型注册表 Registry（M6 引入）

```python
@register("Qwen3ForCausalLM")
class Qwen3(LLMModel): ...

@register("Qwen3MoeForCausalLM")
class Qwen3Moe(LLMModel): ...

def load_model(hf_path):
    arch = json.load(open(f"{hf_path}/config.json"))["architectures"][0]
    return MODEL_REGISTRY[arch].load_from_hf(hf_path)
```

按 HF `config.json` 的 `architectures` 字段查表分发（vLLM `model_executor/models/registry.py` 同款）。

#### C. WeightMap（每个模型自己声明 HF → 自有命名）

```python
class Qwen3:
    WEIGHT_MAP = {
        "model.embed_tokens.weight": "tok_emb.weight",
        "model.layers.{l}.self_attn.q_proj.weight": "layers.{l}.attn.wq.weight",
        # ... 约 20–30 条
        "lm_head.weight": "lm_head.weight",
    }
```

加载逻辑统一在框架层：遍历 safetensors → 查 WEIGHT_MAP → 塞进对应参数。

#### D. 差异如何被"消化"在模型类内部

| 差异点 | 框架怎么不关心 |
| --- | --- |
| Attention 类型（MHA/GQA/MQA） | 模型内部决定 head 切分；cache 只看 `num_kv_heads` |
| 位置编码（RoPE/ALiBi/NoPE） | 模型内部加；cache 只存 K/V |
| Norm（RMSNorm/LayerNorm） | 模型内部 |
| FFN（SwiGLU/GeGLU/MoE） | 模型内部；MoE forward 黑盒 |
| 词表 / 上下文长度 | 来自 `ModelConfig`，sampler 读 `vocab_size` |
| Tokenizer | 不归框架，用 `transformers.AutoTokenizer` |

#### E. 进阶 Hook（M9+ 视需要再加）

| Hook | 用途 |
| --- | --- |
| `get_input_embeddings(input_ids)` | VLM 注入 image token embedding |
| `attention_metadata` | 让 attention 知道 prefill/decode、block_table 映射 |
| `sampler` | 大多数共用，特殊模型可重写 |
| `lora_adapter_hook` | LoRA serving |

#### F. 落地节奏（关键）

| 阶段 | 模型扩展机制实现度 |
| --- | --- |
| **M1·P1–M5** | 单文件 `inferlite/model/qwen3.py`，**不做 registry**，先钉 `LLMModel` / `Sampler` / `EngineCore.step()` 三个最小契约；KV / Scheduler / Executor / Drafter 只保留占位，等 M2/M3/M7 用真实需求反推签名 |
| **M6** | 加入 `model/registry.py` + `qwen3_moe.py`，按 `architectures` 字段分发；契约第一次被验证 |
| **M9+** | 每个新模型 100–200 行（attention + MLP + WEIGHT_MAP），90% 框架代码完全复用 |

**M1 关键纪律**：哪怕只有一个模型，也要**走 `LLMModel` 接口调用**，不要在 cli 里直接 `Qwen3()`。但除 `LLMModel` 外的协议不要过早追求“签名稳定”，避免 M2/M3 返工。


<!-- anchor:tech-stack -->
## 2. 技术栈

| 维度 | 选型 | 理由 |
| --- | --- | --- |
| 仓库 | `luhao-lab/inferlite` | 已创建，MIT，公开 |
| 语言 | Python 3.11 | 教学优先 |
| DL 框架 | PyTorch 2.4+（当前 lock：2.12.0） | SDPA、Triton 入口成熟；数值对齐以 lockfile 为准 |
| 模型 | **Qwen3-0.6B**（主，M1–M5） / M13 VLM 模型待定 | Qwen3-0.6B 同权重支持 thinking/non-thinking；VLM 候选到 M13 前再按可下载性、显存和架构复杂度确认 |
| Tokenizer | `transformers` 直接复用 | 不造 BPE 轮子 |
| 数值基准 | `transformers==5.10.2`（当前 lock） | L0/L1 allclose 只承诺对齐该版本，避免 HF 内部实现升级导致测试漂移 |
| Attention | `F.scaled_dot_product_attention` → 自写 PagedAttention（后期 Triton） | 渐进 |
| 服务层 | FastAPI | SSE 流式省事 |
| 硬件 | Mac MPS（M1–M7 主开发） + GPU（M5 benchmark / M8 Triton 必需，按需租用云 GPU） | 见 §2.5 硬件矩阵 |

<!-- anchor:hardware-validation -->
## 2.5 硬件矩阵 + 验证 + 评测约定（贯穿所有里程碑）

### 2.5.1 硬件可达性

| 里程碑 | Mac MPS (16GB+) | CPU only | 单 GPU (A10/3090+) |
| --- | --- | --- | --- |
| M1–M3 | 流畅，10–30 tok/s | 慢但可跑 | 100+ tok/s |
| M4 PagedAttention (PyTorch 伪版) | 正确性可验 | 可跑 | 显存利用率才有意义 |
| M5 benchmark | 主开发可 Mac，**性能数对照必须 GPU** | 不推荐 | 必需 |
| M6 MoE A2.7B | 32GB 内存够，速度感人 | 太慢 | 推荐 |
| M7 n-gram Spec | 全平台可跑 | 可 | 推荐 |
| **M8 Triton kernel** | **不支持 MPS** | 不支持 | **必需 NVIDIA GPU** |
| M9+ (grouped GEMM / EAGLE / 量化 / TP/PP / VLM) | 大概率不支持 | — | 必需 |

**硬件路径**：Mac 主开发到 M7；M5 benchmark 与 M8 起按需租用云 GPU（Colab Pro / AutoDL / RunPod / Lambda 等），不建议自购卡。

### 2.5.2 四层正确性验证策略

每个函数 / 模块 / 系统按需做这四层验证，对应文件按目录分类：

| 层 | 粒度 | ground truth 来源 | 文件位置 | 适用 |
| --- | --- | --- | --- | --- |
| **L0 单元** | 单函数 / 单算子 | ① 数学闭式定义 / ② `transformers` 同名函数 / ③ 性质断言 | `tests/unit/test_<fn>.py` | rope / rmsnorm / silu / swiglu / attention 等 |
| **L1 数值对齐** | 单模块 forward | `transformers` 同模型 logits：`torch.allclose(my, ref, atol=1e-3)` | `tests/module/test_<mod>_logits.py` | Attention 层、FFN 层、单层 Decoder、整模型 forward |
| **L2 端到端行为** | 整模型 + 调度 | 固定 seed + 贪心解码，断言 token 序列与 `transformers.generate` 完全一致 | `tests/e2e/test_<scene>.py` | 所有里程碑的 smoke test |
| **L3 不变式** | 系统状态 | 内部断言（无外部 ref） | `tests/invariant/test_<sys>.py` | KV 长度、refcount、scheduler 三队列守恒、spec 输出不变 |

#### L0 单元测试 — 三种 ground truth 来源

- **① 数学闭式定义**：`rmsnorm = x * rsqrt(x.pow(2).mean(-1)+eps) * w`、`swiglu = silu(x@Wg) * (x@Wu)`
- **② `transformers` 同名函数**：`apply_rotary_pos_emb`、`repeat_kv`、`scaled_dot_product_attention`
- **③ 性质断言**：`attention(causal=True)` 位置 i 的输出只受位置 ≤ i 影响；`softmax(x).sum(-1) == 1`

**M1 典型 L0 清单**（写 Qwen3 时同步产出）：

| 函数 | ground truth |
| --- | --- |
| `build_rope` / `apply_rope` | transformers `apply_rotary_pos_emb` |
| `rmsnorm` | 闭式公式 |
| `swiglu` | 闭式公式 |
| `repeat_kv` (GQA) | `x.repeat_interleave(n_rep, dim=-3)` |
| `attention(q,k,v,causal)` | `F.scaled_dot_product_attention` + causal 性质 |
| `Qwen3Attention.forward` | transformers 同层 |
| `Qwen3MLP.forward` | transformers 同层 |
| `Qwen3DecoderLayer.forward` | transformers 同层 |

#### L3 关键不变式速查

- KV cache：`cache.shape[len] == prompt_len + step`
- Scheduler：`len(waiting) + len(running) + len(finished) == total_requests`
- Block manager：`sum(refcount) == sum(block_table 中所有引用)`
- Prefix cache：相同前缀两请求 → 前缀部分 block_id 完全一致
- Spec decoding：任何接受率下，最终输出 == 不开 spec 的输出

#### 测试目录与跑法

```
tests/
  conftest.py              # 共享 fixture：tiny_config（4 层/8 head/64 dim）、tiny_qwen3
  unit/                    # L0 — 函数级，秒级
  module/                  # L1 — 模块级数值对齐，~10s
  e2e/                     # L2 — 行为对齐，~分钟
  invariant/               # L3 — 系统不变式
```

```bash
pytest tests/unit                    # 改一个函数就跑
pytest tests/module                  # 改一个模块就跑
pytest tests -m "not slow"           # 默认（跳过加载真实大模型的）
pytest tests                         # 全量
```

**工程纪律**：
1. **TDD-lite**：写核心数学函数前，先写 5–10 行 ground truth 调用，再写实现去对齐
2. **fixture 共享**：`conftest.py` 造 tiny 模型，所有测试复用
3. **slow 标签**：加载真实 Qwen3-0.6B 的测试打 `@pytest.mark.slow`，平时跳过

**CI 约定**：M1–M4 不上 CI（本地 pytest 即可）；**M5 引入最小 GitHub Actions**，跑 CPU-only 的 `unit + module(tiny) + invariant`；e2e/slow 本地手跑。README 加 CI badge。

### 2.5.3 性能评测三件套（M5 一次性建好，后续 M 复用）

**核心指标**（OpenAI 业界标准）：

| 指标 | 含义 | 用途 |
| --- | --- | --- |
| **TTFT** | Time To First Token (ms) | 用户体感 |
| **ITL / TPOT** | Inter-Token Latency (ms) | 流式速度 |
| **Throughput** | 聚合 tok/s | 服务器成本 |
| Mem | GPU 显存峰值 (GB) | M4 之后必看 |
| p50/p95/p99 | 请求级 latency 分布 | 看长尾 |

**Benchmark 三件套**：

1. **内置 micro-bench**：`inferlite bench --model qwen3-0.6b --prompt-len 256 --gen-len 128 --concurrency 1,4,8,16`
2. **对照 transformers baseline**：证明教学版相对朴素实现有 N 倍提升（文章爽点）
3. **对照 vLLM upper bound**：证明能达官方 30–50%

**标准 prompt 集**：

| 数据集 | 用途 | 规模 | 为什么 |
| --- | --- | --- | --- |
| **ShareGPT-100**（主） | 性能 benchmark 主战场 | 100 条真实对话 | 长度方差大，才能测出 continuous batching / PagedAttention / prefix cache 的真实收益；vLLM/SGLang 论文事实标准，数字可横向对比 |
| **GSM8K-20**（选） | M5 reasoning 模式验证 | 20 条数学题 | 长 CoT 输出，暴露 reasoning 模式下 KV cache 容量、SSE `reasoning_content` 分流、长生成 ITL 稳定性 |
| ~~MMLU~~（**不用**） | — | — | 输出只有 1 token，调度器/PagedAttention 无用武之地，等于"拉力测试评 F1" |

**固定 `--seed 42`** 保证可复现。

**评测一致性五原则**：
1. 同 prompt 集 + 同 batch + 同硬件 + 同精度（bf16）
2. Warmup 一轮后再统计
3. 三家精度必须一致（避免 fp32 vs bf16 对比）
4. 跑 bench 时机器静默（不开 Chrome / 不切训练任务）
5. 数据落 `bench/results/<date>-<hardware>.md`，每次跑都归档

**M5 输出的标准对照表**（文章用）：

```
模型：Qwen3-0.6B  硬件：A10 24GB  精度：bf16  prompt：ShareGPT-100  seed=42

| 框架           | conc=1 TTFT | conc=1 ITL | conc=8 TTFT | conc=8 Throughput | Mem |
| transformers  | 120 ms      | 25 ms      | 980 ms      | 60 tok/s          | 5.2 GB |
| inferlite     | 95 ms       | 12 ms      | 110 ms      | 380 tok/s         | 18.5 GB |
| vLLM          | 80 ms       | 8 ms       | 92 ms       | 920 tok/s         | 21.3 GB |
```

文章一句话总结模板：「inferlite 用 vLLM **1/N** 的代码量，拿到 vLLM **X%** 的性能、比 transformers 快 **Y×**。」

<!-- anchor:milestones-core -->
## 3. 核心里程碑 M1–M5（"demo 跑起来"）

每个里程碑给出：**完成定义** / **关键概念** / **必读资料** / **配套文章**。
不给时间，按你节奏推进；每完成一个，仓库打 tag `v0.<N>`。

### M1 Phase 1 — Qwen3 数值对齐：先把 L1 打穿
- **L 层覆盖**：L1 Model 主战场

- **完成定义**：
  1. `RMSNorm / SwiGLU / RoPE / Attention / WeightMap / Qwen3Model` 手写完成
  2. `tests/unit/` 8 个 L0 单元测试全绿
  3. `tests/module/test_qwen3_logits.py` 与当前 lock 的 `transformers==5.10.2` logits `allclose(atol=1e-3, rtol=1e-3)`
- **显式不做**：CLI、Engine、KV cache、Scheduler、Server、文章定稿
- **硬件**：Mac MPS 主开发；必要时 CPU 跑小 shape debug
- **关键概念**：HF safetensors 加载、GQA、RoPE、QK-norm、RMSNorm、SwiGLU、tie embedding
- **必读**：
  - `transformers` 当前 lock 版本的 `models/qwen3/modeling_qwen3.py`
  - Sebastian Raschka《Qwen3 from scratch》notebook
  - Qwen3 技术报告 arXiv:2505.09388 §3 Architecture
- **本质题**：为什么“形状对”不等于“数值对”？

### M1·P2 — 单序列前向：Engine + CLI 能出字
- **L 层覆盖**：L1 主 + L3 最小 Engine 骨架

- **完成定义**：
  1. `python -m inferlite.cli "讲个笑话" --max-new-tokens 32` 在 Mac MPS 上能输出连续文本，无 KV cache
  2. CLI 通过 `LLMModel` 接口调用，不直接 `Qwen3()`
  3. `Sampler` 与 `EngineCore.step()` 最小契约落地；`KVCache / Scheduler / Executor / Drafter` 仅占位，不承诺签名稳定
- **验证标准**：`tests/e2e/test_greedy.py` 前 16 个 token 与 HF `generate(do_sample=False)` 完全一致 + `tests/invariant/test_step_three_phase.py` 覆盖三段式
- **硬件**：Mac MPS 主开发
- **关键概念**：贪心解码循环、三段式 step、接口隔离、为什么 M1 不做 KV cache
- **必读**：
  - Attention Is All You Need
  - Llama 2 paper §2（架构基线）
  - Qwen2 / Qwen3 技术报告（GQA group 数、RoPE θ 差异）
- **本质题**：RoPE 为什么是"旋转"而不是"加法"？
- **配套文章**：《从零手写 Qwen3 —— 解剖一个现代 decoder》

### M2 — KV Cache：从 O(n²) 到 O(n)
- **L 层覆盖**：L2主（`ContiguousKVCache`） + L1（接口接 KV）

- **完成定义**：单序列吞吐相对 M1 提升 ≥ 5×，prefill/decode 两阶段清晰；落地 `KVCache` Protocol 的第一个实现 `ContiguousKVCache`
- **验证标准**：`tests/module/test_kv_cache_logits.py`（带 cache vs 不带 cache logits `allclose`）+ `tests/invariant/test_kv_invariant.py`（cache 长度守恒）
- **硬件**：Mac MPS
- **L 层覆盖**：L1 + L2
- **关键概念**：prefill vs decode、`past_key_values` 张量布局 `[layers, 2, batch, kv_heads, len, head_dim]`、为什么 Q 不 cache
- **必读**：GPT-3 §2.1、MQA paper、GQA paper
- **本质题**：为什么 K 和 V 都要 cache，而 Q 不用？
- **配套文章**：《为什么只 cache K 和 V —— KV 缓存的本质》

### M3 — Continuous Batching：调度器的诞生
- **L 层覆盖**：L3主（`FCFSScheduler` + 三队列）

- **完成定义**：8 并发请求每 step 重新组 batch，无 head-of-line blocking；serial vs batch 输出 token 级等价
- **验证标准**：`test_scheduler_invariant.py`（三队列守恒、EOS 立即出队）+ `test_batch_e2e.py`（8 并发结果与单条串行 token 级 `torch.equal`）
- **硬件**：Mac MPS 可跑功能，性能数最好上 GPU
- **L 层覆盖**：L3
- **关键概念**：`waiting / running / finished` 三队列、变长 attention mask、EOS 立即出队、新请求立即入队
- **性能预期（重要）**：纯 PyTorch 教学版在 MPS 上 batch_generate 比 M2 serial **慢**（实测 0.38~0.44x）。主因是 for 循环写 cache（63%）+ fancy index gather（22%）+ `.item()` 同步（15%），是 batched decode 的固有 Python 开销，nano-vllm 同样有。**性能收益要到 M4（部分缓解）和 M8 Triton kernel（彻底解决）才体现**。M3 的价值是 continuous batching 语义（请求进退 + slot 复用），不是性能。
- **必读**：
  - Orca paper（OSDI'22） — 必读
  - vLLM paper §3（调度部分）
- **本质题**：static batching 的"木桶效应"在 decode 阶段为何尤其致命？
- **对照源码**：`nano-vllm/nanovllm/engine/scheduler.py`
- **配套文章**：《Orca 那篇论文做了什么 —— LLM 调度器的诞生》

### M4 — PagedAttention（PyTorch 伪版）：显存当虚拟内存
- **L 层覆盖**：L2主（`PagedKVCache` block table） + L3（scheduler 意识到 block）

- **完成定义**：长 prompt + 多并发显存利用率 ≥ 80%（GPU 上测），按 block 取 KV，支持 Copy-on-Write fork
- **验证标准**：`test_paged_logits.py`（分页前后 logits `allclose`）+ `test_block_invariant.py`（refcount 守恒、CoW 正确性）
- **硬件**：功能在 Mac 可验；显存利用率指标必须 GPU
- **L 层覆盖**：L2（核心）
- **关键概念**：物理 block table、逻辑 block table、引用计数、CoW、为什么不需要 swap
- **关键裁剪**：**只写 PyTorch `index_select` 伪版**，Triton kernel 推迟到 M8
- **前置关系**：M4 是 **M5·P2 Prefix Cache 的硬前置**（prefix cache 依赖 block 级管理才能复用前缀 KV）。M5·P1（API+SSE）不依赖 M4，可先做 P1 再补 M4。
- **必读**：
  - vLLM paper（SOSP'23 全文，特别是 §4） — 必读
  - FlashAttention v2（对比 SRAM tiling vs HBM 分块）
- **本质题**：PagedAttention 和 OS paging 唯一不同的一点是什么？
- **配套文章**：《把显存当虚拟内存用 —— PagedAttention 的设计精髓》

### M5 — 服务化收口（API + Prefix Cache + Benchmark）

> M5 拆分为三个 Phase 顺序推进，但作为一个 milestone 统一管理（与 M1 同思路）：
> - **Phase 1（API + SSE）**：把 EngineCore 包成可调用服务
> - **Phase 2（Prefix + Reasoning）**：复用能力，TTFT 优化
> - **Phase 3（Benchmark + CI）**：v1 收官 + 对照表

- **L 层覆盖**：L2 + L3 + L4 全部覆盖
- **里程碑标志**：仓库打 `v1.0` tag，代码量 ≈ 2000 行
- **演示价值**：**高**（P1 流式输出可做 demo 视频，curl 看到流式 token 很直观）
- **配套文章**：《2000 行实现一个 vLLM —— inferlite v1 总览》（**核心宣传文**）

#### Phase 1 — 服务化 demo：采样 + OpenAI API + SSE
- **完成定义**：
  1. `pip install -e .` + `inferlite serve qwen3-0.6b` 起服务
  2. `curl http://localhost:8000/v1/chat/completions` 兼容 OpenAI（含 SSE）
  3. 支持 greedy / temperature / top-k / top-p / repetition penalty
- **验证标准**：`test_openai_api.py` 覆盖 curl 兼容性 + SSE 流式格式；sampler 单测覆盖 seed 可复现
- **硬件**：Mac MPS 主开发
- **关键概念**：logits processor 链、OpenAI 流式协议、请求参数到 sampler 的映射

#### Phase 2 — 复用能力：前缀缓存 + Reasoning 字段分流
- **完成定义**：
  1. 多轮对话第二轮 TTFT 比第一轮 ↓ 5×（prefix cache 命中）
  2. thinking 模式下 `reasoning_content` 字段分流
  3. 前缀 block hash / refcount / eviction 策略可解释
- **验证标准**：`test_prefix_invariant.py` 覆盖同前缀两请求 block_id 一致；reasoning SSE 格式有 e2e 测试
- **硬件**：Mac 可验功能；性能收益最好上 GPU 复测
- **关键概念**：RadixTree-Lite + block hash、prefix cache 命中率、reasoning 字段约定

#### Phase 3 — Benchmark + CI：核心 v1 收口
- **完成定义**：
  1. Benchmark 表：inferlite vs transformers.generate vs vLLM 三栏 throughput / TTFT / ITL（见 §2.5.3 标准模板）
  2. `bench/run_all.sh` 一键产出对照表，归档到 `bench/results/`
  3. 最小 CI 上线：`.github/workflows/ci.yml`，推送自动跑 CPU-only 的 L2+L3，依赖 tiny fake 模型，README 加绿勾 badge
- **硬件**：benchmark 必须 GPU；CI 跑在 GitHub 免费 Ubuntu runner
- **关键概念**：TTFT / ITL / throughput、warmup、seed 固定、硬件与精度一致性
- **必读**：
  - SGLang paper（RadixAttention）
  - The Curious Case of Neural Text Degeneration（top-p 起源）
  - OpenAI Chat Completions API spec
- **本质题**：prefix cache 命中率在生产里被什么决定？

<!-- anchor:milestones-extension -->
## 4. 扩充里程碑 M6+（同仓库长期迭代，无截止）

> M5 之后所有新能力都作为新 M 并入主仓库。排序按三个维度权衡：
> - **教学价值**：该概念对理解 LLM 推理引擎的核心程度
> - **硬件可用性**：Mac MPS 能做 vs 必须 NVIDIA GPU
> - **演示价值**：是否适合做 demo / 文章（流式、spec、VLM 演示效果好）

**排序逻辑**：
- M6 MoE / M7 Spec 是"能力扩充"（新模型/新算法），Mac 可做，先做
- M8 Triton / M9 grouped GEMM 是"性能优化"，必须 GPU，Mac 用户可跳过
- M10 EAGLE / M11 Chunked / M12 Long context 是"能力升级"，Mac 可做
- M13 / M14 VLM 是"场景扩充"，演示价值高，Mac 可做

**Mac 用户的推荐路径**：M6 → M7 → M10 → M11 → M12 → M13 → M14（跳过 M8/M9）
**GPU 用户的推荐路径**：按 M6-M14 顺序，或按兴趣跳

### M6 — MoE 教学版（for-loop）
- **L 层覆盖**：L1主（二路模型架构） + Registry 首实现
- **完成定义**：跑通 Qwen1.5-MoE-A2.7B 或 Qwen3-30B-A3B
- **验证**：L1 logits 对齐 `transformers` MoE 实现
- **硬件**：A2.7B 需 GPU（或 Mac 32GB+）
- **关键概念**：router top-k softmax、per-expert for-loop dispatch、`[num_experts, hidden, ffn]` 权重布局
- **裁剪**：不做 grouped GEMM、不做 EP（留 M9）
- **配套文章**：《MoE 推理：从 router 到 dispatch》

### M7 — Speculative Decoding（n-gram lookup）
- **L 层覆盖**：L3主（`Drafter` Plugin + step() B 点 hook）
- **完成定义**：长 prompt 续写场景 ≥ 1.5× 加速
- **验证**：L3 不变式 —— 任何接受率下最终输出 == 不开 spec 的输出
- **硬件**：Mac 即可
- **演示价值**：**高**（有/无 spec 的速度对比很直观）
- **关键概念**：n-gram 查表当 draft、verify 阶段一次性算 K+1 logits、接受率统计
- **裁剪**：不训练 draft 模型，EAGLE 留 M10
- **配套文章**：《零训练 spec decoding：n-gram lookup 也能涨吞吐》

### M8 — Triton PagedAttention kernel
- **L 层覆盖**：L1↔L2 边界（kernel 级，不动上层）
- **完成定义**：替换 M4 的 PyTorch 伪版，性能逼近 vLLM 官方 kernel 的 50%
- **验证**：L1 与 PyTorch 伪版 logits 完全一致；L3 与 vLLM kernel 抽样对照
- **硬件**：**必须 NVIDIA GPU**（Triton 不支持 MPS）
- **关键概念**：Triton block 编程模型、warp / tile 设计、HBM ↔ SRAM 数据流
- **Mac 用户替代路径**：M8 需要 NVIDIA GPU，Mac 用户做完 M7 后可选：
  - **方案 A**：跳过 M8，转做 M10（EAGLE）/ M11（Chunked Prefill）/ M12（Long context）/ M13（VLM），这些不需要 GPU
  - **方案 B**：M8-lite —— 用 `torch.compile` + `flex_attention` 做 Mac 友好的半优化版本，性能介于 PyTorch 伪版和 Triton 之间（教学价值有限，仅作过渡）
  - **方案 C**：等有 GPU 环境再回头做 M8
- **配套文章**：《手写第一个 Triton kernel —— PagedAttention 内部》

### M9 — MoE 升级（grouped GEMM）
- **L 层覆盖**：L1主
- **完成定义**：M6 升级版，30B-A3B 吞吐 ≥ 5 tok/s
- **关键概念**：token permutation、grouped GEMM、负载均衡观测

### M10 — Speculative Decoding 升级（EAGLE-1 简化版）
- **L 层覆盖**：L1（draft head） + L3主（verify/rollback）
- **完成定义**：训一个 draft head，加速 ≥ 3×
- **关键概念**：draft head 训练、tree verify、接受路径回滚

### M11 — Chunked Prefill
- **L 层覆盖**：L3主（`ChunkedPrefillScheduler`）
- **完成定义**：超长 prompt（≥ 32K）分块预填，避免 prefill 阶段 OOM 或长尾阻塞
- **关键概念**：prefill 切片、和 decode mix-batching 的优先级
- **顺序理由**：长上下文先要"喂得进"再谈"算得对"，调度层是前置依赖

### M12 — Long context（YaRN / NTK RoPE 缩放）
- **L 层覆盖**：L1主（RoPE 重映射） + L2（KV 长度上限）
- **完成定义**：Qwen3-0.6B 上下文从 32K 推到 128K
- **关键概念**：RoPE 频率重映射、外推 vs 内插

### M13 — VLM 教学版（图→文）
- **L 层覆盖**：L1主（vision encoder + `inputs_embeds`） + L4（multipart 请求）
- **完成定义**：接入一个小型 VLM（候选 Qwen-VL / Llava / native multimodal 小模型，M13 前再定），单图 + 文本对话能跑通
- **演示价值**：**高**（图文对话吸睛，适合做 demo）
- **选型原则**：优先选权重易下载、Mac/GPU 显存压力小、语言侧尽量复用 M1–M12 架构的模型；不要现在绑定具体未验证模型名
- **关键概念**：Vision encoder（ViT/SigLIP）独立前向、image token embedding 注入（走 `inputs_embeds`）、变长 image token 数处理
- **裁剪**：不做 image prefix cache、不做 encoder/LLM 异步、单 batch 单图（留 M14）
- **配套文章**：《推理框架怎么吃下一张图 —— VLM 接入解剖》

### M14 — VLM 工程化
- **L 层覆盖**：L2主（image hash prefix cache） + L3（encoder/LLM 异步）
- **完成定义**：多分辨率 + image content hash prefix cache + encoder/LLM 异步流水线
- **关键概念**：图片内容哈希、EncoderCache、多模态 + 文本 batch 混排

### M15+ — 持续扩充候选池（无截止）

**核心计算 / 显存**：
- **量化**：GPTQ / AWQ / FP8 权重加载 + 自定义 matmul
- **KV cache 量化**（FP8）：显存翻倍
- **MLA attention**：DeepSeek-V2/V3 风格，KV cache 结构重写
- **Hybrid SSM**（Jamba/Mamba2）：非 KV 范式

**调度 / 解码**：
- **CUDA Graph 捕获**：降低 kernel launch 开销
- **Preemption 升级**：swap-out vs recompute 切换策略
- **Async Scheduling**：调度和 forward 重叠（vLLM 仍实验中）
- **Guided Decoding（FSM/grammar bitmask）**：JSON / 正则约束输出，xgrammar 后端
- **LogProbs 完整支持**：调试 / 评测必备

**并行**：
- **Tensor Parallel**：单机多卡，切 head
- **Pipeline Parallel**：跨机切 layer
- **Expert Parallel**：MoE 专家分卡

**Serving**：
- **Disaggregated P/D**：prefill / decode 拆机器，KV 通过 connector 转移
- **Prometheus `/metrics`**：QPS、p99、KV 利用率
- **LoRA 多 adapter 共服**：一个底模 + N 个 LoRA

**多模态扩展**：
- **Audio 输入**（Whisper / Qwen-Audio）：encoder 接入 + 长音频分块

**暂不规划**（成本/教学价值不划算）：
- Omni 全双工（流式 audio in/out + 打断检测，工业框架都还没成熟）
- Beam Search（少用）
- Encoder-Decoder（T5/Whisper-T5 范式）
- Embedding / Reranker 服务（不属于推理引擎核心）
- Diffusion LLM（LLaDA / Mercury）
- Data Parallel（直接起多个 inferlite 实例 + 外层 LB 即可）

**节奏**：每选 1 个就开 1 个 M，文章 + 代码 + Review 闭环。

<!-- anchor:model-matrix -->
## 5. 模型类型矩阵与归属里程碑

| # | 类型 | 代表 | 改动量 | 归属 |
| --- | --- | --- | --- | --- |
| 1 | Dense | Qwen3-0.6B | 基线 | M1–M5 |
| 2 | Reasoning | Qwen3-thinking, DeepSeek-R1 | 极小 | M5 |
| 3 | MoE | Qwen3-30B-A3B, Mixtral | 中 | M6 / M9 |
| 4 | Speculative decoding | n-gram, EAGLE, Medusa | 中 | M7 / M10 |
| 5 | Long context (YaRN) | Qwen3-1M | 小 | M11 |
| 6 | Quantized | GPTQ/AWQ/FP8 | 中 | M15+ |
| 7 | MLA attention | DeepSeek-V2/V3 | 中 | M15+ |
| 8 | Hybrid SSM | Jamba, Mamba2 | 大 | M15+ |
| 9 | Multimodal (VLM 图→文) | Qwen-VL / Llava / native multimodal 小模型（M13 前再定） | 中（加 vision encoder + `inputs_embeds`） | M13 / M14 |
| 10 | Audio 输入 | Whisper, Qwen-Audio | 中 | M15+ |
| 11 | Omni 全双工 | Qwen2.5-Omni, GPT-4o 类 | 大（流式 in/out + 打断） | 暂不规划 |
| 12 | Diffusion LLM | LLaDA, Mercury | 极大 | 暂不规划 |

<!-- anchor:collab -->
## 6. Agent / 你 分工 + 文章工作流

### 6.1 分工边界

| 工作 | 谁来做 |
| --- | --- |
| 里程碑定义、范围裁剪、依赖分析 | Agent |
| 论文导读、源码对照、概念讲解、伪代码、最小复现片段 | Agent |
| **写 `inferlite/*.py` 任何业务代码** | **你手敲**（Agent 不主动改） |
| 卡壳时给"提示而非答案" | Agent（先给思路，再要原理，最后才给代码片段） |
| 代码 Review（结构、命名、复杂度、文档字符串） | Agent |
| 文章初稿（基于你的代码 + 笔记 + 卡片） | Agent 起草，你定稿 |
| 知乎发布（`to_zhihu_md.py` / `check_zhihu_format.py`） | 你 |

**硬约束**：Agent 不主动修改 `inferlite/*.py`，除非你显式说"帮我把这段写一下"。

### 6.2 单个里程碑的闭环

```
M<N> 启动
   ↓
Agent 给"里程碑 brief"：完成定义 + 关键概念清单 + 必读资料 + 文件骨架建议
   ↓
你手敲代码（可随时问 Agent 原理 / 卡壳 / 设计选择）
   ↓
你跑通 → push → 仓库打 tag v0.<N>
   ↓
Agent Review：结构、命名、注释、是否偏离主线
   ↓
你写"开工 + 收工"笔记草稿（50–200 字关键洞察）
   ↓
Agent 扩成完整原理篇（卡片化、对照源码、引用论文）
   ↓
渲染 + 转知乎
   ↓
你发布 → PROGRESS.md 更新
```

### 6.3 文章结构模板（每篇 M 文章固定章节）

1. **一句话本质**（开头 50 字内）
2. **从一个问题出发**（为什么需要这个能力 / 不做会怎样）
3. **原理**：论文核心思想 + 类比 + 公式（必要时）
4. **代码精讲**：贴你手敲的关键片段，逐行解释设计意图
5. **对照实现**：和 vLLM / nano-vllm 的差异，说明为什么这样简化
6. **Benchmark / 现象**：数字 + 截图
7. **本质题 + 一句话压缩**（呼应开头）

<!-- anchor:risks -->
## 7. 风险与应对

| 风险 | 触发 | 应对 |
| --- | --- | --- |
| **动力中断** | 中间某周事多没推进 | 里程碑越小越好；用 `docs/journal/` 周记保持手感 |
| **Agent 越界写代码** | Agent 主动改 `inferlite/*.py` | 你随时打断，要求改成"给思路 + 伪代码" |
| **过早优化** | 在 M4 死磕 Triton 性能 | 严守"M4 只 PyTorch 伪版，性能留 M8"原则 |
| **范围蔓延** | 想顺手加量化 / VLM | 一律新开 M，不并入 M1–M5 |
| **没 GPU 跑大模型** | M6 之后需要更大显存 | 按需租用云 GPU（Colab Pro / AutoDL / RunPod 等） |
| **文章烂尾** | 代码跑通了但没发文 | 文章发布作为里程碑硬性闭环，未发文不能开下一个 M |

<!-- anchor:next-action -->
## 8. 立刻可做的下一步

1. **建项目骨架**：`pyproject.toml` + `inferlite/{model,cache,scheduler,sampler,executor,server}/__init__.py` + 7 个 Protocol 空文件 + `tests/{unit,module,e2e,invariant}/`
2. **开 M1**：Agent 给 M1 brief（完成定义 + 关键概念清单 + 必读 + 文件骨架建议），你手敲 `model/qwen3.py` 起步

> 工作流约定：每开一个新 M，你说一声"开 M<N>"；Agent 不主动推进。
