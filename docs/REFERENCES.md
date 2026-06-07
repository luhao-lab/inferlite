# REFERENCES — LLM 推理框架参考项目调研

> 学习 / 实现 inferlite 时可对照的开源项目、论文、博客清单。
> 按"对 inferlite 当前阶段的价值"排序，**不是评判项目本身好坏**。
>
> 阅读纪律：避免一次读全部。每张任务卡（M1.md §4）开工前查本文档对应阶段，**只读当前阶段必看的 1-2 个**。

---

## 🥇 第一梯队：必看必抄（M1 阶段）

### 1. GeeeekExplorer/nano-vllm

- **GitHub**: https://github.com/GeeeekExplorer/nano-vllm
- **体量**: ~1200 行 Python，单 GPU，纯 PyTorch
- **覆盖**: prefix cache + tensor parallelism + CUDA graph + 简化 PagedAttention
- **支持模型**: Qwen3 / LLaMA / Mistral
- **对 inferlite 的价值**: **目标体量参照物**。inferlite 千行 Python 原型对标的就是它
- **怎么读**:
  - M1: 只看 `nanovllm/models/qwen3.py`（~200 行，能跑通的最简 Qwen3）
  - M2/M3: 看 `nanovllm/engine/` 和 `nanovllm/cache/`
  - M4: 看 `nanovllm/layers/attention.py`（PagedAttention 简化版）
- **不要做**: 不要一次读完，会被淹

### 2. rasbt/LLMs-from-scratch（Sebastian Raschka）

- **GitHub**: https://github.com/rasbt/LLMs-from-scratch
- **特别看**: `ch05_main_chapter_code/qwen3.ipynb`
- **风格**: 教学最清晰，每个公式 → 代码逐句对照，Jupyter 可逐 cell 跑
- **对 inferlite 的价值**: **M1a 数值对齐的最佳教学材料**。RMSNorm / RoPE / GQA 每个都有"公式 → 代码 → 测试"三栏对照
- **怎么读**: 写一个模块前，把 Qwen3 notebook 对应 cell 跑一遍，确认 ground truth

### 3. huggingface/transformers（仅 Qwen3 部分）

- **GitHub**: https://github.com/huggingface/transformers
- **文件**: `src/transformers/models/qwen3/modeling_qwen3.py`（~600 行）
- **对 inferlite 的价值**: **L1 数值对齐的唯一 ground truth**
- **怎么读**: 不读全文，写每个模块时打开搜对应类
  - `Qwen3RMSNorm` → 写 T1 RMSNorm 时对照
  - `Qwen3MLP` → 写 T2 SwiGLU 时对照
  - `Qwen3RotaryEmbedding` + `apply_rotary_pos_emb` → 写 T3 RoPE
  - `Qwen3Attention` → 写 T4 GQA
  - `Qwen3DecoderLayer`、`Qwen3Model` → 写 T5/T6

---

## 🥈 第二梯队：架构参考（M3+ 阶段）

### 4. vllm-project/vllm

- **GitHub**: https://github.com/vllm-project/vllm
- **体量**: 生产级 ~10 万行 Python+CUDA
- **对 inferlite 的价值**: **架构师视角的"成品答案"**。三段式 EngineCore、PagedAttention、Continuous Batching 工业实现
- **危险**: 直接读会被淹死
- **怎么读**: M1/M2 阶段只挑 3 个文件读
  - `vllm/v1/engine/core.py::step()` — 看三段式不变量（M1b 借鉴）
  - `vllm/model_executor/models/registry.py` — 看 Registry 写法（M6 MoE 时用）
  - `vllm/v1/core/sched/scheduler.py` FCFS 部分 — 看 M3 Continuous Batching
- **暂不看**: TP/PP/spec decoding/CUDA kernel（M9+ 才回来）

### 5. microsoft/MinivLLM（本地已 clone）

- **路径**: `~/learning/MinivLLM/`
- **特点**: 微软出品的极简版 vLLM，~3000 行
- **对 inferlite 的价值**: vLLM 简化路径的另一个样本，对照 nano-vllm 看两种简化哲学
- **已有笔记**: `docs/opensource/minivllm/`

### 6. karpathy/nanoGPT

- **GitHub**: https://github.com/karpathy/nanoGPT
- **风格**: ~600 行，训练+推理一体，**GPT-2 不是 LLaMA 家族**
- **对 inferlite 的价值**: 单文件极简风格的鼻祖
- **危险**: GPT-2 风格（绝对位置编码、bias=True、LayerNorm 而非 RMSNorm），**别误抄到 Qwen3**
- **怎么读**: 只读 `model.py`，欣赏代码密度

### 7. InternLM/lmdeploy

- **GitHub**: https://github.com/InternLM/lmdeploy
- **特点**: 上海 AI Lab 推理框架，TurboMind C++ 后端 + PyTorch 前端
- **对 inferlite 的价值**: vLLM 的对照组（C++ kernel + Python orchestration）
- **怎么读**: 仅看 `lmdeploy/pytorch/` 目录

---

## 🥉 第三梯队：专题精读（M4+ 阶段）

### 8. flashinfer-ai/flashinfer

- **GitHub**: https://github.com/flashinfer-ai/flashinfer
- **阶段**: M8 Triton kernel
- **价值**: PagedAttention / Append Attention 的 SOTA kernel 实现

### 9. Dao-AILab/flash-attention

- **GitHub**: https://github.com/Dao-AILab/flash-attention
- **阶段**: M5 / M8（性能优化背景）
- **价值**: FlashAttention v1/v2/v3 论文 + 实现
- **学习**: 看 README + Tri Dao 的 4 篇博客足够

### 10. NVIDIA/TensorRT-LLM

- **GitHub**: https://github.com/NVIDIA/TensorRT-LLM
- **阶段**: 架构参考，不读源码
- **价值**: 生产级 C++ 推理栈，PluginRegistry 设计

### 11. mlc-ai/mlc-llm

- **GitHub**: https://github.com/mlc-ai/mlc-llm
- **阶段**: 架构对照
- **价值**: TVM + Apache Relax "编译派"代表，与 PyTorch eager 派对照
- **学习**: 看 README 即可

### 12. SafeAILab/EAGLE

- **GitHub**: https://github.com/SafeAILab/EAGLE
- **阶段**: M10 EAGLE 投机解码
- **价值**: EAGLE-1/2/3 官方实现

---

## 📚 论文与博客

| # | 资料 | 形式 | 阶段 | 用途 |
| --- | --- | --- | --- | --- |
| 1 | [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388) | 论文 | M1 | §3 架构细节、超参 |
| 2 | [RoFormer (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) | 论文 | M1 / T3 | RoPE 数学推导 |
| 3 | [GQA (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245) | 论文 | M1 / T4 | 分组查询注意力 |
| 4 | [RMSNorm (Zhang & Sennrich 2019)](https://arxiv.org/abs/1910.07467) | 论文 | M1 / T1 | 归一化层 |
| 5 | [GLU Variants Improve Transformer (Shazeer 2020)](https://arxiv.org/abs/2002.05202) | 论文 | M1 / T2 | SwiGLU 由来 |
| 6 | [FlashAttention (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) | 论文 | M5 / M8 | 性能优化背景 |
| 7 | [vLLM / PagedAttention SOSP'23 (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) | 论文 | M4 | PagedAttention 由来 |
| 8 | [Continuous Batching (Orca, OSDI'22)](https://www.usenix.org/conference/osdi22/presentation/yu) | 论文 | M3 | Continuous Batching 由来 |
| 9 | [EAGLE-1 (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) | 论文 | M10 | Spec Decoding |
| 10 | [Lilian Weng — LLM Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | 博客 | 全阶段 | 优化技术全景综述 |
| 11 | [HuggingFace LLM Inference Course](https://huggingface.co/learn/llm-course) | 课程 | 入门 | 工程视角 |

---

## 🎯 inferlite 阶段-资料对照

```
M1a 阶段（数值对齐）:
  ✅ 必看: rasbt/LLMs-from-scratch Qwen3 notebook（写模块前精读对应 cell）
  ✅ 必看: transformers/modeling_qwen3.py（写模块前搜对应类）
  ✅ 论文: Qwen3 报告 §3 / RoFormer §3.4.2 / GQA §2
  ⏸ 暂不看: nano-vllm engine/, vllm

M1b 阶段（出字闭环）:
  ✅ 看: nano-vllm engine/（理解 step() 三段式）
  ✅ 看: vllm/v1/engine/core.py step()（50 行，看不变量）

M2 阶段（KV cache）:
  ✅ 看: nano-vllm cache 实现
  ✅ 看: vllm ContiguousKVCache（历史版本）

M3 阶段（Continuous Batching）:
  ✅ 看: nano-vllm scheduler
  ✅ 看: vllm/v1/core/sched/scheduler.py FCFS
  ✅ 论文: Orca OSDI'22

M4 阶段（PagedAttention）:
  ✅ 看: vllm PagedAttention 论文 + 简化实现
  ✅ 看: nano-vllm 简化版

M5+ 阶段:
  ✅ FlashAttention 论文
  ✅ Lilian Weng 综述
  ✅ EAGLE 论文（M10）
  ✅ flashinfer（M8 kernel）
```

---

## 💡 反模式预警

### ❌ 不要现在读 vllm 全源码

vllm 是 30+ 人团队 2 年产物，单抽象层数（Engine/Executor/Worker/ModelRunner/Scheduler/SequenceGroupMetadata/...）就足够让初学者迷失方向。

**正确顺序**:
1. 用 nano-vllm 建立"全景肌肉记忆"（先看到鸟瞰图）
2. 用 transformers 建立"数值对齐反射"（每个算子知道 ground truth 在哪）
3. M3+ 带着具体问题去翻 vllm 对应部分（不要泛读）

### ❌ 不要混抄 GPT-2 与 LLaMA 家族细节

nanoGPT 是 GPT-2：绝对位置编码、`nn.LayerNorm`、`bias=True`。
Qwen3 是 LLaMA 家族：RoPE、`RMSNorm`、`bias=False`、QK-norm。
**混抄会得到一个"形状对但 logits 全错"的模型**，调试起来极痛苦。

### ❌ 不要追求"读完一个项目"

每个项目都只读"当前 M 用得上的 1-2 个文件"。读完整个 vllm 的人，大多没把任何东西写出来。

---

## 🔄 维护规范

- 新发现的有价值项目 → 在第三梯队追加，注明"哪个 M 阶段用"
- 已读完的项目 → 在标题后加 ✅ + 写一句"读后感"
- 已沉淀到 `docs/opensource/<project>/` 的项目 → 末尾加链接指过去

---

## 🔗 已有的精读笔记

- `~/learning/docs/opensource/minivllm/`（MinivLLM 源码精读）
- `~/learning/docs/papers/`（按主题分类的论文精读）
