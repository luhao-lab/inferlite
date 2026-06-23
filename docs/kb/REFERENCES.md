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
- **对 inferlite 的价值**: **M1 Phase 1（数值对齐）的最佳教学材料**。RMSNorm / RoPE / GQA 每个都有"公式 → 代码 → 测试"三栏对照
- **怎么读**: 写一个模块前，把 Qwen3 notebook 对应 cell 跑一遍，确认 ground truth

### 3. huggingface/transformers（仅 Qwen3 部分）

- **GitHub**: https://github.com/huggingface/transformers
- **固定参考**: https://github.com/huggingface/transformers/blob/0dad7b822255a0ae261ec45ae937371e859ffd1a/src/transformers/models/qwen3/modeling_qwen3.py
- **文件**: `src/transformers/models/qwen3/modeling_qwen3.py`（~600 行）
- **对 inferlite 的价值**: **L1 数值对齐的唯一 ground truth**
- **怎么读**: 不读全文，写每个模块时打开搜对应类
  - `Qwen3RMSNorm` → 写 T1 RMSNorm 时对照
  - `Qwen3MLP` → 写 T2 SwiGLU 时对照
  - `Qwen3RotaryEmbedding` + `apply_rotary_pos_emb` → 写 T3 RoPE
  - `Qwen3Attention` → 写 T4 GQA
  - `Qwen3DecoderLayer`、`Qwen3Model` → 写 T5/T6
- **定位**: 数学真值，不负责推理系统架构。

---

## 🥈 第二梯队：架构参考（M3+ 阶段）

### 4. vllm-project/vllm

- **GitHub**: https://github.com/vllm-project/vllm
- **Qwen3 文件**: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py
- **体量**: 生产级 ~10 万行 Python+CUDA
- **对 inferlite 的价值**: **架构师视角的"成品答案"**。三段式 EngineCore、PagedAttention、Continuous Batching、Qwen3 权重加载的工业实现
- **危险**: 直接读会被淹死
- **怎么读**: M1/M2 阶段只挑 4 个入口读
  - `vllm/model_executor/models/qwen3.py::Qwen3Attention` — 看 QK-norm、RoPE、GQA 在推理框架里的组织（T4 辅助参考）
  - `vllm/model_executor/models/qwen3.py::Qwen3ForCausalLM.load_weights` — 看 tied embedding / packed mapping（T7 必看）
  - `vllm/v1/engine/core.py::step()` — 看三段式不变量（M1·P2 借鉴）
  - `vllm/v1/core/sched/scheduler.py` FCFS 部分 — 看 M3 Continuous Batching
- **暂不看**: TP/PP/spec decoding/CUDA kernel（M9+ 才回来）
- **定位**: 推理工程参考，不作为数值 reference。

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

### 10. QwenLM/Qwen3

- **GitHub**: https://github.com/QwenLM/Qwen3
- **阶段**: 全阶段按需查
- **价值**: 官方使用说明、deployment 推荐、thinking/no-thinking、chat template 与 tokenizer 行为说明
- **定位**: 官方产品/使用参考；不是 M1 数值对齐源码主参考。

### 11. ggml-org/llama.cpp

- **GitHub**: https://github.com/ggml-org/llama.cpp
- **阶段**: M5+ 本地推理/量化/Metal/CPU 路线
- **价值**: GGUF 权重格式、量化、本地 CPU/Metal 推理、非 Python 后端组织方式
- **定位**: 后端与量化参考；M1 不深读，避免 C++/GGML graph 干扰 PyTorch 数值对齐。

### 12. sgl-project/sglang

- **GitHub**: https://github.com/sgl-project/sglang
- **阶段**: M5+ serving / prefix cache / runtime
- **价值**: 高吞吐 serving、prefix cache、speculative decoding、reasoning parser 的系统参考
- **定位**: 后续 serving/runtime 参考；M1 不读。

### 13. huggingface/text-generation-inference

- **GitHub**: https://github.com/huggingface/text-generation-inference
- **阶段**: M5+ serving
- **价值**: HF 生产级 router/server/batching 设计
- **定位**: serving 架构参考；不作为 Qwen3 算子 ground truth。

### 14. NVIDIA/TensorRT-LLM

- **GitHub**: https://github.com/NVIDIA/TensorRT-LLM
- **阶段**: M8+ kernel / 高性能 GPU 推理
- **价值**: NVIDIA GPU 高性能实现、fused op、engine build、plugin 思路
- **定位**: 高性能后端参考；M1 不读源码。

### 15. mlc-ai/mlc-llm

- **GitHub**: https://github.com/mlc-ai/mlc-llm
- **阶段**: 架构对照
- **价值**: TVM + Apache Relax "编译派"代表，与 PyTorch eager 派对照
- **学习**: 看 README 即可

### 16. SafeAILab/EAGLE

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
M1·P1 阶段（数值对齐）:
  ✅ 必看: rasbt/LLMs-from-scratch Qwen3 notebook（写模块前精读对应 cell）
  ✅ 必看: transformers/modeling_qwen3.py（数学真值；写模块前搜对应类）
  ✅ T4 辅助: vllm/model_executor/models/qwen3.py::Qwen3Attention（只看工程组织差异）
  ✅ T7 必看: vllm/model_executor/models/qwen3.py::load_weights（只看权重加载，不泛读 engine）
  ✅ 官方说明: QwenLM/Qwen3（只查 tokenizer/chat template/thinking/deployment 说明）
  ✅ 论文: Qwen3 报告 §3 / RoFormer §3.4.2 / GQA §2
  ⏸ 暂不看: llama.cpp / SGLang / TGI / TensorRT-LLM / vllm scheduler/paged-attention

M1·P2 阶段（出字闭环）:
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
  ✅ llama.cpp（GGUF/量化/本地 Metal/CPU）
  ✅ SGLang / TGI（serving / prefix cache / router / batching）
  ✅ TensorRT-LLM（NVIDIA 高性能后端）
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

---

## 🧭 AI 协作方法论 / Vibe Coding 工作流

> 与第一/二/三梯队不同：这里参考的不是 LLM 推理代码，而是"AI 协作开发本身"的方法论与工具。
> 对 inferlite 这种"作者手撕 + AI 辅助 plan/review/doc"的项目尤其相关。

### A1. github/spec-kit（业界事实标准）

- **GitHub**: https://github.com/github/spec-kit
- **核心**: Spec-Driven Development 工具包，标准化 `/specify → /plan → /tasks → /implement` 四阶段
- **对 inferlite 借鉴**:
  - 任务卡 7 字段模板（前置/产出/算法/测试/DoD/坑/估时）
  - `plan.md` ↔ `tasks.md` 双向链接
  - `constitution.md`（项目原则，等价于本仓库的 `AGENTS.md`）
- **不直接迁移的理由**: docs/ 结构已九成像 spec-kit，重组 ROI 低

### A2. Anthropic — Claude Code 官方最佳实践

- **链接**: https://www.anthropic.com/news/claude-code-best-practices
- **核心档**: `CLAUDE.md`（项目常驻记忆）+ `.claude/commands/`（自定义 slash 命令）+ Plan Mode
- **对 inferlite 借鉴**: 已落地（见 `.claude/commands/` 目录）

### A3. OpenSpec（change-set 风格）

- **GitHub**: https://github.com/Fission-AI/OpenSpec
- **核心**: 每次变更一个 spec → 实现 → archive 四态机
- **对 inferlite 借鉴**: M1.md 任务卡天然就是 change-set；可学其状态标注

### A4. Addy Osmani — "My LLM Coding Workflow Going Into 2026"

- **链接**: https://medium.com/@addyosmani/my-llm-coding-workflow-going-into-2026-52fe1681325e
- **核心论点**:
  - "Vibe coding" ≠ "AI-assisted engineering"
  - 小步 commit + 好 message 本身就是开发过程文档
  - 对 AI 提"约束"比提"功能"更重要

### A5. Addy Osmani — "Vibe coding vs AI-Assisted Engineering"

- **链接**: https://medium.com/@addyosmani/vibe-coding-is-not-the-same-as-ai-assisted-engineering-3f81088d5b98
- **划线**: "无脑跟着 AI 写" vs "把 AI 嵌入完整 SDLC"

### A6. Cursor/Cline/Windsurf Rules 仓库

- **代表**: https://github.com/PatrickJS/awesome-cursorrules
- **对 inferlite 借鉴**: `AGENTS.md` / `CLAUDE.md` 措辞素材

### A7. Aider（极简 diff 驱动）

- **GitHub**: https://github.com/Aider-AI/aider
- **核心**: edit → diff → confirm 三步循环，git diff 作为唯一交互界面

### A8. 配套阅读

| 资料 | 用途 |
| --- | --- |
| [Simon Willison — Coding with LLMs in Late 2025](https://simonwillison.net/2025/Oct/27/coding-with-llms/) | 综述含 prompt 模式 |
| [Sourcegraph — Spec-First with AI](https://sourcegraph.com) | spec 文件结构推荐 |
| [Karpathy — LLM as kernel of OS](https://karpathy.ai/) | "AI 是新 runtime"视角 |

---

## 🎯 inferlite 已落地的 AI 协作纪律

| 借鉴源 | 本仓库实现 |
| --- | --- |
| spec-kit tasks 模板 | `docs/plan/M1.md` §4 任务卡总表 + `docs/tasks/T*.md` 每卡独立 |
| CLAUDE.md 常驻记忆 | `~/learning/AGENTS.md`（仓库根） |
| Claude Code custom commands | `.claude/commands/`（plan / work / review / archive / preflight） |
| OpenSpec 状态机 | 任务卡 status 列（⬜ pending / 🟡 in-progress / ✅ done） |
| Addy Osmani 小步 commit | 当前 commit 历史已遵循（bddf42e → 6163a14 一连串小提交） |
| Plan Mode（先方案再执行） | 每张任务卡先发卡 → 用户实现 → review → commit |

### 元洞察（来自 T1 RMSNorm 复盘）

> **"地基"和"算法"是两个频道，不要混着切。**
>
> 这就是 spec-driven 的本质：把"做什么"与"怎么做"分离到两个时间窗口。
> AI 协作时尤其重要——AI 上下文有限，频道切换成本高。
