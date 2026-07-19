# inferlite Roadmap — 从零手写 LLM 推理引擎

> 目标：在 `luhao-lab/inferlite` 中手敲一个可读、可跑、可解释的 LLM 推理框架，覆盖 vLLM 核心思想：KV Cache / PagedAttention / Continuous Batching / Prefix Cache，并长期扩展 MoE、Spec Decoding、Triton kernel、长上下文与 VLM。

<!-- anchor:tldr -->
## 0. TL;DR

### 一句话

把“一个 LLM 请求”变成“一个可服务的推理系统”：单请求推理 → 多请求调度 → 分页显存管理 → 前缀复用 → API/SSE 服务化 → 长期能力扩展。

### 当前推荐路径

```text
M1-M2（已完成：单请求 + KV Cache）
    ↓
M3 Continuous Batching（L3 调度地基）
    ├──> M6 API + SSE（L4 服务化；技术上只硬依赖 M3）
    ↓
M4 PagedAttention（L2 block table）
    ↓
M5 Prefix Cache + Reasoning（L2 复用能力；硬依赖 M4）
    ↓
Release Checklist（Benchmark + CI + README + v1.0 tag；不单独算 M）
    ↓
M7+ 能力主题包（MoE 模型支持 / 推测解码加速 / 核心算子加速 / 长上下文 / 多模态）
```

### 四个技术主题

| 主题 | 里程碑 | 解决的问题 |
| --- | --- | --- |
| 多请求调度 | M3 | 多个请求如何挤进同一次 forward |
| 显存管理与复用 | M4-M5 | KV 如何分页存储、共享、复用 |
| 对外服务 | M6 | 如何像 ChatGPT 一样通过 API/SSE 流式输出 |
| 能力扩展与性能深化 | M7+ | 按主题包集中推进：模型结构、解码加速、算子加速、长上下文、多模态 |

<!-- anchor:invariants -->
## 1. 不变量

- **仓库**：`luhao-lab/inferlite`（公开 / MIT）。
- **代码边界**：核心业务代码 `inferlite/*.py` 由你手敲；Agent 做计划、Review、测试建议、文档、文章草稿。
- **学习 > 性能**：先清晰复现机制，再逐步优化性能；M4 不死磕 Triton，kernel 留给 M9。
- **里程碑闭环**：每个 M 完成 = 代码 push + 测试通过 + 文档回填 + 文章发布 + PROGRESS 更新。
- **任务卡先行**：开工前写清前置、产出、算法核心、验证、DoD、坑点。
- **提交纪律**：同一 commit 不混“基础设施改动”和“算法实现”。

<!-- anchor:milestone-table -->
## 2. 里程碑总表

| M | 状态 | 能力方向 | 主层 | 依赖 | 交付 |
| --- | --- | --- | --- | --- | --- |
| M1·P1 | 已完成 | 跑通基础模型 | L1 Model | - | 手写 Qwen3，logits 对齐 transformers |
| M1·P2 | 已完成 | 最小生成能力 | L1 + L3 | M1·P1 | CLI 能出字，最小 Engine/Sampler |
| M2 | 已完成 | 单请求加速 | L2 Memory | M1 | 单序列 decode 不重算历史 |
| M3 | 已完成 | 多请求并发 | L3 Scheduler | M2 | 多请求三队列调度，slot 复用 |
| M4 | 进行中 | 显存分页管理 | L2 Memory | M3 | KV 按 block 管，支持 CoW/refcount |
| M5 | 未开始 | 前缀复用 | L2 Memory | M4 | 相同前缀复用 KV，reasoning 字段可解释 |
| M6 | 未开始 | 服务化输出 | L4 Server | M3（建议 M5 后） | `inferlite serve` + curl 流式输出 |
| Release | 未开始 | 工程发布 | 工程发布 | M3-M6 | 对照表、CI、README badge、`v1.0` tag |
| M7 | Backlog | MoE 模型支持 | L1 Model | M6 | 阶段 1 跑通 MoE；阶段 2 MoE 性能优化 |
| M8 | Backlog | 推测解码加速 | L3 Engine | M6 | 阶段 1 n-gram；阶段 2 EAGLE |
| M9 | Backlog | 核心算子加速 | L1/L2 Kernel | M4 | 阶段 1 cache write kernel；阶段 2 paged attention kernel |
| M10 | Backlog | 长上下文能力 | L1 + L2 + L3 | M4 | 阶段 1 Chunked Prefill；阶段 2 YaRN |
| M11 | Backlog | 多模态能力 | L1 + L2 + L4 | M6 | 阶段 1 VLM 教学版；阶段 2 VLM 工程化 |
| M12+ | Backlog | 工程能力扩展 | 不定 | 不定 | Serving / Quant / Distributed / Audio 等按需新开 |

### 关键依赖说明

| 依赖 | 类型 | 原因 |
| --- | --- | --- |
| M3 → M6 | 硬依赖 | API/SSE 要包 `batch_generate` 或后续 EngineCore；单请求 API 不是目标服务形态。 |
| M4 → M5 | 硬依赖 | Prefix Cache 依赖 block table / refcount / CoW；没有 M4 只能做字符串级缓存。 |
| M5 ↔ M6 | 无算法依赖 | Prefix Cache 是 L2 Memory；API/SSE 是 L4 Server；二者只是共同服务于 v1.0 demo。 |
| M4 → M9 | 硬依赖 | Triton kernel 替换 M4 的 PyTorch PagedAttention 伪版。 |
| M7+ 内部 | 主题内阶段依赖 | M7/M8/M10/M11 都遵循“先跑通 → 再优化/工程化”；同一 M 内只做一个能力主题。 |

<!-- anchor:current-mainline -->
## 3. 当前主线：M3–M6

> 主线顺序建议：**M4 → M5 → M6 → Release**。M6 技术上可在 M3 后启动，但服务层最好暴露完整的调度 + 内存复用能力，所以建议放在 M5 后收口。

### M3 — Continuous Batching：调度器的诞生

> 方向：L3 Scheduler / 多请求调度 / Continuous Batching ｜ 主层：L3 ｜ 依赖：M2

**目标**：8 个并发请求每 step 重新组 batch，无 head-of-line blocking；serial vs batch 输出 token 级等价。

**核心概念**：
- `waiting / running / finished` 三队列。
- 变长 attention mask。
- EOS 立即出队，新请求立即入队。
- slot 复用与 per-slot KV 长度。

**验证**：
- `test_scheduler_invariant.py`：三队列守恒、EOS 立即出队。
- `test_batch_e2e.py`：多请求 batch 输出与单条串行 token 级一致。
- benchmark 归档到 `bench/results/`。

**性能结论（必须保留）**：
- 纯 PyTorch 教学版在 MPS 上 `batch_generate` 比 M2 serial 慢，实测约 `0.38x–0.44x`。
- 主瓶颈不在 attention，而在 cache 读写路径：Python for-loop 写 cache 约 63%，fancy index gather 约 22%，`.item()` 同步约 15%。
- 这是“纯 PyTorch + 不调 kernel”的路线选择，不是 continuous batching 概念本身的问题。
- nano-vllm 性能接近 vLLM，是因为用了 Triton `store_kvcache_kernel` + FlashAttention，而不是纯 PyTorch。
- 性能收益预计到 M4 部分缓解，到 M9 Triton kernel 才系统解决。

**文章**：《Orca 那篇论文做了什么 —— LLM 调度器的诞生》

### M4 — PagedAttention（PyTorch 伪版）：显存当虚拟内存

> 方向：L2 Memory / PagedAttention / Block Table ｜ 主层：L2 ｜ 依赖：M3

**目标**：把连续 KV cache 改成 block table 管理；长 prompt + 多并发时减少显存浪费；支持 refcount 和 Copy-on-Write。

**核心概念**：
- 物理 block、逻辑 block table。
- refcount / Copy-on-Write。
- PyTorch `index_select` 伪版 PagedAttention。
- M4 只做可读教学版，Triton kernel 留到 M9。

**验证**：
- `test_paged_logits.py`：分页前后 logits 对齐。
- `test_block_invariant.py`：refcount 守恒、CoW 正确。
- GPU 上验证显存利用率；Mac 只验证功能正确性。

**前置关系**：M4 是 M5 Prefix Cache 的硬前置；没有 block table，就讲不清 prefix block 复用。

**文章**：《把显存当虚拟内存用 —— PagedAttention 的设计精髓》

### M5 — Prefix Cache + Reasoning：把 KV 变成可复用资产

> 方向：L2 Memory / Prefix Cache / KV 复用 ｜ 主层：L2 ｜ 依赖：M4

**目标**：相同前缀的请求复用 prefix KV；多轮对话第二轮 TTFT 明显下降；reasoning 字段为服务协议输出做准备。

**核心概念**：
- block hash / RadixTree-Lite。
- prefix cache hit rate。
- refcount 与 eviction。
- Qwen3 thinking / non-thinking 的 `reasoning_content` 分流。

**验证**：
- `test_prefix_invariant.py`：同前缀两请求 block_id 一致，refcount 正确，eviction 后不悬挂。
- reasoning 字段解析单测。
- GPU 上复测 TTFT 收益。

**与 M6 的关系**：M5 和 M6 无算法依赖。M5 是 L2 Memory，M6 是 L4 Server；只是最终都会进入 v1.0 demo。

**文章**：《为什么多轮对话第二轮能更快 —— Prefix Cache 的本质》

### M6 — API + SSE 服务化：把 Engine 包成服务

> 方向：L4 Server / OpenAI API / SSE 流式输出 ｜ 主层：L4 ｜ 依赖：M3；建议 M5 后做

**目标**：`inferlite serve qwen3-0.6b` 起服务；兼容 OpenAI Chat Completions 基本格式；curl 可看到 SSE 流式 token delta。

**核心概念**：
- FastAPI。
- OpenAI Chat Completions 基本格式。
- SSE chunk 格式。
- logits processor：greedy / temperature / top-k / top-p / repetition penalty。
- 如果 M5 已完成，映射 `reasoning_content`。

**验证**：
- `test_openai_api.py`：请求/响应格式兼容。
- SSE 流式格式测试。
- sampler seed 可复现测试。
- 服务启动 smoke test。

**文章**：《从 Python 函数到 ChatGPT 式流式服务》

<!-- anchor:release-checklist -->
## 4. Release Checklist（v1.0，不单独算 M）

**触发条件**：M3/M4/M5/M6 都完成，核心 demo 具备“多请求调度 + 分页内存 + 前缀复用 + 流式服务”。

**交付**：
1. Benchmark 表：`inferlite` vs `transformers.generate` vs `vLLM`，包含 TTFT / ITL / throughput / Mem。
2. `bench/run_all.sh` 一键产出结果，归档到 `bench/results/`。
3. 最小 GitHub Actions：CPU-only 跑 `unit + module(tiny) + invariant`。
4. README 更新 v1.0 demo、限制、benchmark 结论、CI badge。
5. 仓库打 `v1.0` tag。

**评测原则**：同 prompt 集、同硬件、同精度、固定 seed、warmup 后统计。性能对照必须上 GPU；Mac 只做功能与开发验证。

**标准 prompt**：
- ShareGPT-100：主 benchmark，长度方差大，能测调度、PagedAttention、Prefix Cache。
- GSM8K-20：reasoning + 服务协议验证。
- 不用 MMLU：输出太短，无法测调度/缓存能力。

**文章**：《2000 行实现一个 vLLM —— inferlite v1 总览》

<!-- anchor:backlog -->
## 5. M7+ Backlog

> M7+ 不再按技术点排成一条线，而是按能力主题包组织。原则：同一个 M 内集中攻克同一主题，先跑通，再优化；不要把 MoE、Spec、Kernel、VLM 等不同主题交错推进。

| M | 能力主题 | 要解决的问题 | 阶段安排 | 典型技术 | Mac 可做 |
| --- | --- | --- | --- | --- | --- |
| M7 | MoE 模型支持 | 支持非 dense 大模型结构 | 阶段 1：跑通 MoE；阶段 2：MoE 性能优化 | for-loop dispatch；grouped GEMM / token permutation | 勉强，推荐 GPU |
| M8 | 推测解码加速 | 同样算力下生成更多 token | 阶段 1：n-gram spec；阶段 2：EAGLE | verify/accept/rollback；draft head | 可以开发，训练建议 GPU |
| M9 | 核心算子加速 | 每次 forward 更快 | 阶段 1：cache write kernel；阶段 2：paged attention kernel；阶段 3：Mac-friendly 分支 | Triton；FlashAttention；torch.compile / flex_attention | 主线需 NVIDIA GPU；Mac 只做分支探索 |
| M10 | 长上下文能力 | 支持超长 prompt 和更长上下文窗口 | 阶段 1：Chunked Prefill；阶段 2：YaRN / NTK RoPE scaling | prefill slicing；RoPE scaling | 可以，小心内存 |
| M11 | 多模态能力 | 支持图片输入并逐步工程化 | 阶段 1：VLM 教学版；阶段 2：VLM 工程化 | vision encoder；image prefix cache；encoder/LLM 异步 | 可以，小模型优先 |
| M12+ | 长期工程能力池 | 按需补齐生产特性 | 每个方向单独开 M | LoRA；量化；TP/PP；metrics；Audio | 不定 |

### 暂不规划

- Omni 全双工：工业框架也未稳定，成本高。
- Beam Search：当前教学价值低。
- Encoder-Decoder：不是当前 decoder-only 主线。
- Embedding / Reranker 服务：不属于 LLM 推理引擎核心。
- Diffusion LLM：实现路径差异太大。
- Data Parallel：外层多实例 + LB 即可，不作为核心引擎里程碑。

<!-- anchor:appendix -->
## 6. 附录：工程约定

### 6.1 四层抽象

| 层 | 本质问题 | 当前主线 |
| --- | --- | --- |
| L1 Model | tokens → logits | Qwen3、MoE、VLM、RoPE、kernel |
| L2 Memory | KV 怎么存、共享、复用 | KV Cache、PagedAttention、Prefix Cache |
| L3 Engine | 多请求怎么调度和生成 | Continuous Batching、Sampler、Spec、Chunked Prefill |
| L4 Server | 外部怎么调用 | OpenAI API、SSE、multipart |

一句话：用 L3 调度器把多个请求复用同一份 L1 模型权重和 L2 显存，再通过 L4 协议对外服务。

### 6.2 模型扩展原则

- M1-M6 只支持 Qwen3 dense，不做 registry。
- M7 引入 `model/registry.py`，按 HF `config.json -> architectures` 分发模型。
- 每个模型内部消化 attention、RoPE、Norm、FFN、MoE 等差异；框架层只依赖 `LLMModel` 最小接口。
- `inputs_embeds` 预留给 M11 VLM，避免后续大改 forward 入口。

### 6.3 测试原则

| 层 | 粒度 | 典型位置 | 目标 |
| --- | --- | --- | --- |
| L0 | 单函数 / 单算子 | `tests/unit/` | 数学闭式或 transformers 同名函数对齐 |
| L1 | 模块 forward | `tests/module/` | logits / hidden states 数值对齐 |
| L2 | 端到端行为 | `tests/e2e/` | 贪心 token 序列一致 |
| L3 | 系统不变式 | `tests/invariant/` | queue、block、refcount、spec 接受率等状态守恒 |

常用命令：

```bash
pytest tests/unit
pytest tests/module
pytest tests -m "not slow"
pytest tests
```

CI 在 Release Checklist 引入，只跑 CPU-only 的 fast tests；真实模型 slow tests 本地手跑。

### 6.4 硬件原则

| 阶段 | Mac MPS | GPU |
| --- | --- | --- |
| M1-M6 | 主开发可用 | 性能评测更准 |
| Release benchmark | 功能可跑 | 必须，用于对照表 |
| M7/M8/M10/M11 | 大多可开发 | 大模型/性能更合适 |
| M9 | 主线不支持；Mac-friendly 分支可探索 | 必须 NVIDIA GPU |

Triton 与 FlashAttention 主要面向 CUDA；Mac 加速可探索 `torch.compile` / `flex_attention`，但不作为主线性能目标。

### 6.5 协作与文章

| 工作 | 谁来做 |
| --- | --- |
| 里程碑定义、范围裁剪、依赖分析 | Agent |
| 核心业务代码 `inferlite/*.py` | 你手敲 |
| Review、测试建议、文档、文章草稿 | Agent |
| 知乎发布、最终取舍 | 你 |

每篇文章固定结构：一句话本质 → 问题背景 → 原理 → 关键代码讲解 → 对照实现 → Benchmark/现象 → 本质题。

### 6.6 风险与应对

| 风险 | 应对 |
| --- | --- |
| Agent 越界写核心代码 | 立刻回滚，改成“思路 + 伪代码 + Review” |
| 过早优化 | M4 只写 PyTorch 伪版，Triton 留 M9 |
| 范围蔓延 | 新能力一律进入 M7+ Backlog，且按主题包集中推进，不塞进 M3-M6 |
| 没 GPU | Mac 做功能，GPU 只在 Release/M9 等性能点租用 |
| 文章烂尾 | 文章发布作为里程碑闭环的一部分 |

<!-- anchor:next-action -->
## 7. 下一步

当前优先级：**收尾 M3 → 启动 M4**。

每开一个新 M，先补任务卡，再写代码；不要在 PLAN 里继续堆实现细节。
