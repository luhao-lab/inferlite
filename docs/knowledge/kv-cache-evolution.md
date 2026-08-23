# 从零手写 LLM 推理引擎（二）：KV Cache 的五次进化

本文是"从零手写 LLM 推理引擎"系列的第二篇，覆盖里程碑 M2 到 M5。上一篇文章（M1）讲了如何从零实现一个 Qwen3 的朴素前向推理；这篇文章讲一件更大的事——**如何让这个推理引擎真正能用**。

M1 的引擎能跑，但慢得离谱：每生成一个 token，都要把前面所有 token 重新算一遍。这不是 bug，是设计如此——M1 没有缓存。接下来的四个里程碑，M2 到 M5，做的事情本质上只有一件：**让已经算过的 K/V 向量不要白算**。

这四个里程碑不是并列关系，而是严格的递进关系。每一步都解决上一步的一个根本瓶颈，而不是在已有方案上打补丁。把这四步串起来看，你会看到一条清晰的演进线索——从"一个人能跑"到"一个系统能服务"。

**本文的叙事结构**：先讲清 KV Cache 在缓存什么（§0），再展开一个 KV Cache 系统需要回答的五个设计问题（设计蓝图），然后展示 inferlite 的代码架构（六个模块怎么协作），接着逐个讲述 M2-M5 的实现——每一步解决蓝图中的一个问题。看完实现后，再深入拆解工业框架 vLLM 和 nano-vllm 的完整架构，对比 inferlite 每一步"学了什么、简化了什么"。读完本文，你应该能回答：**一个推理引擎的 KV Cache 为什么长成现在这个样子。**

---

## 项目概览

| | |
|---|---|
| **项目地址** | [inferlite](https://github.com/luhao-lab/inferlite) · Tags `m2/static-kv-cache` ~ `m5/prefix-caching` |
| **代码版本** | Tag `m5/prefix-caching`（commit `09cc07c`）· 下文所有代码解读均基于此版本 |
| **模型** | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| **代码规模** | ~2,800 行核心实现（cache + engine + attention），314 个单元测试全绿 |

```bash
git clone https://github.com/luhao-lab/inferlite.git
cd inferlite && uv sync

# 切到 M5 tag（本文所有代码解读基于此版本）
git checkout m5/prefix-caching

# 运行全部测试
uv run pytest tests/ -q
```

**里程碑 Tag 对照**：每个里程碑对应一个 git tag，可直接 checkout 查看该阶段的完整代码。

| Tag | Commit 短哈希 | 核心变更 |
|-----|-------------|---------|
| `m1/naive-forward` | — | 无 cache，每步 full forward |
| `m2/static-kv-cache` | — | LayerKVCache + 切片读写 |
| `m3/continuous-batching` | — | BatchedKVCache + FCFS 调度 + slot 管理 |
| `m4/paged-attention` | — | BlockPool + BlockTable + ForwardContext + CacheAdapter |
| `m5/prefix-caching` | `09cc07c` | chain hash + LRU + CoW |

> **注意**：本文的"整体调用架构"章节以 M5 tag 的代码为准讲解（此时架构最完整）。各 M 章节中引用的文件路径和行号也基于 M5 tag——如果你 checkout 到更早的 tag，文件结构可能不同（例如 M2 时代 engine 目录是 `core.py`，到 M4 重构为 `engine.py` + `loop.py`）。

**本文覆盖的四个里程碑**：

| 里程碑 | 一句话 | 核心概念 |
|--------|-------|---------|
| M2 Static KV Cache | 缓存 K/V 向量，单步 O(T²) → O(T) | 静态预分配 + 切片读写 |
| M3 Continuous Batching | 多个请求共享一个模型，动态进退 | 三队列调度 + per-slot 独立长度 |
| M4 PagedAttention | KV 内存从连续数组变成分页块 | block table + scatter/gather |
| M5 Prefix Caching | 相同前缀的请求复用已缓存的 KV | chain hash + LRU + CoW |

---

## 0. 先搞清一个问题：KV Cache 到底在缓存什么

在进入任何一个里程碑之前，有一件事必须先想清楚：**自回归推理的哪些计算是可以跳过的？**

### 符号约定

本文统一使用以下符号，避免歧义：

| 符号 | 含义 | Qwen3-0.6B 典型值 |
|------|------|----------------|
| T_p | prompt 长度（prefill 阶段的 token 数） | 100～1024 |
| N | 最大生成 token 数 | 100～512 |
| T | 当前序列总长 = T_p + 已生成 token 数 | ≤ L_max |
| L_max | KV Cache 的最大容量（generate 前指定） | 1024 |
| L | num_hidden_layers（Transformer 层数） | 28 |
| H_q | num_attention_heads（Q 头数） | 16 |
| H_kv | num_key_value_heads（KV 头数，GQA < MHA） | 8 |
| D | head_dim（每个头的维度） | 64 |
| block_size | 每个物理 block 容纳的 token 数（M4+） | 16 |

> **GQA**（Grouped Query Attention）：H_kv < H_q，多个 Q head 共享同一组 K/V head。Qwen3-0.6B 中 H_q=16、H_kv=8，每 2 个 Q head 共享 1 对 KV。KV Cache 大小与 H_kv 成正比，GQA 直接减少 cache 显存。

### 自回归推理的两个阶段

一个 Transformer 的每一层都有 Attention 计算。Attention 的核心是 Q、K、V 三个向量——Q 来自当前要预测的 token，K 和 V 来自所有历史 token。推理过程被自然地分成两个阶段：

| 阶段 | 输入规模 | Q、K、V 形状（单层，GQA） | Attention 计算量 | 可缓存？ |
|------|---------|------------------------|----------------|---------|
| **Prefill** | T_p 个 token 并行 | Q: [B, H_q, T_p, D]，K/V: [B, H_kv, T_p, D] | O(T_p²)，T_p × T_p 矩阵乘 | 否——token 间两两计算，无法跳过 |
| **Decode**（每步） | 1 个 token | Q: [B, H_q, 1, D]，K/V: [B, H_kv, T, D]（T 含历史） | O(T)，1 × T 向量乘 | **是**——历史 K/V 不受新 token 影响 |

关键性质是：**位置 t 的 K/V 向量只依赖 t 及之前的 token，不受后续 token 影响**。这是 causal mask 的数学保证。这意味着：prefill 阶段算完所有 prompt token 的 K/V 之后，这些 K/V 向量在整个生成过程中**永远不变**。decode 阶段每步只需要算新 token 的 Q，然后和所有历史的 K/V 做 attention。

### 无缓存 vs 有缓存的计算量

如果不缓存，decode 第 k 步要把前面所有 token 重新过一遍模型才能拿到 K/V：

```text
M1（无缓存）：
  decode 第 k 步：输入序列长度 = T_p + k，每层 Attention 计算量 O((T_p+k)²)
  N 步总量 ≈ Σ(T_p+k)² ≈ O(N·T²)

M2（有缓存）：
  decode 第 k 步：只算 1 个 token 的 Q，读 cache 中 T_p+k-1 个 K/V，计算量 O(T)
  N 步总量 = O(N·T)
```

**以 Qwen3-0.6B 为例**（T_p=100，N=500，T≈600）：

| 指标 | M1（无 cache） | M2（有 cache） |
|------|--------------|--------------|
| decode 单步 Attention | O(T²) ≈ 360,000 | O(T) ≈ 600 |
| N=500 步总量 | O(N·T²) ≈ 1.8 亿 | O(N·T) ≈ 30 万 |
| 理论加速比 | 1× | **600×** |

**KV Cache 就是这件事。** 没有它，LLM 推理是不可用的；有了它，推理引擎才有了效率基础。后面所有的 M3/M4/M5，都是在"已经缓存了 K/V"这个前提下，解决**怎么存、怎么共享、怎么复用**的问题。

---

## 设计蓝图：一个 KV Cache 系统要回答的五个问题

§0 讲清了 KV Cache **缓存什么**（K/V 向量）和**为什么能缓存**（causal mask 保证 K/V 不变）。但"缓存"本身只是起点——把 K/V 存下来只是第一步，真正要把一个推理引擎做到能用、好用、高效，需要回答一系列递进的设计问题。

这一节把这些问题的全貌摊开来讲，让你在看任何一个里程碑之前，就知道整条路线要走到哪里、为什么是这个顺序。

### 问题 1：存在哪里？——内存布局

最直觉的做法是每层维护一个动态增长的 list，每生成一个 token 就 append。但这在 GPU/MPS 上极其低效——每次 append 都要重新分配内存、拷贝数据。

**正确做法**：预分配一个固定大小的 tensor `[num_layers, num_kv_heads, max_seq_len, head_dim]`，每步只写对应位置的切片，不做任何内存分配。这是 M2 做的事情。

### 问题 2：多人怎么办？——并发服务

问题 1 解决的是单请求的缓存。但实际场景是多个请求同时到达——一个用户在问问题，另一个用户在等回答。每个请求有自己的序列长度、自己的生成进度、自己的 K/V 数据。

**正确做法**：给每个请求分配一个独立的 slot，每个 slot 有自己的序列长度和缓存区域。请求可以随时加入（admit）和完成（finish），不影响其他请求。这是 M3 做的事情——continuous batching。

### 问题 3：内存碎片怎么消除？——分页管理

M2 和 M3 的缓存都是**连续分配**的：每个请求（或 slot）预占一段连续内存，长度等于 `max_seq_len`。但绝大多数请求的实际长度远小于 `max_seq_len`——一个 slot 分配了 2048 个位置，实际只用了 200 个，剩下的 1848 个位置全是浪费。当并发请求多时，这种内碎片会迅速耗尽显存。

**正确做法**：把缓存空间切成固定大小的 block（类似操作系统的虚拟内存页），每个请求维护一个 block table 记录自己用了哪些物理 block。内存不再是连续分配，而是按需分页。这是 M4 做的事情——PagedAttention。

### 问题 4：相同计算为什么要做两遍？——前缀复用

M4 解决了内存碎片，但两个请求如果有相同的前缀（比如都包含同一段 system prompt），它们各自分配了独立的 block，各自做了相同的 prefill 计算，存了完全一样的 K/V 数据。这些重复计算和重复存储在真实服务场景中极其浪费——多轮对话、共享 system prompt、few-shot examples 都是高前缀重合的场景。

**正确做法**：对已填满的 block 计算 chain hash，新请求到来时查 hash 索引，命中就直接复用已有的物理 block 和 KV 数据，跳过这部分 prefill。多个请求共享 block 时通过引用计数 + Copy-on-Write 保证写入安全。这是 M5 做的事情——Prefix Caching。

### 问题 5：谁来协调？——架构解耦

上面四个问题都解决之后，系统里已经有了：调度器（管谁该跑）、缓存管理器（管 block 怎么分配和复用）、attention 层（管 K/V 怎么读写）、采样器（管下一个 token 是什么）。这些模块之间怎么传递信息？

如果让 attention 直接调用缓存管理器、让调度器直接操作缓存 tensor，模块之间会互相耦合，改一处牵全身。

**正确做法**：引入一个统一的上下文对象（ForwardContext）传递每步的元数据（哪些 slot 在跑、各自的序列长度、block 映射），引入一个 Protocol 接口（CacheAdapter）统一缓存的读写操作。attention 层通过 context 读元数据、通过 adapter 操作缓存，不依赖任何具体的缓存实现。这是 M4 引入的架构，M5 在此之上零改动 attention 就完成了 prefix cache。

### 为什么是这个顺序

这五个问题不是可以任意排列的——它们有严格的依赖关系：

```text
问题 1（内存布局）
  ↓  不知道数据怎么存，就无法讨论并发
问题 2（并发服务）
  ↓  没有并发就没有碎片问题
问题 3（分页管理）
  ↓  没有分页就无法做 block 级的复用
问题 4（前缀复用）

问题 5（架构解耦）贯穿始终：M2/M3 用简单接口，M4 引入完整架构
```

跳步会出问题。如果你在 M2 就直接做 PagedAttention，会同时面对"内存布局 + 分页 + block 管理"三个新概念，代码和测试都会变得不可控。如果你跳过 M3 直接做 M4，没有并发调度就无法理解 block table 为什么存在。每一步都只引入一个新概念，在上一步的基础上解决一个根本瓶颈——这是 inferlite 的核心设计原则。

### inferlite 的设计目标

| 目标 | 说明 | 不是什么 |
|------|------|---------|
| **可读性优先** | 纯 PyTorch 实现，所有 cache 操作都是 Python/PyTorch 原生代码，可以用 debugger 逐行跟踪 | 不是追求性能最优 |
| **概念完整性** | 覆盖 vLLM 的核心架构概念（PagedAttention / Continuous Batching / Prefix Cache / ForwardContext） | 不是 vLLM 的复刻 |
| **递进式理解** | 每个里程碑只解决一个新问题，代码增量最小化，测试覆盖每一步的正确性 | 不是一步到位的生产系统 |
| **可验证性** | 每个里程碑都有与 ground truth 的等价性验证（有 cache = 无 cache、batch = serial） | 不是只测 happy path |

inferlite 的定位不是"另一个推理引擎"，而是一个**教学框架**：用最小的代码量，把工业推理引擎的核心机制拆成四步，每一步都完整、可跑、可验证。理解了 inferlite 之后再去看 vLLM 的源码，你会知道每一层代码在解决什么问题。

---

## 整体调用架构：六个模块怎么协作

在讲 M2~M5 的具体实现之前，必须先把 inferlite 的**代码架构全貌**讲清楚。不然后面每个里程碑讲到 cache 读写、metadata 传递、adapter 切换时，你会不知道这些组件从哪来、怎么接上的。

这一节的结构是**先总后分**：先看用户的一次调用怎么穿越整个系统（总），再逐个拆解每个模块的内部设计（分）。

### 一句话总结

inferlite 的 ~2800 行代码分成 **6 个包**，每个包的职责用两个字概括：

```text
engine   → 驱动     scheduler → 排谁     cache   → 存数据
model    → 计算     sampler   → 选词     cli     → 入口
```

**依赖方向是单向的**：`engine → scheduler + cache + model + sampler`。model 不知道 scheduler 的存在，scheduler 不知道 cache 的实现，cache 不知道 model 的存在。

### 用餐厅比喻理解整个系统

如果你没看过推理引擎的代码，可以先用这个比喻建立直觉：

> inferlite 就像一个**餐厅**。顾客（请求）来了要排队，前台安排谁先吃，厨房按菜单做菜，做好的菜端给顾客。

```text
┌────────────────────────────────────────────────────────────────────────┐
│                         inferlite = 餐厅                              │
│                                                                        │
│  顾客（prompt）  ──►  前台（scheduler）  ──►  厨房（engine + loop）     │
│                         │                        │                     │
│                    "谁先来谁先吃"            "每轮做两件事：             │
│                     排队等候                 prefill 备料               │
│                                              decode 上菜"              │
│                         │                        │                     │
│                         │                   ┌────┴────┐               │
│                         │                   ▼         ▼               │
│                         │             厨师（model）  菜谱（attention） │
│                         │              "28 层工序"    "每层怎么做       │
│                         │                                菜"           │
│                         │                                              │
│                    冰箱（cache）                                       │
│                  "做好的半成品（KV）                                    │
│                   存这里，下次不用重做"                                  │
│                                                                        │
│                    冰箱管理员（adapter）                                 │
│                  "不同里程碑的冰箱不一样大，                              │
│                   但管理员对外接口一样"                                  │
│                                                                        │
│                    取菜单（ForwardContext）                              │
│                  "每做一道菜前贴一张单子：                                │
│                   告诉厨师这次用冰箱的哪一格"                             │
└────────────────────────────────────────────────────────────────────────┘
```

**为什么需要这么多模块？** 直觉上你可能觉得"一个函数就能跑推理"。但当你要支持多种 cache 策略（静态/slot/分页）、多种并发方式（单请求/多请求）时，如果全写在一个文件里，每加一种策略就要改一堆地方。inferlite 的解法是**分层 + 接口隔离**：

- **engine/loop.py** 写主循环（"排队 → 备料 → 上菜 → 检查完成"），但它**不关心冰箱是哪种**——它只和管理员（adapter）说话
- **cache/adapter.py** 是管理员，对外接口统一（8 个方法），但内部有 3 种实现（对应 M2/M3/M4 三种冰箱）
- **model/attention.py** 是厨师，它只看两样东西：冰箱在哪（通道 A）和这次的取菜单（通道 B），不关心外面的调度逻辑

这意味着：**加一种新的 cache 策略，只需要写一个新的 adapter 实现，engine 和 model 一行代码都不用改。** M5 的 prefix cache 就是这样加的——attention.py 零改动。

### 两条信息通道：厨师怎么知道该做什么

这是架构中最核心的设计。厨师（attention）在厨房最里面，隔了三层才到前台。它需要两类信息才能正确工作：

```text
通道 A（静态，开工前接好一次）：冰箱在哪？
  类比：把冰箱搬到厨师面前，告诉他"所有半成品都在这个冰箱里"
  代码：adapter.bind_kv_cache(model) → attention.kv_cache = cache tensor
  时机：推理开始前调一次，整个推理期间不变

通道 B（动态，每做一道菜更新一次）：这次用冰箱的哪一格？
  类比：每道菜开做前，往厨师面前的白板（ForwardContext）上贴一张取菜单
  代码：set_forward_context(metadata) → attention 通过 get_forward_context() 读取
  时机：每次 model forward 前更新一次
```

为什么不用参数直接传？因为厨师和前台隔了 `engine → model → 28 layers → attention` 这么深的调用链，如果每层函数都要加参数传 metadata，所有中间代码都要改。全局变量（ForwardContext）就像厨房墙上的一块公共白板——谁都能看到，谁都不用传话。

### 总：用户一次调用穿越全系统

假设用户调了 M4（PagedAttention）的 batch 推理，传入 3 条 prompt，最多生成 100 个 token：

```python
from inferlite.engine.engine import batch_generate_paged

results = batch_generate_paged(
    model, sampler, prompts, max_new_tokens=100,
    num_blocks=64, block_size=16, config=config,
)
```

这行调用背后发生了什么？按时间顺序走一遍：

```text
batch_generate_paged()                                    ← engine/engine.py
  │
  ├─ ① 构造三件套：cache + adapter + scheduler
  │    cache    = PagedKVCache.from_config(...)       # 分配 [64, 16, 8, 64] 的物理 block tensor
  │    adapter  = PagedCacheAdapter(cache)             # 包装成统一接口
  │    scheduler = FCFSScheduler(max_num_seqs=64)      # 排队管理
  │    for prompt in prompts: scheduler.submit(req)    # 3 个请求进入 waiting 队列
  │
  └─ ② 进入主循环：batch_generate_loop()              ← engine/loop.py
       │
       ├─ adapter.bind_kv_cache(model)                 ← 一次性接线：cache tensor → 28 个 Attention 层
       │
       └─ while has_unfinished():                      ← 每轮迭代 4 步
            │
            ├─ ❶ ADMIT：scheduler.waiting → running
            │    adapter.can_admit(prompt_len)         ← 容量够不够？（block 是否充足）
            │    adapter.allocate(req_id, prompt_len)  ← 分配 block
            │
            ├─ ❷ PREFILL：新请求拼 batch → 一次 forward
            │    input_ids, positions = _build_prefill_batch(admitted)
            │    metadata = adapter.make_prefill_metadata(...)   ← 构造元数据（含 block_table）
            │    with set_forward_context(metadata):              ← 【通道 B】写入全局变量
            │      logits = model(input_ids, positions)           ← 【通道 A】读 cache tensor
            │    next_token = sampler(logits)
            │
            ├─ ❸ DECODE：running 请求拼 batch → 一次 forward
            │    adapter.prepare_decode(request_ids)              ← 为新 token 分配 block 空间
            │    next_tokens, positions = _build_decode_batch(running)
            │    metadata = adapter.make_decode_metadata(...)
            │    with set_forward_context(metadata):
            │      logits = model(next_tokens, positions)
            │    next_token = sampler(logits)
            │
            └─ ❹ 完成检查
                 if done: scheduler.mark_finished(req) + adapter.free(req_id)
```

注意 model 内部发生了什么（以一次 forward 为例）：

```text
model(input_ids, positions)                               ← model/qwen3.py
  │
  ├─ hidden = embed_tokens(input_ids)                     ← [B, T] → [B, T, 1024]
  ├─ pos_emb = rotary_emb(hidden, positions)              ← 只算一次，传入 28 层
  │
  └─ for layer in 28 layers:                              ← model/qwen3.py → attention.py
       layer(hidden, pos_emb)
         └─ attention.forward(q, k, v):
              │
              ├─ cache = self.kv_cache            ← 【通道 A】bind_kv_cache 设好的物理 tensor
              ├─ metadata = get_forward_context() ← 【通道 B】set_forward_context 设好的元数据
              │
              ├─ cache_rw(k, v, metadata):        ← 写入 + 读取 cache
              │    写入：scatter k/v 到 block_table 指定的物理位置
              │    读取：gather block_table 中的物理 block 拼回连续序列
              │
              └─ attention 计算：repeat_kv → q @ k^T → mask → softmax → @ v

logits = lm_head(hidden)                                  ← [B, T, 1024] → [B, T, 151936]
```

从用户调 `batch_generate_paged()` 到拿到 `results`，数据穿越了 4 层代码：`engine → loop → model → attention`。每层的职责边界极其清晰——**engine 管组装，loop 管编排，model 管计算，attention 管 cache**。

### 代码文件全景

```text
inferlite/
├── engine/                  推理引擎：谁在驱动整个流程
│   ├── engine.py   (249L)   三个入口函数，构造 cache + adapter + scheduler
│   ├── loop.py     (249L)   M3/M4 共享的主循环：admit → prefill → decode → free
│   ├── context.py  (111L)   ForwardContext + AttentionMetadata（元数据传递）
│   └── metrics.py           性能指标采集
│
├── cache/                   KV 缓存：数据怎么存
│   ├── adapter.py  (338L)   CacheAdapter Protocol + 3 种实现
│   ├── kv_cache.py (135L)   M2: 单序列静态 tensor
│   ├── batched_kv_cache.py  M3: 多 slot 的 BatchedKVCache
│   ├── paged_kv_cache.py    M4: block 分页 PagedKVCache
│   └── block_pool.py        M4+M5: BlockPool + BlockTable + prefix hash
│
├── scheduler/               调度器：谁该跑
│   ├── fcfs.py     (101L)   FCFS 调度器：waiting → running → finished
│   └── request.py  (84L)    RequestState 数据类
│
├── model/                   模型：怎么算
│   ├── qwen3.py    (342L)   Qwen3ForCausalLM = Qwen3Model + lm_head
│   ├── attention.py(434L)   Qwen3Attention + Attention（cache 读写核心）
│   ├── layers.py            RMSNorm + RoPE + SwiGLU
│   └── weights.py           HF 权重加载
│
├── sampler/
│   └── greedy.py            argmax 采样
│
└── cli.py                   命令行入口
```

---

### 分①：三个入口函数 — engine.py

engine.py 提供三个入口，对应三条 cache 路径。每个入口做的事情都一样：**构造 cache → 构造 adapter → 构造 scheduler → 提交请求 → 启动循环**。

```python
# ── M2 单序列 ──
def generate(engine, input_ids, max_new_tokens, kv_cache=None):
    adapter = SingleCacheAdapter(kv_cache)
    adapter.bind_kv_cache(engine.model)
    # 手动跑 prefill + decode 循环（单请求，不需要 scheduler）
    with set_forward_context(adapter.make_prefill_metadata(...)):
        logits = engine.model(input_ids, positions=...)
    ...

# ── M3 batched ──
def batch_generate(model, sampler, prompts, max_num_slots, ...):
    cache     = BatchedKVCache.from_config(...)    # [S, 8, 1024, 64] 多 slot tensor
    adapter   = BatchedCacheAdapter(cache)
    scheduler = FCFSScheduler(max_num_seqs=S)
    for prompt in prompts: scheduler.submit(RequestState(...))
    return batch_generate_loop(model, sampler, scheduler, adapter, ...)  # ← 共享主循环

# ── M4 paged ──
def batch_generate_paged(model, sampler, prompts, num_blocks, block_size, ...):
    cache     = PagedKVCache.from_config(...)      # [num_blocks, block_size, 8, 64] block tensor
    adapter   = PagedCacheAdapter(cache)
    scheduler = FCFSScheduler(max_num_seqs=num_blocks)
    for prompt in prompts: scheduler.submit(RequestState(...))
    return batch_generate_loop(model, sampler, scheduler, adapter, ...)  # ← 同一个主循环！
```

M3 和 M4 最后都调用了**同一个** `batch_generate_loop()`——传入的参数只有 adapter 不同。这就是 CacheAdapter Protocol 的效果：**loop.py 的代码一字不改，行为完全由 adapter 决定**。

### 分②：主循环 — loop.py

M3 和 M4 共享的主循环 `batch_generate_loop()` 只有 ~100 行有效代码，每轮迭代做四件事：

```python
def batch_generate_loop(model, sampler, scheduler, adapter, ...):
    adapter.bind_kv_cache(model)           # ① 初始化：cache tensor 绑定到 Attention 层（只做一次）

    while scheduler.has_unfinished():

        # ── ❶ ADMIT：从 waiting 取请求到 running ──
        while scheduler.waiting:
            req = scheduler.waiting[0]
            if not adapter.can_admit(prompt_len):       # ← adapter 决定容量检查逻辑
                break
            num_cached = adapter.can_admit_with_cache(prompt_ids)  # M5: prefix cache
            scheduler.waiting.popleft()
            scheduler.running[req.request_id] = req
            if num_cached > 0:
                adapter.allocate_with_cache(...)         # M5: cache-aware 分配
            else:
                adapter.allocate(req.request_id, prompt_len)

        # ── ❷ PREFILL：新请求拼 batch → 设置 metadata → model forward ──
        if admitted:
            input_ids, positions = _build_prefill_batch(admitted, device)
            metadata = adapter.make_prefill_metadata(input_ids, positions)
            with set_forward_context(metadata):         # ← metadata 放进全局变量
                logits = model(input_ids, positions=positions)
            # 采样首个 token ...

        # ── ❸ DECODE：running 请求拼 batch → 设置 metadata → model forward ──
        adapter.prepare_decode(request_ids)              # ← 为新 token 分配 cache 空间
        next_tokens, positions = _build_decode_batch(running, device)
        metadata = adapter.make_decode_metadata(next_tokens, positions)
        with set_forward_context(metadata):
            logits = model(next_tokens, positions=positions)
        # 采样 + 检查完成 + 释放 ...

        # ── ❹ 完成检查 ──
        if is_done:
            scheduler.mark_finished(req)
            adapter.free(req.request_id)
```

loop.py **从不直接操作 cache tensor**。它只通过 adapter 的方法间接操作。这意味着 loop.py 不知道也不关心底层是 slot 还是 block——它只看到一个统一的接口。

### 分③：两条通道的代码实现

前面用餐厅比喻讲了两条通道的"为什么"，这里看具体代码的"怎么做"。

**通道 A：bind_kv_cache** — 一次性接线

在推理开始前调用一次，把 cache 的物理 tensor 连接到 28 个 Attention 层的 `self.kv_cache` 属性上：

```text
bind_kv_cache 之前：
  Attention.kv_cache = None          ← 不知道数据存在哪

bind_kv_cache 之后（以 M4 为例）：
  layer[0].self_attn.attn.kv_cache = PagedKVCache 实例
  layer[0].self_attn.attn.layer_idx = 0
  layer[1].self_attn.attn.kv_cache = PagedKVCache 实例（同一个！）
  layer[1].self_attn.attn.layer_idx = 1
  ...
  layer[27].self_attn.attn.kv_cache = PagedKVCache 实例（同一个）
  layer[27].self_attn.attn.layer_idx = 27
```

此后 Attention 通过 `self.kv_cache.layers[self.layer_idx]` 访问本层的物理 tensor。整个推理期间这个绑定不变。

**通道 B：ForwardContext** — 每轮更新

问题很清楚：**loop.py 知道 metadata（哪些请求在跑、各自的序列长度、block 映射），但 attention 层隔了三层调用**。如果把这些 metadata 作为参数一层层传下去，所有中间函数签名都要改。

vLLM V1 的解法是一个**全局 context manager**：

```python
# context.py — 整个文件只有 60 行

@dataclass
class AttentionMetadata:
    """每轮 forward 的元数据。纯 tensor，不含 cache 引用。"""
    num_seqs: int                              # 当前 batch 中有多少个请求
    seq_lens: torch.Tensor                     # [num_seqs] 每个请求的序列长度
    slot_mapping: torch.Tensor | None = None   # [num_tokens] M3: 每个请求对应的 slot_id
    block_table: torch.Tensor | None = None    # [num_seqs, max_blocks] M4: 每个请求的 block 映射

_forward_context: ForwardContext | None = None  # 模块级全局变量

@contextmanager
def set_forward_context(attn_metadata):
    global _forward_context
    _forward_context = ForwardContext(attn_metadata)
    try:
        yield                        # ← with 块内 model forward 执行
    finally:
        _forward_context = None      # ← forward 结束自动清除

def get_forward_context():
    return _forward_context          # ← attention 层调这个拿到 metadata
```

工作方式极其简单：

1. **loop.py** 在每次 `model(...)` 之前调 `set_forward_context(metadata)`
2. **attention.py** 在 `forward()` 内部调 `get_forward_context().attn_metadata` 拿到 `seq_lens`、`block_table` 等
3. `with` 块结束后自动清除，下一轮 forward 重新设置

**metadata 不经过 model 的函数参数传递**。model.forward 签名始终是 `model(input_ids, positions)`——不管底层是 M2/M3/M4，签名不变。metadata 走的是旁边一条"暗道"（全局变量），避免了 model 层的接口被 cache 实现细节污染。

### 分④：CacheAdapter — 8 个方法的统一接口

CacheAdapter 是一个 Python Protocol（类似 Java 的 interface），定义了 loop.py 和 cache 实现之间的契约：

```python
class CacheAdapter(Protocol):
    # ── 生命周期管理（3 个）──
    def can_admit(self, prompt_len: int) -> bool: ...       # 还有没有容量？
    def allocate(self, request_id: str, prompt_len: int) -> None: ...  # 分配 cache 空间
    def free(self, request_id: str) -> None: ...            # 释放 cache 空间

    # ── decode 准备（1 个）──
    def prepare_decode(self, request_ids: list[str]) -> None: ...  # 为新 token 分配空间

    # ── metadata 构造（2 个，纯函数）──
    def make_prefill_metadata(self, input_ids, positions) -> AttentionMetadata: ...
    def make_decode_metadata(self, next_tokens, positions) -> AttentionMetadata: ...

    # ── cache 绑定（1 个）──
    def bind_kv_cache(self, model) -> None: ...   # 把 cache tensor 绑到 Attention 层

    # ── M5 prefix cache（1 个）──
    def can_admit_with_cache(self, prompt_ids) -> int: ...  # 返回命中 block 数
```

**三种实现的核心差异**：

**SingleCacheAdapter（M2）** — 每层绑独立的 `LayerKVCache`

```python
def bind_kv_cache(self, model):
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.attn.kv_cache = self.cache.layers[i]   # LayerKVCache [1, 8, 1024, 64]
        layer.self_attn.attn.layer_idx = i

def make_prefill_metadata(self, input_ids, positions):
    return AttentionMetadata(num_seqs=1, seq_lens=torch.tensor([T]))

def make_decode_metadata(self, next_tokens, positions):
    self.cur_len += 1
    self.cache.cur_len = self.cur_len
    return AttentionMetadata(num_seqs=1, seq_lens=torch.tensor([self.cur_len]))
```

M2 的 metadata 只有 `num_seqs=1` 和 `seq_lens`——单请求，不需要 slot_mapping 也不需要 block_table。

**BatchedCacheAdapter（M3）** — 每层绑 `BatchedLayerKVCache`，多了 `slot_mapping`

```python
def bind_kv_cache(self, model):
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.attn.kv_cache = self.cache.layers[i]   # [S, 8, 1024, 64]

def allocate(self, request_id, prompt_len):
    self.cache.allocate_slot(request_id)    # SlotManager 分配一个空闲 slot_id

def make_prefill_metadata(self, input_ids, positions, request_ids=None):
    slots = [self.cache.slot_manager.req_to_slot[rid] for rid in request_ids]
    return AttentionMetadata(
        num_seqs=B, seq_lens=seq_lens,
        slot_mapping=torch.tensor(slots),   # ← M3 特有：[B] 每个请求占哪个 slot
    )
```

**PagedCacheAdapter（M4）** — 所有层共享同一个 `PagedKVCache`，用 `block_table` 替代 `slot_mapping`

```python
def bind_kv_cache(self, model):
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.attn.kv_cache = self.cache       # ← 同一个 PagedKVCache！
        layer.self_attn.attn.layer_idx = i               # ← 通过 layer_idx 区分各层

def allocate(self, request_id, prompt_len):
    self.cache.allocate_request(request_id, prompt_len)  # BlockPool 分配 block + 创建 BlockTable

def make_prefill_metadata(self, input_ids, positions, request_ids=None):
    return AttentionMetadata(
        num_seqs=B, seq_lens=seq_lens,
        block_table=self._build_block_table(request_ids),  # ← M4 特有：[B, max_blocks]
    )

def _build_block_table(self, request_ids):
    tables = [self.cache.block_tables[rid].block_ids for rid in request_ids]
    max_blocks = max(len(t) for t in tables)
    block_table = torch.zeros(len(request_ids), max_blocks, dtype=torch.long)
    for i, t in enumerate(tables):
        block_table[i, :len(t)] = torch.tensor(t)
    return block_table
```

### 分⑤：attention.py — 根据 cache 类型选读写路径

attention.py 是整个架构中**唯一直接操作 cache tensor 的地方**。它的 `Attention` 类有一个运行时绑定的属性 `self.kv_cache`，forward 时用 `isinstance` 判断类型，选择对应的读写方法：

```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        # ── 1. 根据 kv_cache 类型选择 cache 读写路径 ──
        if self.kv_cache is None:
            pass  # M1: 不做 cache，直接用当前 q/k/v

        elif isinstance(self.kv_cache, PagedKVCache):
            metadata = get_forward_context().attn_metadata   # ← 从通道 B 拿 metadata
            k, v, valid_lens = self._paged_cache_rw(k, v, metadata)

        elif isinstance(self.kv_cache, BatchedLayerKVCache):
            metadata = get_forward_context().attn_metadata
            k, v, cache_positions = self._batched_cache_rw(k, v, metadata)

        elif isinstance(self.kv_cache, LayerKVCache):
            metadata = get_forward_context().attn_metadata
            k, v, cache_position = self._single_cache_rw(k, v, metadata)

        # ── 2. GQA repeat_kv + attention 计算（四条路径共享）──
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        scores = q @ k^T * scaling
        # causal mask + valid_lens mask ...
        output = softmax(scores) @ v
        return output
```

三条 cache RW 路径的核心操作：

**M2 `_single_cache_rw`** — 全局 cur_len，切片读写

```python
cache = self.kv_cache                    # LayerKVCache: k.shape [1, 8, 1024, 64]
cache.k[:, :, pos:pos+T, :] = k          # 写入：切片赋值（原地，零拷贝）
k = cache.k[:, :, :pos+T, :]            # 读取：切片取有效范围（view，零拷贝）
```

**M3 `_batched_cache_rw`** — per-slot 写入，fancy-index 读取

```python
cache = self.kv_cache                    # BatchedLayerKVCache: k.shape [S, 8, 1024, 64]
for i in range(B):
    cache.k[slot_mapping[i], :, pos, :] = k[i]        # 写入：for-loop 逐 slot
k = cache.k[slot_mapping, :, :max_len, :]              # 读取：fancy-index gather（产生拷贝）
```

**M4 `_paged_cache_rw`** — scatter 写入，gather 读取

```python
layer = cache.layers[self.layer_idx]     # [num_blocks, block_size, 8, 64]
flat_cache = layer.k.view(-1, n_kv, D)   # 展平为 [num_blocks*block_size, n_kv, D]

# 写入（scatter）：每个 token 写到 block_table 指定的物理位置
flat_cache[phys * block_size + offset] = k[i, :, 0, :]

# 读取（gather）：按 block_table 把物理 block 拼回连续序列
gathered_k = layer.k[block_table]        # [B, max_blocks, block_size, n_kv, D]
k = gathered_k.reshape(B, L, n_kv, D)
k = k.masked_fill(~valid_mask, 0.0)     # NaN 安全：清零无效位置
```

### 三种 adapter 的差异总结

| | M2 Single | M3 Batched | M4 Paged |
|---|---|---|---|
| **cache tensor shape** | `[1, 8, 1024, 64]` | `[S, 8, 1024, 64]` | `[num_blocks, block_size, 8, 64]` |
| **bind_kv_cache** | 每层绑 LayerKVCache | 每层绑 BatchedLayerKVCache | 每层绑同一个 PagedKVCache + layer_idx |
| **metadata 关键字段** | `seq_lens` | `seq_lens` + `slot_mapping` | `seq_lens` + `block_table` |
| **cache 写入** | 切片 `[:,:,:pos,:]` | for-loop `k[slot,:,pos,:]` | scatter `flat[slot*bs+off]` |
| **cache 读取** | 切片 `[:,:,:pos,:]` | fancy-index `k[slots,:,:max,:]` | gather `k[block_table]` |
| **can_admit 检查** | 永远 True | 有空 slot？ | 有足够 block？ |
| **allocate** | reset cur_len | allocate_slot | allocate_request (分配 block) |
| **prepare_decode** | pass | seq_lens += 1 | append_token (可能分配新 block) |

---

## 1. M2：静态预分配——最简单的缓存方案

> **设计蓝图中的位置**：回答**问题 1（存在哪里）**——用最简单的方案解决内存布局。

### 问题：M1 浪费在哪里

M1 没有 KV Cache。每次 decode 都把完整序列（prompt + 所有已生成 token）重新过一遍 28 层 Transformer。以 Qwen3-0.6B 为例，生成 500 个 token、prompt 100 tokens 的情况下，M1 的总 attention 计算量约 1.8 亿次操作。

这些操作中，绝大部分是重复的：token 0 到 token 598 的 K/V 向量，在生成第 500 个 token 时算了一遍，在生成第 499 个 token 时也算了一遍，在生成第 1 个 token 时还算了一遍。每步都在算，每步结果都一样。

### 方案：预分配一个固定大小的 tensor

M2 的方案极其直觉：在开始推理之前，为每一层分配一个固定大小的 tensor，形状为 `[1, num_kv_heads, max_seq_len, head_dim]`。prefill 时把 prompt 所有 token 的 K/V 写进去，decode 时每步追加一个 token 的 K/V。

```python
@dataclass
class LayerKVCache:
    k: torch.Tensor  # [1, 8, 1024, 64]  — 8 个 KV head，最长 1024，每头 64 维
    v: torch.Tensor  # 同上
```

写入和读取都是最简单的**切片操作**：

```python
# 写入：把当前 token 的 K/V 追加到第 pos 个位置
cache.k[:, :, pos:pos+1, :] = k

# 读取：拿出前 pos+1 个有效位置的 K/V
k_full = cache.k[:, :, :pos+1, :]
```

这两行代码就是 M2 KV Cache 的全部核心。写入是原地修改（不分配新内存），读取返回的是原始 tensor 的 view（零拷贝）。

一个全局的 `cur_len` 变量记录"当前已写入多少个 token"，generate loop 每步显式推进它。所有 28 层共享同一个 `cur_len`——这既是优点（简单），也是 M2 的根本限制（后面讲）。

### M2 的内存布局图

```text
单层 KV Cache tensor：shape [1, 8, 1024, 64]
                        B=1  KV头  L_max   D

  ┌─────────────────────────────────────────────────────────┐
  │ KV head 0                                               │
  │  pos: 0    1    2   ...  T_p-1  T_p  T_p+1  ...  1023  │
  │       [K₀] [K₁] [K₂] ... [Kₜₚ₋₁]  0     0    ...   0   │
  │       ├────────────────────┤  ↑                     ↑   │
  │            已写入 T_p 个     cur_len              未使用  │
  │                                                     │   │
  │ decode 第1步后：                                    │   │
  │       [K₀] [K₁] ... [Kₜₚ₋₁] [Kₜₚ]  0    0   ...   0   │
  │                              ↑                     │   │
  │                          新写入              cur_len+1   │
  │                                                     │   │
  │ KV head 1 ~ 7：同样结构                               │   │
  └─────────────────────────────────────────────────────────┘

  × 28 层 = 完整 KV Cache
```

写入 `cache.k[:, :, pos:pos+1, :] = k` 是**原地修改**——直接写入预分配内存的指定位置，零额外分配。读取 `cache.k[:, :, :cur_len, :]` 返回**view**——不拷贝数据，只是指向同一块内存的不同窗口。这两个操作的开销都是 O(1)。

### 代价：显存开销

KV Cache 的显存开销有一个精确公式：

```text
显存(bytes) = 2 × B × L × L_max × H_kv × D × dtype_bytes
              ↑K+V  ↑批   ↑层数  ↑最大长度  ↑KV头 ↑头维度
```

| max_seq_len | Qwen3-0.6B (bf16) | 说明 |
|------------|-------------------|------|
| 1024 | 58 MB | 大多数对话场景 |
| 4096 | 233 MB | 长文本 |
| 32K | 1.8 GB | 极端长上下文 |
| MHA (H_kv=16) | ×2 | GQA 已减半 |

**L_max 必须在 generate 之前由调用方指定**。这是静态预分配方案的根本约束——`T_p + N ≤ L_max`，否则 cache 越界。实践中有三种做法：固定值（如 1024，最简单）、`T_p + max_new_tokens`（零浪费）、模型最大位置编码长度（最灵活但最费显存）。

### 一次 decode 步的完整数据流

M2 完成后，一步 decode 从 generate loop 到 attention 的完整数据流（基于 M5 tag 的代码，M2 路径已统一使用 ForwardContext 架构）：

```text
generate()                   engine/engine.py
  │  adapter = SingleCacheAdapter(kv_cache)
  │  adapter.bind_kv_cache(model)          ← 一次性：把 cache tensor 绑到 28 个 Attention 层
  │
  │  ── prefill ──
  │  metadata = adapter.make_prefill_metadata(input_ids, positions)
  │  with set_forward_context(metadata):   ← 通道 B：metadata 放进全局
  │    logits = model(input_ids, positions)
  │
  │  ── decode loop ──
  │  cur_token [B,1]
  │  position = [[cur_len]]                ← 绝对位置，不是 [[0]]
  ▼
model(cur_token, positions)                model/qwen3.py
  │
  ├─ embed(cur_token) → hidden [B,1,H]
  │
  ├─ position_embeddings = rotary_emb(hidden, position_ids)  ← 只算一次，传入所有 28 层
  │
  └─ for i in range(28):                                      28 层 Transformer
       layer[i](hidden, position_embeddings)
         │
         └─ GQAAttention.forward(...)                  model/attention.py
              │
              ├─ q/k/v projection
              ├─ q_norm / k_norm / RoPE
              ├─ cache = self.kv_cache               ← 通道 A：绑定时已设好
              ├─ metadata = get_forward_context()     ← 通道 B：从全局读
              ├─ 写入：kv.k[:,:,pos:pos+1,:] = k    ← 切片追加
              ├─ 读取：full_k = kv.k[:,:,:pos+1,:]  ← 切片读历史
              └─ Attention(q, full_k, full_v) → output

generate()
  ├─ kv_cache.cur_len += 1                              ← 显式推进
  └─ 采样下一个 token → 继续循环
```

M2 虽然在架构上比 M3/M4 简单（单请求、全局 cur_len、切片操作），但在 M5 tag 的代码中它已经使用了和 M3/M4 相同的 **bind_kv_cache + ForwardContext** 模式。这意味着 attention 层的代码结构是统一的——变的只是 cache tensor 的形状和 metadata 的内容。

两个细节至关重要：

1. **position_ids 必须是绝对位置**：decode 步用 `[[cur_len]]` 而非 `[[0]]`。RoPE 的正确性依赖 q 和 k 都用绝对位置编码，否则相对距离计算全部错误。写 `[[0]]` 是沉默 bug——推理不报错，但 RoPE 失效，输出质量下降。

2. **causal mask 用 T > 1 判断**：prefill 时 T = T_p > 1，需要 causal mask（prompt 内部的因果性）；decode 时 T = 1，不需要（当前 token 只看历史，没有"未来"）。条件 `if T > 1: construct_causal_mask()` 覆盖了所有场景。

### 对照：transformers 的两种 Cache

vLLM 不是唯一的参照。HuggingFace transformers 提供了两种 Cache 实现，代表了 KV Cache 设计的两个极端：

```text
DynamicCache（默认）                     StaticCache
─────────────────                       ─────────────
每步 torch.cat 拼接新 K/V                预分配 [B, H_kv, L_max, D]
shape 动态增长                            shape 固定
优点：灵活，不知序列长度也能用              优点：零内存分配，支持 CUDA Graph
缺点：每步分配新内存 + 拷贝旧数据          缺点：需要 attention_mask 屏蔽尾部零值

inferlite M2 ≈ StaticCache              inferlite M2 用切片写入替代
（预分配 + 原地写入 + 切片读取）           index_copy_（MPS 兼容，更可读）
```

vLLM 的 KV Cache 则是完全不同的路线：不在每层分配独立 tensor，而是**全局一次性分配一整块物理内存**，所有层共享。这是 M4 PagedAttention 的前提——后面会详细讲。

### 性能：理论 600× vs 实测 7.36×

M2 把 decode 单步计算量从 O(T²) 降到 O(T)。理论上 T=600 时应该快 600 倍。实测（Mac M3 Pro, MPS, bf16）：

| prompt 长度 | M1 tok/s | M2 tok/s | 加速比 |
|-----------|---------|---------|-------|
| 32 | 13.8 | 24.9 | 1.80× |
| 128 | 9.6 | 25.7 | 2.67× |
| 512 | 3.3 | 24.1 | **7.36×** |

理论和实测差距巨大的原因是：每步 decode 的主要时间不是花在 attention 计算上，而是花在从内存读取 28 层模型权重（Qwen3-0.6B 约 1.2 GB）上。这是 M1 和 M2 共有的开销，KV Cache 无法消除。只有到了 M3 多请求共享权重读取时，KV Cache 的收益才会被进一步放大。

### M2 的根本限制

M2 的 `cur_len` 是全局唯一的整数。所有请求必须共享它——同时 prefill、同步 decode、同步结束。这不是一个服务，而是一个只能一次跑一个任务的脚本。

要支持多个请求并发，`cur_len` 不能是一个数字，必须是**每个请求独立维护**的一组数字。这就是 M3 要做的事。

---

## 2. M3：多请求槽位——推理引擎的诞生

> **设计蓝图中的位置**：回答**问题 2（多人怎么办）**——从单请求到并发服务，推理引擎真正诞生。

### 问题：为什么不能多个请求同时跑

M2 处理 3 个请求只能串行：

```python
output_a = generate(engine, prompt_a, kv_cache)   # 跑完
output_b = generate(engine, prompt_b, kv_cache)   # 再跑完
output_c = generate(engine, prompt_c, kv_cache)   # 再再跑完
```

请求 A 只生成 2 个 token 就 EOS 结束了，但 B 和 C 必须等 A 完全退出。GPU 在 A 结束到 B 开始之间是空闲的。这在生产环境不可接受：你不可能让一个 ChatGPT 用户等前一个用户聊完才能开始。

### 方案：每个请求占一个 slot，独立进退

M3 引入了三个新模块：`scheduler/`（调度器）、`BatchedKVCache`（多槽位缓存）、`engine/metrics.py`（指标采集）。其中和 KV Cache 直接相关的变化是 `BatchedKVCache`：

```python
class BatchedKVCache:
    layers: list[BatchedLayerKVCache]  # 每层 [S, H_kv, L_max, D]
    seq_lens: torch.Tensor             # [S]  — 每个 slot 独立长度
    slot_manager: SlotManager          # request_id ↔ slot_id 映射
```

和 M2 的 `KVCache` 相比，最关键的变化是 **第一维的含义**：M2 是 `batch`（同步组，所有行共享 `cur_len`），M3 是 `slot`（独立请求，每行有自己的 `seq_lens[s]`）。

`LayerKVCache` 和 `BatchedLayerKVCache` 的 tensor shape 完全相同（`[S, H_kv, L_max, D]`），但语义不同。这正是从 static batching 到 continuous batching 的本质跳跃——同样的内存空间，用不同的元数据管理方式，获得了完全不同的并发能力。

### 调度器：谁该跑

M3 引入了 `FCFSScheduler`，管理 `waiting / running / finished` 三个队列。每个 decode iteration 的边界：

1. **出队**：已完成的请求从 running 移到 finished，释放 slot
2. **入队**：waiting 中的请求被 admit，分配 slot，执行 prefill
3. **batch decode**：所有 running 请求共享一次 forward

这三个步骤在每个 iteration 重复执行，running 集合动态变化。这就是 **continuous batching**——Orca 论文（OSDI'22）提出的 iteration-level scheduling。

### M3 调度流程图

```text
                        ┌──────────────────────┐
                        │  waiting (deque)       │
                        │  [reqA, reqB, reqC]   │
                        └──────────┬───────────┘
                                   │ admit（有空 slot 时）
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                    每个 decode iteration                      │
│                                                              │
│  1. Finish:  finished 请求出队 → 释放 slot                    │
│              reqA 生成 EOS → slot 0 释放                      │
│                                                              │
│  2. Admit:   waiting 请求入队 → 分配 slot → prefill           │
│              reqC 从 waiting 取出 → slot 0 → prefill          │
│                                                              │
│  3. Decode:  所有 running 请求组 batch → 一次 forward         │
│              [reqB, reqC] → batch size = 2 → 一次 forward    │
│                                                              │
│  4. Update:  seq_len += 1, 检查 EOS / max_tokens             │
└──────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │  running (dict)              │
                    │  {reqB: slot1, reqC: slot0} │
                    └──────────────────────────────┘
```

### inferlite M3 vs nano-vllm 调度器对照

| 维度 | inferlite M3 | nano-vllm |
|------|-------------|-----------|
| 调度策略 | FCFS（先来先服务） | token-budget（prefill 优先） |
| 队列 | waiting / running / finished | waiting / running（finished 不单独存） |
| prefill 方式 | 逐条 prefill（B=1） | batched prefill + chunked prefill |
| decode 方式 | batched | batched |
| 抢占 | ❌（容量不够就停止 admit） | ✅（空间不足时踢回 waiting） |
| prefix cache | ❌ | ✅（can_allocate 返回 num_cached） |
| token budget | ❌ | ✅（max_num_batched_tokens 限制单轮总量） |

nano-vllm 的 **preemption** 机制值得注意：当 decode 阶段某个请求需要新 block 但 free pool 为空时，调度器会把 running 队列尾部的请求踢回 waiting（释放它的 block），让当前请求继续。被 preempt 的请求下次重新 prefill。inferlite 不做这个——这是教学简化，preemption 引入的"状态回退 + 重新 prefill"复杂度会模糊 continuous batching 的核心概念。

### Attention 读写的变化

M2 的切片操作无法表达"每行写不同位置"的语义。M3 不得不改用 for-loop + fancy-index：

```python
# 写入：每个请求写自己 slot 的当前位置
for i in range(B):
    cache.k[slot_i, :, pos_i, :] = k[i]

# 读取：按 slot 列表 gather
k = cache.k[cache_slots, :, :max_len, :]
```

M2 用切片是零拷贝 view，M3 的 fancy-index 会产生内存拷贝。这是 independent progress 的固有代价——不同请求在不同位置、不同长度，无法用一次切片操作表达。

### 性能：一个意外

M3 在 MPS 上的实测结果让人意外：batch throughput 只有 serial 的 0.38~0.44×。也就是说，**多个请求一起跑反而更慢了**。

分段计时定位了根因：

| 段 | 占比 | 说明 |
|---|---|---|
| for-loop 写 cache | **63%** | 28 层 × N 步 = 上千次 Python for-loop + slice assign |
| fancy-index gather | 22% | 每次 gather 产生一次内存拷贝 |
| `.item()` 同步 | 15% | tensor.item() 触发 CPU-GPU 同步 |

这不是 continuous batching 本身的问题。nano-vllm 用几乎相同的 Python 代码量（~1200 行）实现了接近 vLLM 的性能，原因是它在底层调用了 **Triton kernel**（`store_kvcache_kernel`）和 **FlashAttention**，不是纯 PyTorch。

inferlite M3 的性能瓶颈是"纯 PyTorch + 不调 kernel"这个路线选择的固有代价。M9 会用 Triton kernel 系统解决这个问题，但那不是 M3 的目标。M3 的目标是**语义正确**：多个请求能独立进退、slot 能跨请求复用、batch 输出与 serial 输出 token 级一致。这些全部验证通过了。

### M3 的根本限制

M3 的每个 slot 预分配了 `max_seq_len` 的连续空间。如果 max_seq_len=1024，一个请求实际只生成了 50 个 token，剩下 974 个位置的空间被白白浪费。4 个并发请求各用 10% 容量时，内存利用率只有 10%。

这是**内碎片**：每个 slot 内部有大量未使用的空间。消除它需要把 KV Cache 从连续数组改成分页管理——就像操作系统把内存从连续分配改成虚拟内存一样。这就是 M4 要做的事。

---

## 3. M4：PagedAttention——把显存当虚拟内存管

> **设计蓝图中的位置**：回答**问题 3（内存碎片怎么消除）+ 问题 5（架构解耦）**——分页管理 + ForwardContext/CacheAdapter 统一架构。

### 问题：连续分配的浪费

M3 的 KV Cache 是一张二维表：每行是一个 slot，每行固定 `max_seq_len` 个位置。短请求浪费了行尾的大量空间，而且不同 slot 之间的物理内存不连续，无法共享。

```text
M3 内存布局（4 个 slot，max_seq_len=1024）：

slot 0: [████░░░░░░░░░░░░░░░░░░░░]   用了 100/1024 = 9.8%
slot 1: [██░░░░░░░░░░░░░░░░░░░░░░]   用了  20/1024 = 2.0%
slot 2: [████████░░░░░░░░░░░░░░░░]   用了 200/1024 = 19.5%
slot 3: [█░░░░░░░░░░░░░░░░░░░░░░░]   用了  10/1024 = 1.0%
                                     总利用率 ≈ 8%
```

vLLM 的论文（Kwon et al., 2023）提出了 PagedAttention：把 KV Cache 从连续数组改成**分页块**管理，就像操作系统的虚拟内存——逻辑地址连续，物理地址可以分散。

### 方案：block pool + block table + scatter/gather

M4 引入了三个核心组件：

```text
BlockPool     →  物理 block 的分配和释放（全局只有一个）
BlockTable    →  每个请求的逻辑→物理映射（每请求一个）
PagedKVCache  →  每层 K/V tensor + scatter/gather 操作（全局只有一个）
```

KV tensor 的物理布局从 `[S, H_kv, L_max, D]` 变成了 `[num_blocks, block_size, H_kv, D]`——不再和请求绑定，而是一个全局共享的物理块池。每个 block 只存 `block_size` 个 token 的 K/V（典型值 16 或 32）。

每个请求有自己的 `block_table`：一个 `list[int]`，记录这个请求的逻辑 block 0/1/2/... 分别对应哪个物理 block。

```text
M4 内存布局（8 个物理 block，block_size=16）：

请求 A（20 tokens）:  block_table = [3, 1]
  物理 block 3: [A 的 token 0~15]     ← 满
  物理 block 1: [A 的 token 16~19]    ← 部分填充

请求 B（8 tokens）:   block_table = [2]
  物理 block 2: [B 的 token 0~7]      ← 部分填充

空闲 block: 0, 4, 5, 6, 7             ← 按需分配
```

内存浪费从 M3 的"每个请求浪费 max_seq_len - actual_len"缩减为"最多浪费 block_size - 1 个 token"。

### M4 Block Table 详解图

```text
                        逻辑视角（请求看到的）
                        ─────────────────────
请求 A（20 tokens）:     [block 0: 16 tok] [block 1: 4 tok]
请求 B（8 tokens）:      [block 0: 8 tok]

                        block_table 映射
                        ─────────────────
请求 A: block_table = [3, 1]     逻辑 block 0 → 物理 block 3
                                 逻辑 block 1 → 物理 block 1

请求 B: block_table = [2]        逻辑 block 0 → 物理 block 2

                        物理视角（内存中实际的）
                        ────────────────────────
物理 block:  0     1     2     3     4     5     6     7
            [空]  [A₁]  [B₀]  [A₀]  [空]  [空]  [空]  [空]
                  ↑4tok  ↑8tok ↑16tok

K tensor shape: [8, 16, 8, 64]
                 ↑  ↑   ↑  ↑
             blocks bs KV头 D

读取请求 A 的 KV：
  block_table = [3, 1]
  k = layer.k[[3, 1]]               → shape [2, 16, 8, 64]
  k = k.reshape(1, 32, 8, 64)       → shape [1, 32, 8, 64]
  k = k[:, :20, :, :]               → 只取前 20 个有效 token
  k = k.masked_fill(pos >= 20, 0)   → NaN 安全：清零 12 个无效位置
```

### inferlite M4 vs nano-vllm BlockManager 对照

| 维度 | inferlite M4 | nano-vllm BlockManager |
|------|-------------|----------------------|
| Block 元数据 | `Block(id, ref_count)` | `Block(id, ref_count, hash, token_ids)` |
| 空闲管理 | `free_block_ids: deque` | `free_block_ids: deque` + `used_block_ids: set` |
| prefix cache | ❌（留 M5） | ✅（`can_allocate` 查 hash，返回 num_cached） |
| CoW | ❌（留 M5） | ❌（nano-vllm 不做 CoW） |
| 写入方式 | PyTorch scatter（`flat_cache[slot_mapping] = flat_k`） | Triton kernel（`store_kvcache_kernel`） |
| 读取方式 | PyTorch gather + masked_fill | FlashAttention（kernel 内部按 block_table 读） |
| hash 注册 | ❌ | ✅（`hash_blocks` 在 postprocess 中调用） |

nano-vllm 的 BlockManager 把 PagedAttention 和 Prefix Cache **融合在一起**——`can_allocate()` 同时做容量检查和 prefix cache 查询。inferlite 把它们拆成 M4（分页）和 M5（前缀缓存）两个独立里程碑，每步只解决一个问题。这是教学框架的优势：你可以先理解分页内存管理，再叠加前缀复用，而不是同时面对两个概念。

### 读写方式的变化

M3 用 for-loop 写单个 slot、用 fancy-index 读多个 slot。M4 需要把数据**scatter**到不连续的物理 block 中，再从 block table 指定的多个物理 block 中 **gather** 回来：

```python
# 写入（scatter）：把 padded batch 的 K/V 分散到物理 block
flat_cache_k[slot_mapping] = flat_k

# 读取（gather）：按 block table 拼回连续序列
gathered_k = layer_cache.k[block_table]     # [B, max_blocks, bs, H_kv, D]
k = gathered_k.reshape(B, max_blocks * bs, H_kv, D)
```

`slot_mapping` 是 scatter 的目标索引：每个 token 对应一个物理 slot（= block_id × block_size + offset），由 block table 计算得出。

### NaN 安全：一个隐藏坑

物理 block 用 `torch.empty` 创建（不为零值初始化浪费计算时间）。未写入的位置含有垃圾值——可能是 NaN 或 Inf。

仅对 attention score 做 mask 是不够的：`softmax` 把无效位置的 score 压到 0，但 `0 × NaN = NaN`——NaN 会通过 value matmul 传播到最终输出。必须在 gather 之后、attention 计算之前，用 `valid_lens` 显式清零无效的 K/V：

```python
invalid = ~(positions[None, :] < valid_lens[:, None])
k = k.masked_fill(invalid, 0)
v = v.masked_fill(invalid, 0)
```

这是 M3 后期发现的一个隐藏 bug，M4 从一开始就把 NaN 安全作为设计约束。score mask 管语义（哪些 token 可以互相 attend），K/V 清零管数值安全（防止垃圾值污染输出）。两者缺一不可。

### M4 最重要的架构贡献：ForwardContext + CacheAdapter

M4 除了分页内存管理，还引入了一套架构模式，彻底改变了整个项目的代码组织。**在"整体调用架构"章节的分③和分④中已经详细讲过这两个机制的代码实现**，这里只总结它们在 M4 中解决的问题：

- **ForwardContext**：metadata 不再经过模型参数传递，而是通过全局 context manager 设置。模型的 forward 签名从 M3 的 5 个参数简化为 2 个（`input_ids` + `positions`）。attention 层自己从全局 context 决定怎么读写 cache。
- **CacheAdapter Protocol**：三种完全不同的 cache 实现（Static/Slot/Paged），通过统一的 8 方法接口被 engine 使用。loop.py 只通过这个接口操作 cache，不关心底层是 slot 还是 block。M3 的主循环**直接复用**于 M4——区别仅在于传入的 adapter 实现不同。

这次架构重构消除了 M3/M4 之间约 80% 的重复代码。engine 目录从 7 个文件精简到 4 个（context / engine / loop / metrics），attention.py 保持单一文件。三种 adapter 构造的 metadata 格式不同（M2 只含 `cur_len`，M3 多了 `slot_mapping`，M4 用 `block_table`），但传递机制和 attention 的调用方式不变——这就是架构解耦的效果。

### M4 的根本限制

M4 解决了内存碎片，但每个请求的 block 都是独立分配的。如果两个请求有完全相同的 system prompt（在实际服务中极其常见），它们各自计算一遍相同的 KV，然后存进各自的物理 block 中。这些 KV 数据完全一样，却占了两份空间。

把已缓存的 KV 数据变成**可复用的资产**——这就是 M5 要做的事。

---

## 4. M5：Prefix Caching——让 KV 变成可复用资产

> **设计蓝图中的位置**：回答**问题 4（相同计算为什么要做两遍）**——前缀复用，让已缓存的 KV 成为可共享的资产。

### 问题：相同前缀为什么要算两遍

在实际的 LLM 服务中，大量请求共享公共前缀：

- **多轮对话**：第二轮的 prompt 包含第一轮的完整历史
- **System prompt**：所有请求可能共享同一段 system instruction
- **Few-shot examples**：相同的示例前缀

M4 的 PagedAttention 解决了内存碎片，但两个有相同前缀的请求，各自的 block 是独立分配、独立计算的。前缀部分的 KV 数据完全一样，却做了两次 forward、占了两份内存。

Prefix Caching 的思想是：**如果新请求的前缀已经在缓存中存在，直接复用那些物理 block 和对应的 KV 数据，跳过这部分 prefill 计算。**

### 方案：chain hash + LRU + CoW

M5 在 M4 的基础上引入了三个核心机制：

**Chain Hash**：对每个填满的 block 计算一个 hash 值，这个 hash 依赖于前面所有 block 的 hash（链式结构）。

```text
block 0: h₀ = xxhash64(tokens[0:bs], prefix=-1)
block 1: h₁ = xxhash64(tokens[bs:2bs], prefix=h₀)
block 2: h₂ = xxhash64(tokens[2bs:3bs], prefix=h₁)
```

chain hash 的位置唯一性至关重要：如果只做 `hash(token_ids)`，那么 `[1,2,3,4]` 出现在 block 0 和出现在 block 3 会产生相同的 hash，导致错误的前缀匹配。chain hash 把"在哪个位置"编码进了 hash 值——前缀不同则 hash 不同。

**LRU 淘汰**：当一个 block 不再被任何请求使用（ref_count=0）但已有 hash 时，它不是直接归还空闲池，而是进入一个 LRU 队列。新请求到来时查 hash 索引：命中了就直接 touch（ref++，从 LRU 中取出）；没命中就正常分配新 block。空闲池不够时，淘汰 LRU 队首（最久未用的 cached block）。

```text
BlockPool 三容器：
├── free_block_ids: deque[int]           真正空闲（ref=0, hash=-1）
├── cached_block_lru: OrderedDict        ref=0, 有 hash（LRU 队尾=最近使用）
└── hash_to_block_id: dict[int, int]     chain hash → block_id 索引

block 的生命周期：
free_pool ──allocate()──► running (ref≥1) ──free(),hash≠-1──► cached_lru
    ▲                          │                                    │
    │                          │ free(),hash=-1                     │ allocate() 淘汰
    └──── free_pool ◄──────────┘                                    │
    └──── free_pool ◄──────────────────────────────────────────────┘
```

**CoW（Copy-on-Write）**：当多个请求共享同一个 block（ref_count > 1），某个请求需要写入时（block 尚未填满，decode 阶段追加 token），必须先 clone 一个独占副本。否则写入会污染其他请求看到的 KV 数据。

```python
# 写入前检查：这个 block 是不是共享的？
if ref_count[block_id] > 1:
    new_bid = allocate()                    # 分配新 block
    copy_kv(old_bid, new_bid)               # clone K/V tensor
    migrate_hash(old_bid, new_bid)          # hash 索引指向新 block
    dec_ref(old_bid)                        # 原 block ref-1
    block_table[block_idx] = new_bid        # 当前请求改用新 block
```

### 一次 prefix cache 命中的完整流程

假设新请求的 prompt 有 12 个 token，block_size=4：

```text
Step 1: 容量检查 → 需要 ceil(12/4) = 3 个 block

Step 2: 查 chain hash
  block 0: hash([t0,t1,t2,t3], prefix=-1) → h₀ → 命中 block_5！
  block 1: hash([t4,t5,t6,t7], prefix=h₀) → h₁ → 命中 block_8！
  block 2: hash([t8,t9,t10,t11], prefix=h₁) → h₂ → 未命中

  结果：num_cached = 2（命中 2 个 block）

Step 3: cache-aware 分配
  touch(block_5) → ref++, 从 LRU 取出
  touch(block_8) → ref++, 从 LRU 取出
  allocate() → block_12（新分配）

  最终 block_table = [5, 8, 12]
  block 5 和 8 的 KV 数据已经是正确的，只需对 block 12 做 prefill
```

第二个请求的 prefill 计算量只有 1/3（只需算 block 12 对应的 4 个 token）。在多轮对话场景中，第二轮的 prompt 几乎完全命中第一轮的前缀，TTFT（Time To First Token）可以大幅下降。

### 与 vLLM V1 的简化

inferlite M5 是有意的教学简化版。和 vLLM V1 的对比：

| 维度 | inferlite M5 | vLLM V1 |
|------|-------------|---------|
| hash 索引 | `dict[int, int]`（一对一） | 树结构（一对多） |
| LRU | `OrderedDict`（标准库） | 双向链表（自定义） |
| CoW | 同步（写入前立即 clone） | 异步批量执行 |
| 孤儿 block | LRU 自然淘汰 | 树结构级联淘汰 |
| skip prefill | 未实现 | 跳过已缓存 token 的计算 |

每一处简化都对应一个明确的工程权衡：教学版选择了最简单的实现，保留了核心机制的完整性。vLLM 的每个"更优方案"都是在生产环境中被逼出来的优化——树结构处理孤儿 block、异步 CoW 减少延迟、skip prefill 真正跳过计算。这些优化可以作为后续里程碑的改进方向。

### M5 Chain Hash 全流程图

```text
                        Chain Hash 注册与查询流程
                        ═══════════════════════

  请求 A: prompt = [1, 2, 3, 4, 5, 6, 7, 8], block_size=4

  ┌─── prefill 完成后，loop.py 调 hash_blocks() ───────────────────┐
  │                                                                │
  │  block 0: tokens=[1,2,3,4]                                     │
  │    h₀ = xxhash64(prefix=-1, data=[1,2,3,4])  = 0x3a7f          │
  │    block_table[0].hash = h₀                                    │
  │    hash_to_block_id[h₀] = block_table[0]                       │
  │                                                                │
  │  block 1: tokens=[5,6,7,8]                                     │
  │    h₁ = xxhash64(prefix=h₀, data=[5,6,7,8])  = 0xb2c1         │
  │    block_table[1].hash = h₁                                    │
  │    hash_to_block_id[h₁] = block_table[1]                       │
  │                                                                │
  │  注意：h₁ 依赖 h₀，h₀ 依赖 prefix=-1                           │
  │  → 相同 [5,6,7,8] 在不同前缀下产生不同 hash                      │
  └────────────────────────────────────────────────────────────────┘

  请求 A 完成后释放，2 个 block 进入 LRU（ref=0，有 hash）

  ┌─── 请求 B 到来，prompt = [1, 2, 3, 4, 9, 10] ────────────────┐
  │                                                               │
  │  can_admit_with_cache():                                      │
  │  block 0: hash([1,2,3,4], -1) = 0x3a7f                        │
  │    → hash_to_block_id[0x3a7f] = 命中！touch(block, ref++)      │
  │  block 1: hash([9,10,...], 0x3a7f) ≠ 任何已注册 hash           │
  │    → 未命中，停止。num_cached = 1                               │
  │                                                               │
  │  allocate_with_cache(num_cached=1):                            │
  │    block_table = [命中的 block_5, 新分配的 block_12]             │
  │    block 5 的 KV 数据已经正确，只需 prefill block 12            │
  └───────────────────────────────────────────────────────────────┘

  ┌─── CoW 场景 ──────────────────────────────────────────────────┐
  │                                                               │
  │  请求 C 命中 block_5（ref=2，和请求 D 共享）                      │
  │  decode 阶段 C 要写入 block_5 的剩余位置：                       │
  │                                                               │
  │  cow_if_shared(C, block_idx=0):                               │
  │    ref_count[5] = 2 > 1 → 需要 CoW                            │
  │    new_bid = allocate() → block_15                             │
  │    K[15] = K[5].clone()     ← 拷贝物理 tensor                  │
  │    V[15] = V[5].clone()                                        │
  │    hash_to_block_id[h₀] = 15  ← hash 索引迁移                  │
  │    dec_ref(5) → ref=1（D 仍持有旧 block）                       │
  │    C.block_table[0] = 15      ← C 改用独占副本                  │
  └───────────────────────────────────────────────────────────────┘
```

### inferlite M5 vs nano-vllm hash_blocks 对照

| 维度 | inferlite M5 | nano-vllm BlockManager |
|------|-------------|----------------------|
| hash 算法 | `xxhash64(prefix_bytes + struct_pack(token_ids))` | `xxhash64(prefix_bytes + numpy_tobytes(token_ids))` |
| LRU 结构 | `OrderedDict`（标准库，O(1) touch/popitem） | ❌ 没有 LRU——free pool 直接回收 |
| 淘汰策略 | free pool → LRU 淘汰 → error | 只有 free pool，不够就 error |
| CoW | ✅ `cow_if_shared()` 同步 clone K/V tensor | ❌ 不做 CoW |
| hash 注册时机 | loop.py 中 prefill 后 + 每步 decode 后调 `hash_blocks()` | `postprocess()` 中调 `hash_blocks()` |
| hash 起始位置 | 遍历 block_table，跳过 `block.hash != -1` 的已注册 block | 用 `num_cached_tokens // block_size` 精确定位起始 block |
| 孤儿 block | 不处理（LRU 自然淘汰） | 不处理（直接回 free pool） |
| `can_allocate` 双重检查 | 先 `can_admit()`（容量），再 `can_admit_with_cache()`（hash） | 单方法 `can_allocate()` 同时做 hash 查询 + 容量检查 |
| token_ids 碰撞保护 | ❌（hash 相等即认为命中） | ✅（`block.token_ids != token_ids` 时 break） |

**nano-vllm 的两个值得注意的设计**：

1. **token_ids 碰撞保护**（第 66 行）：即使 hash 相等，还要比对 `token_ids` 是否完全一致。理论上 xxhash64 碰撞概率极低，但在大规模生产中这是一个防御性措施。inferlite M5 省略了这一步——教学场景下 hash 碰撞不是主要风险。

2. **没有 LRU**：nano-vllm 的 `deallocate()` 直接把 ref=0 的 block 归还 free pool，即使它有 hash。这意味着 prefix cache 的"缓存"生命周期很短——只有在上一个请求释放、下一个请求到来之间的窗口期内有效。inferlite M5 引入了 LRU，让 cached block 有更长的生存期，在多轮对话等场景中更实用。

### hash_blocks 的注册时机

chain hash 不是自动计算的——它需要在正确的时刻被显式调用。`hash_blocks()` 在 loop.py 中被两个时刻触发：

**Prefill 后**：prompt 中填满的 block 立即注册 hash。

```python
# loop.py — prefill 循环
for req in admitted:
    logits = model(req.prompt_ids, ...)
    adapter.cache.hash_blocks(req.request_id, prompt_token_ids)
    # 为 prompt 中所有填满的 block 计算 chain hash 并注册
```

**Decode 每步后**：新 token 可能让一个 block 从"未满"变成"填满"，此时注册。

```python
# loop.py — decode 循环
for req, tok in zip(running, sampled):
    req.seq_len += 1
    all_ids = req.prompt_ids + req.generated_tokens
    adapter.cache.hash_blocks(req.request_id, all_ids)
    # 内部跳过 block.hash != -1 的已注册 block，只注册新填满的
```

**跳过已注册**是关键优化：`hash_blocks()` 内部检查 `block.hash != -1`，已注册的 block 直接用已有 hash 继续链式计算，不重复注册。这意味着 decode 第 k 步只需要检查最后 1~2 个 block，而不是全部。

### cache-aware 的 admit 流程

M5 对 loop.py 的 admit 流程做了三处改动，让调度器在分配 block 时优先复用 cached block：

```python
# M4 流程（无前缀缓存）
if not adapter.can_admit(prompt_len):    # 容量检查
    break
adapter.allocate(req.request_id, prompt_len)  # 直接分配新 block

# M5 cache-aware 流程
if not adapter.can_admit(prompt_len):           # ① 容量检查（不变）
    break
num_cached = adapter.can_admit_with_cache(prompt_ids)  # ② 查 chain hash
if num_cached == -1:
    break  # 容量不够
if num_cached > 0:
    adapter.allocate_with_cache(req_id, prompt_ids, num_cached)  # ③ cache-aware 分配
else:
    adapter.allocate(req.request_id, prompt_len)  # 回退到 M4 路径
```

`can_admit_with_cache()` 从 block 0 开始逐个查 chain hash，遇到第一个未命中就停止——这正是 chain hash 的前缀匹配特性：block 0 不匹配则后续都不可能匹配。M2/M3 的 adapter 的 `can_admit_with_cache()` 返回 0（no-op），确保不影响已有路径。

---

## 5. 工业框架怎么做：vLLM 与 nano-vllm 深度拆解

> 看完 inferlite M2-M5 的实现后，再回头看工业界的标准答案。理解 vLLM 和 nano-vllm 的完整架构后，才能看清 inferlite 每一步"学了什么、简化了什么"。

### vLLM V1 的五层架构

vLLM 是当前最主流的开源 LLM 推理引擎。它的 V1 架构（2024 年重构后）可以拆成五层：

```text
┌─────────────────────────────────────────────────────────────────────┐
│  L5  Entrypoint          OpenAI-compatible API Server (FastAPI)     │
├─────────────────────────────────────────────────────────────────────┤
│  L4  LLMEngine           入口：接收请求，驱动 step 循环              │
├─────────────────────────────────────────────────────────────────────┤
│  L3  Scheduler           waiting/running 队列 + token-budget 调度    │
│      ├ KVCacheManager    block 分配/释放/prefix-cache/CoW           │
│      └ OutputProcessor   采样 + detokenize + 流式输出               │
├─────────────────────────────────────────────────────────────────────┤
│  L2  ModelRunner         构造 input tensors → 调用 model forward    │
│      ├ ForwardContext    metadata 全局上下文（slot/block/seq_lens）   │
│      └ CUDA Graph        decode 阶段捕获计算图，消除 kernel launch   │
├─────────────────────────────────────────────────────────────────────┤
│  L1  Model               nn.Module：Qwen3/Llama 等模型实现          │
│      └ Attention         Triton kernel 写 cache + FlashAttention 读 │
└─────────────────────────────────────────────────────────────────────┘
```

每层的职责边界极其清晰：**Scheduler 管"谁该跑"，ModelRunner 管"怎么跑"，Model 管"怎么算"**。KV Cache 的管理分散在三层：L3 管 block 的元数据（分配/释放/hash），L2 管 metadata 的传递（通过 ForwardContext），L1 管物理 tensor 的读写（通过 Triton kernel）。

### nano-vllm：1200 行的 vLLM 教学版

nano-vllm 是一个极其精炼的 vLLM 实现——**1200 行 Python 代码，但底层调用了 Triton kernel 和 FlashAttention**。它保留了 vLLM 的完整架构骨架，砍掉了所有生产级复杂度。理解 nano-vllm 是理解 inferlite 定位的最佳参照。

**nano-vllm 的核心模块**（~1200 行）：

```text
nanovllm/
├── engine/
│   ├── sequence.py (84L)       Sequence 数据类：token_ids + block_table + status
│   ├── scheduler.py (93L)      调度器：waiting/running + token-budget + preemption
│   ├── block_manager.py (121L) block 管理：分配/释放/hash/prefix-cache
│   ├── model_runner.py (258L)  tensor 准备 + CUDA Graph + 多 GPU TP
│   └── llm_engine.py (91L)     顶层引擎：add_request → step → generate
├── layers/
│   └── attention.py (76L)      Triton kernel 写 cache + FlashAttention 读
├── models/
│   └── qwen3.py                Qwen3 模型实现
└── utils/
    └── context.py (28L)        全局 Context（metadata 传递）
```

### nano-vllm 的调度器：token-budget + preemption

nano-vllm 的 `scheduler.schedule()` 比 inferlite M3 的 FCFS 复杂得多。它实现了 vLLM V1 的完整调度策略：

```text
schedule() 的决策流程：

1. 先尝试 prefill（优先调度 decode 是 vLLM V1 的策略，nano-vllm 先试 prefill）
   ├── 检查 token budget：remaining = max_num_batched_tokens - 已调度
   ├── 检查 prefix cache：can_allocate(seq) → num_cached_blocks
   ├── 检查 chunked prefill：只有第一个 seq 允许 chunked
   └── 通过 → 分配 block，移入 running

2. prefill 无请求 → 尝试 decode
   ├── 检查能否 append（当前 block 满时需要新 block）
   ├── 空间不够 → preempt：踢掉 running 尾部请求回 waiting
   └── 通过 → num_scheduled_tokens = 1
```

**Preemption**（前面 M3 对比表中提过）是 vLLM V1 的安全网：block 耗尽时踢回低优先级请求，保证系统不崩溃。nano-vllm 保留了这个机制，inferlite 没有。

### nano-vllm 的 BlockManager：prefix cache + hash 一体化

nano-vllm 的 `BlockManager` 把 PagedAttention 的 block 管理和 Prefix Cache 的 hash 机制**融合在同一个类中**——不像 inferlite 分成 M4 和 M5 两个里程碑：

```python
# nano-vllm: block_manager.py 核心结构
class BlockManager:
    blocks: list[Block]              # 所有物理 block 的元数据
    free_block_ids: deque[int]       # 空闲 block（含 cached block）
    used_block_ids: set[int]         # 正在被使用的 block
    hash_to_block_id: dict[int, int] # chain hash → block_id

    def can_allocate(self, seq) -> int:
        """返回 prefix cache 命中的 block 数（-1 表示容量不足）"""

    def allocate(self, seq, num_cached_blocks):
        """分配 block table：前 num_cached_blocks 个复用，后续新分配"""

    def hash_blocks(self, seq):
        """prefill/decode 后，为新填满的 block 注册 chain hash"""

    def deallocate(self, seq):
        """释放所有 block：ref_count--，为 0 时归还 free pool"""
```

**关键区别**：nano-vllm 没有独立的 LRU 队列。cached block（ref=0 但有 hash）和真正空闲的 block 混在同一个 `free_block_ids` deque 中。当需要分配时，cached block 和空 block 同等对待——被分配后 hash 被清除。这意味着 nano-vllm 的 prefix cache 没有淘汰策略，cached block 一直保留直到被自然分配覆盖。

inferlite M5 增加了 OrderedDict LRU，让最近使用的 cached block 最后被淘汰——这是一个有意义的改进。

### nano-vllm 的 Attention：Triton kernel + FlashAttention

nano-vllm 的 `attention.py`（76 行）是整个项目中信息密度最高的文件：

```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()

        # Step 1: 用 Triton kernel 把 k/v 写入物理 cache
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # Step 2: 用 FlashAttention 做 attention 计算
        if context.is_prefill:
            o = flash_attn_varlen_func(q, k_cache, v_cache,
                cu_seqlens_q, cu_seqlens_k, block_table=...)
        else:
            o = flash_attn_with_kvcache(q, k_cache, v_cache,
                cache_seqlens=..., block_table=...)
```

这段代码揭示了 **inferlite 性能瓶颈的根本原因**：

| 操作 | nano-vllm | inferlite |
|------|-----------|-----------|
| 写 cache | Triton kernel：一次 launch 写所有 token 到所有物理 slot | Python for-loop：28 层 × N 步 = 上千次 slice assign |
| 读 cache | FlashAttention：kernel 内部按 block_table gather，不 materialize 完整 tensor | PyTorch fancy-index：materialize 到临时 tensor，产生内存拷贝 |
| Attention | FlashAttention：fused softmax + matmul，O(1) HBM 访问 | PyTorch matmul + 手动 softmax：多次 HBM 往返 |

**nano-vllm 看起来只有 1200 行 Python，但它调用的是经过高度优化的 CUDA kernel。** 这就是为什么 nano-vllm 的吞吐量接近 vLLM（官方 benchmark：1434 vs 1362 tok/s）——Python 层只是调度逻辑，真正的计算全在 kernel 里。

### nano-vllm 的 KV Cache 物理布局

nano-vllm 的 KV Cache 物理布局和 inferlite M4/M5 高度相似：

```text
nano-vllm（全局预分配）：
kv_cache.shape = [2, L, num_blocks, block_size, n_kv, D]
                  ↑  ↑  ↑           ↑           ↑     ↑
                 K+V 层 block数    每block       KV头  头维度
                              token数

inferlite M4/M5（全局预分配）：
layer.k.shape = [num_blocks, block_size, n_kv, D]
                 ↑           ↑           ↑     ↑
              block数    每block       KV头   头维度
                         token数

差异：nano-vllm 把 K/V 和所有层合在一个 tensor 里（一次性分配）；
      inferlite 每层独立 tensor（M4 PagedKVCache.layers[i].k/v）。
      逻辑等价，nano-vllm 的做法对 CUDA 内存管理更友好。
```

### nano-vllm 的 Context：metadata 传递

nano-vllm 的 `Context`（28 行）是 inferlite `ForwardContext` 的直接灵感来源：

```python
# nano-vllm: context.py
@dataclass(slots=True)
class Context:
    is_prefill: bool
    cu_seqlens_q: Tensor | None    # 变长 batch 的累积长度（FlashAttention 需要）
    cu_seqlens_k: Tensor | None
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: Tensor | None    # scatter 写入的目标 slot
    context_lens: Tensor | None    # 每请求的有效 KV 长度
    block_tables: Tensor | None    # block table 矩阵
```

inferlite 的 `AttentionMetadata` 是这个结构的精简版——去掉了 `cu_seqlens`（inferlite 不用 FlashAttention）和 `max_seqlen`（inferlite 的 attention 不需要提前知道最大长度）。保留了核心的 `slot_mapping`、`seq_lens`、`block_table`。

### nano-vllm 的 CUDA Graph

nano-vllm 在 decode 阶段使用 CUDA Graph 消除 kernel launch 开销：

```text
启动时：
  for bs in [1, 2, 4, 8, 16, 32, ...]:
    1. 构造固定大小的 dummy tensors
    2. warmup：跑一次 forward
    3. capture：torch.cuda.graph(graph) 记录所有 kernel 调用
    4. 保存 graph

运行时（decode）：
  1. 找到 ≥ 当前 batch size 的最小预捕获 graph
  2. 把真实数据 copy 到 dummy tensors
  3. graph.replay()：重放所有 kernel，零 launch 开销
```

CUDA Graph 对 decode 特别有效：decode 每步的计算图完全固定（1 个 token 输入，shape 不变），只有数据不同。捕获后 replay 比每次重新 launch kernel 快得多——尤其在小 batch 时，kernel launch 开销占比很高。

inferlite 不做 CUDA Graph（Mac MPS 不支持）。这是教学框架和生产框架的又一个差异点。

### 小结

§5 从 vLLM V1 五层架构、nano-vllm 1200 行骨架，一路拆到 Attention kernel、物理布局、Context 传递和 CUDA Graph。三者的完整对照表和定位差异，留到 §6 统一展开。

---

## 6. 回看整条演进线

把 M1 到 M5 放在一起看，会看到一条清晰的逻辑链：

```text
M1  每步重算所有 K/V          →  O(N·T²)  不可用
M2  缓存 K/V，切片读写        →  O(N·T)   单请求可用
M3  多请求独立 slot           →  并发服务  但有内碎片
M4  分页 block 管理           →  消除碎片  但前缀不复用
M5  chain hash + CoW         →  前缀复用  接近生产级
```

每一步解决的都是上一步的**根本瓶颈**，而不是叠加优化。这条线和工业界 LLM 推理引擎的演进历史高度吻合：从 transformers 的 DynamicCache（对应 M2），到 Orca 的 continuous batching（M3），到 vLLM 的 PagedAttention（M4），到 SGLang/vLLM V1 的 prefix cache（M5）。

### 三方架构全景对照

把 vLLM V1、nano-vllm 和 inferlite M5 放在一起看，能看到三种不同定位下的架构选择：

| 维度 | vLLM V1（生产级） | nano-vllm（教学+性能） | inferlite M5（教学+机制） |
|------|-------------------|----------------------|--------------------------|
| **代码规模** | ~50,000+ 行 | ~1,200 行 | ~2,800 行 |
| **KV 写入** | Triton kernel（GPU fused） | Triton kernel（`store_kvcache_kernel`） | PyTorch slice assign（纯 CPU/MPS） |
| **KV 读取** | FlashAttention kernel | FlashAttention kernel | PyTorch gather + masked_fill |
| **调度策略** | token-budget + preemption | token-budget + preemption | FCFS + slot 队列 |
| **Block 管理** | 五层架构分散管理 | BlockManager 一体化 | BlockPool + PagedKVCache 分离 |
| **Prefix Cache** | 树结构 + 1:N hash + 级联淘汰 | dict + hash_blocks（无 LRU） | dict + LRU（OrderedDict） |
| **CoW** | 异步批量 `_pending_cow_copies` | ❌ 不做 | 同步 `cow_if_shared()` |
| **CUDA Graph** | ✅ decode 阶段捕获 | ✅ decode 阶段捕获 | ❌（M9 规划） |
| **metadata 传递** | ForwardContext（全局 dataclass） | Context（全局 dataclass） | ForwardContext + CacheAdapter Protocol |
| **skip prefill** | ✅ `num_computed_tokens` | ✅ `num_cached_tokens` | ❌ defer |
| **hash 碰撞保护** | ✅ | ✅（token_ids 比对） | ❌ |
| **硬件依赖** | 必须 NVIDIA GPU + CUDA | 必须 NVIDIA GPU + CUDA | Mac MPS / CPU（无 GPU 依赖） |

**三种定位的本质区别**：

- **vLLM V1** 是生产引擎：每层都做了性能极致优化（Triton kernel、FlashAttention、CUDA Graph、异步 CoW），代价是复杂度爆炸——一个 PagedAttention 涉及 5 层代码协作。
- **nano-vllm** 是"1200 行调用 Triton"：架构极其精炼，但性能不输 vLLM 太多，因为计算密集部分全在 Triton/FlashAttention kernel 里。它的教学价值在于让你理解 vLLM 的骨架；它的局限在于你无法看到 kernel 内部的实现。
- **inferlite** 是"2800 行纯 PyTorch"：所有 cache 操作都是 PyTorch 原生 tensor 操作（slice assign、gather、scatter），没有任何 kernel 层隐藏。代价是 63% 的 decode 时间在 Python for-loop 里（M3 benchmark），但你可以逐行理解每一个 tensor 操作的 shape、dtype 和语义。

这不是"谁更好"的问题，而是"给谁看"的问题：

```text
想看 kernel 怎么在 GPU 上高效读写 KV？  → 读 vLLM Triton source
想看 vLLM 的架构骨架长什么样？           → 读 nano-vllm（1200 行）
想看 PagedAttention 每个 tensor 操作？   → 读 inferlite（2800 行）
```

### 代码架构的稳定

M4 T7 引入的 ForwardContext + CacheAdapter 架构让后续里程碑的改动越来越小：

| 里程碑 | engine/ 改动 | attention.py 改动 | cache/ 改动 |
|--------|-------------|-----------------|------------|
| M2 | 拆 prefill/decode | 新增 cache 读写分支 | 新增 kv_cache.py |
| M3 | 新增 batch_core.py | 新增 batched 分支 | 新增 batched_kv_cache.py |
| M4 | **合并为统一 loop.py** | 新增 paged 分支 | 新增 3 个文件 |
| M5 | **loop.py 改 3 处** | **不改** | block_pool.py 升级 |

M5 对 attention.py 零改动——所有变化都在 cache 管理和调度层。这是架构解耦的成果：attention 计算不关心 KV 是怎么存的，cache 管理不关心 attention 是怎么算的。

### 测试覆盖的递进

| 里程碑 | 测试数 | 核心验证 |
|--------|-------|---------|
| M1 | 95 | 与 transformers logits 精确对齐 |
| M2 | 123 | 有 cache == 无 cache（torch.equal） |
| M3 | 211 | serial vs batch token 级一致 + continuous batching trace |
| M4 | 270 | scatter/gather 正确性 + NaN 安全 + M3 回归 |
| M5 | **314** | prefix hit/miss + CoW 不污染 + M4 回归 |

每个里程碑的测试覆盖三层：**新功能专项**、**旧路径回归**、**语义等价验证**。这意味着 314 个测试同时验证了 M1~M5 的所有路径。

---

## 7. 关键教训

从 M1 到 M5，有几个教训值得记录，因为它们不是"做错了"，而是"必须踩一次才能理解"。

**NaN 不是 mask 能解决的。** M3 后期发现变长 batch gather 会读到 `torch.empty` 中未初始化的垃圾值。如果恰好是 NaN，仅做 score mask 不够——`0 × NaN = NaN` 会通过 value matmul 传播。必须额外做 K/V 清零。这个 bug 用 NaN 注入回归测试锁定了。

**for-loop 写 cache 是纯 PyTorch 的固有代价。** M3 benchmark 显示 63% 的 decode 时间花在 Python for-loop 的 slice assign 上。nano-vllm 看起来和 vLLM 一样快，但它的 1200 行 Python 下面垫着 Triton kernel 和 FlashAttention——不是纯 PyTorch。inferlite 选择了纯 PyTorch 路线（先理解机制），性能代价留 M9 解决。

**ref_count 不等于 CoW。** M4 实现了 block 的引用计数，但不做 Copy-on-Write。两个请求共享一个 block 时，谁也不能写。M5 补上了 CoW：写入前检查 ref_count，大于 1 就 clone 独占副本。这个"先有 ref_count，再补 CoW"的顺序是刻意的——M4 聚焦分页，M5 聚焦复用，每步只解决一个问题。

**chain hash 必须包含位置信息。** 如果只做 `hash(token_ids)`，相同的 token 序列出现在不同位置会产生相同 hash，导致错误的前缀匹配。chain hash（`hash(prefix_hash ‖ token_ids)`）把"前面是什么"编码进去，实现了位置唯一性。

---

## 8. 接下来

M5 完成了 KV Cache 的主题线。inferlite 的下一步是 M6（API + SSE 服务化）——把 M1~M5 搭建的推理引擎包装成一个可以 curl 调用的 HTTP 服务。M5 的 prefix cache 能力会通过 API 自然暴露给调用方。

更远的路线上，M9 会用 Triton kernel 替换 M3/M4 的纯 PyTorch cache 读写，M10 会实现 Chunked Prefill 支持长上下文，M11 会扩展到 VLM 多模态。但这些都是建立在这五个里程碑打下的架构基础之上。

**完整代码和测试**：[github.com/luhao-lab/inferlite](https://github.com/luhao-lab/inferlite)
