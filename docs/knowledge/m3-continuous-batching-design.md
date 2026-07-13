# inferlite M3 设计文档：Continuous Batching

> **状态**：🟡 进行中
> **作者**：luhao
> **基于**：M2 tag `m2/static-kv-cache`（28 单测全通过，bench 7.36× at T=512）

---

## 摘要

M2 支持单请求的 KV Cache 加速，但一次只能处理一个请求。M3 引入 continuous batching，支持多请求共享 KV Cache，在 decode iteration 边界动态调度：finished 请求释放 slot，waiting 请求被 admit，实现资源的持续复用。

本文档按任务卡逐步记录设计决策、踩坑和技术细节。M3 结束后会基于本文重写为完整设计文档。

---

## 符号说明

| 符号 | 含义 | M3 典型值 |
|------|------|----------|
| max_num_seqs | 最大并发请求数（running 集合容量上限） | 2~8 |
| max_num_slots | KV Cache 的 slot 总数（= max_num_seqs） | 2~8 |
| T_p | prompt 长度 | 变化 |
| N | max_new_tokens，最大生成 token 数 | 16~128 |
| slot_id | 请求在 KV Cache 中的位置编号 | 0..max_num_slots-1 |

---

## 1. 从 M2 到 M3：为什么需要 continuous batching

### 1.1 M2 的调度方式

M2 没有 scheduler 模块。`engine/core.py` 里的 `generate()` 函数就是全部调度逻辑：

```python
# M2 的 generate：单请求，串行处理
def generate(engine, input_ids, max_new_tokens, kv_cache):
    kv_cache.reset()                          # 清空 cache
    logits = model(input_ids, kv_cache=cache) # prefill：处理整个 prompt
    next_token = sampler(logits[:, -1, :])    # 采样第一个 token

    for _ in range(max_new_tokens - 1):       # decode loop
        if next_token == eos: break
        logits = model(next_token, kv_cache=cache)
        next_token = sampler(logits[:, -1, :])

    return input_ids  # 返回完整序列
```

M2 处理多个请求只能完全串行：

```python
# M2：三个请求依次执行，没有 overlap
output_a = generate(engine, prompt_a, kv_cache)   # cache reset → 跑完 → 返回
output_b = generate(engine, prompt_b, kv_cache)   # cache reset → 跑完 → 返回
output_c = generate(engine, prompt_c, kv_cache)   # cache reset → 跑完 → 返回
```

请求 A 只跑了 2 步就 EOS 结束，但 B 和 C 必须等 A 完全退出才能开始。GPU 在 A 结束到 B 开始之间是空闲的。

### 1.2 M3 要解决的问题

```text
请求 A 跑了 2 步就 EOS → 它的 KV slot 应该立刻释放
请求 C 在排队 → 应该立刻进入 A 腾出的 slot 开始跑
请求 B 还在跑 → B 和 C 应该共享同一个 batch 做 decode
```

这就是 continuous batching：在 decode iteration 边界，finished 请求退出释放资源，waiting 请求进入复用资源，running 集合动态变化。

### 1.3 模块职责拆分

M2 的 `generate()` 把三件事混在一起。M3 拆开：

```text
scheduler/  →  "谁该跑"       →  纯状态机，不碰模型
engine/     →  "怎么跑"       →  模型 forward + 采样
model/      →  "计算细节"     →  attention + KV cache + 网络层
```

`scheduler/` 是 M3 新增的模块，负责管理请求集合（waiting/running/finished）、状态迁移、容量控制和调度策略。

---

## 2. M3 目标与边界

### 2.1 目标

实现教学版 continuous batching：每个 decode iteration 重新组 batch，running 请求共享一次模型 forward；KV Cache 用固定 slot 方案，每个请求维护自己的 `slot_id + seq_len`。

| 能力 | M3 支持 | 说明 |
|---|---|---|
| 多请求 waiting 队列 | ✅ | 支持同时提交多个请求 |
| waiting / running / finished 三队列 | ✅ | 调度核心 |
| prefill 逐条做 | ✅ | 每个请求单独 prefill，避免 padding 浪费 |
| decode batch | ✅ | 所有 running 请求共享一次 forward |
| 每请求独立 cache length | ✅ | `seq_lens[slot]` |
| 固定 slot-based KV Cache | ✅ | 每个请求占一个 slot |
| 不同请求 EOS 早停 | ✅ | finished 请求立即出队释放 slot |
| 新请求中途加入 | ✅ | decode iteration 边界 admit waiting 请求 |
| server-style 调度循环 | ✅ | 最小形态：不等整个 wave 结束 |

### 2.2 边界（不做）

| 能力 | 说明 | 留给 |
|---|---|---|
| batch prefill | prompt 长度不同，padding 浪费严重 | 后续 |
| chunked prefill | 长 prompt 切块，需要 token budget | M11 |
| decode-first / token-budget scheduling | vLLM V1 等生产策略 | M11+ |
| prefill/decode mixed batch | 需要 chunked prefill + varlen 表示 | M11+ |
| PagedAttention | KV Cache 按 page/block 分配 | M4 |
| Prefix / session cache | 跨请求 KV 复用 | M5 |
| 请求抢占 / preemption | running 请求被踢回 waiting | 后续 |
| batching window | 等待 5ms 凑满 batch | — |
| HTTP Server / 流式输出 | 实时接收请求、SSE streaming | M5 |

**一句话边界**：M3 只保证 fixed-slot continuous batching 的语义闭环 —— 多请求共享 decode batch、slot 可跨请求复用、iteration 边界动态调度。

### 2.3 关键设计决策

#### 为什么 prefill 不做 batch

每个请求的 prompt 长度不同，batch prefill 需要 padding 到最大长度：

```text
reqA prompt_len = 100, reqB = 300, reqC = 80

naive batch prefill → padding 到 300：
  真实 token: 480
  实际计算: 3 × 300 = 900
  浪费: ≈ 47%
```

而且 prefill 的 attention 是 O(T²)，padding 浪费是平方级的。浪费的不只是 KV cache，整条 forward（embedding / QKV projection / RMSNorm / MLP / lm_head）都在算无用的 pad token。

decode 每个请求每步都是 1 token（`input_ids = [B, 1]`），天然对齐。唯一的浪费是 KV 历史长度不同导致 gather 多读了无效位置（O(T)），远小于 prefill 的 O(T²)。

**结论：prefill 逐条做，decode 组 batch。**

#### 为什么用固定 slot 而不是 PagedAttention

M3 要学习的是 **scheduler 如何每步重新组 batch**，不是**显存 allocator 怎么设计**。固定 slot 最容易理解：`slot_id` 就是地址，`seq_len` 就是长度，不需要 block table、page 分配、block 回收。

代价是每个 slot 预分配 max_seq_len 大小的空间，短请求也占满一个 slot。但 M3 的 max_num_seqs 很小（2~8），总显存可控。消除浪费留给 M4。

#### 为什么不做 decode-first / token-budget scheduling

vLLM V1 在 chunked prefill 启用时优先调度 decode，再用剩余 token budget 调度 prefill，倾向优化 ITL 并混合 compute-bound 的 prefill 与 memory-bound 的 decode。

M3 不做这个策略：需要 chunked prefill（M11）、token budget 管理、prefill/decode mixed batch（varlen 表示）作为基础设施。M3 的策略是简单的 FCFS + prefill-first：有空 slot 就逐条 prefill，然后 batch decode。

#### 为什么不等凑满 batch

M3 采用 opportunistic batching：每个 iteration 边界有多少 running 请求就组多大的 batch，不为凑满而等待。M3 没有 HTTP server，所有请求一次性提交，"等凑 batch" 没有意义。

#### 请求从哪来

M3 的"动态"不是指请求实时到达，而是指请求完成速度不同导致的逐入逐出。所有请求一次性提交到 waiting 队列，finished 释放 slot 后 waiting 被 admit。HTTP server 实时接收请求留 M5。

---

## 3. T1: RequestState + FCFSScheduler

> 完成时间：2026-07-05

T1 是 M3 的第一步：建立纯 Python 的请求状态机。不碰模型、不碰 KV Cache。

### 3.1 核心数据结构

#### RequestStatus

四个状态：

```text
WAITING ──→ RUNNING ──→ FINISHED
    │           │
    └───────────┴──→ CANCELLED
```

- **WAITING**：已提交，排队中，不占 KV 资源
- **RUNNING**：已被 admit，正在使用 KV slot
- **FINISHED**：正常完成（EOS / max_tokens），终态
- **CANCELLED**：外部取消，终态但不算成功

FINISHED vs CANCELLED：都是终态，但 FINISHED 表示输出可用，CANCELLED 表示输出不完整。M5 Server 阶段需要区分返回给调用方的结果码。

#### RequestState

`@dataclass`，承载一个请求的全部运行时状态：

```python
@dataclass
class RequestState:
    request_id: str
    prompt_ids: torch.Tensor          # [1, prompt_len]
    max_new_tokens: int
    eos_token_id: int | None = None   # None = 不用 EOS 停止

    last_token: torch.Tensor | None = None
    num_generated: int = 0
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: list[torch.Tensor] = field(default_factory=list)

    slot_id: int | None = None
    seq_len: int = 0                   # prompt_len + num_generated
```

字段按职责分组：身份 / 输入 / 结束条件 / 输出进度 / KV Cache。

#### FCFSScheduler

```python
class FCFSScheduler:
    waiting: deque[RequestState]       # popleft O(1)，天然 FCFS
    running: dict[str, RequestState]   # 按 request_id O(1) 索引
    finished: dict[str, RequestState]  # 终态
    cancelled: dict[str, RequestState] # 终态
    _known_request_ids: set[str]       # 防重复，只加不删
```

FCFS（First Come First Served）是调度领域术语，等价于数据结构领域的 FIFO。调度器用 FCFS，队列本身是 FIFO deque。

### 3.2 设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| waiting 数据结构 | deque | popleft O(1)，天然 FCFS |
| running/finished/cancelled | dict | 按 request_id O(1) 查找删除 |
| status 放哪 | RequestState 上，只由 Scheduler 修改 | 外部只读检查方便；nano-vllm 同样做法 |
| _known_request_ids 清理 | M3 不删 | 批量提交场景，无清理需求 |
| eos_token_id 默认 | None | 不用 EOS 停止，只靠 max_new_tokens |
| generated_tokens 初始化 | field(default_factory=list) | 避免多请求共享 list 引用 |

### 3.3 与主流框架对比：status 管理

| | vLLM | SGLang / Orca | nano-vllm / inferlite |
|---|---|---|---|
| status enum | ✅ SequenceStatus | ❌ 没有 | ✅ RequestStatus |
| 谁改 status | 只 Scheduler（_status + @property 只读） | 不存在 | 只 Scheduler（公开字段） |
| 真相来源 | status 字段 | 队列位置 | 两个都有，约定只在 Scheduler 改 |

**选择**：和 nano-vllm 一致 —— status 在 Request 对象上作为公开字段，约定只在 Scheduler 修改。vLLM 用 `@property` 封装更严格但对单人项目过度设计；SGLang 完全没有 status 字段，外部每次要问 Scheduler 不方便。

### 3.4 踩坑记录

1. **`field(default_factory=list)` vs `= []`**：dataclass 字段默认值在 class 定义时创建，不是每次构造对象时。可变对象（list/dict）会变成所有实例共享。Python 变量保存的是对象引用，不是值本身 —— 这和函数传参是零拷贝是同一个原因。

2. **`mark_finished` 的顺序**：必须先检查请求是否在 running 中，再修改 status 和移动队列。否则 status 被污染，finished dict 多脏数据。

3. **`cancel` 的 `elif`**：先判断请求在 running 还是 waiting，用 `elif` 保证只从一个队列移除。先检查再修改状态，避免不存在时仍放入 cancelled dict。

4. **Python 传参是零拷贝**：Scheduler 方法里修改 `request.status` 直接影响外部持有的同一个 RequestState 对象，因为 Python 传参传的是对象引用，不做任何拷贝。

### 3.5 测试覆盖

15 个单测（`tests/unit/test_scheduler.py`）：
- submit / admit / mark_finished / cancel 基本功能
- FCFS 顺序、容量限制
- 重复 id 抛 ValueError
- 四队列守恒不变量（多步骤全生命周期验证）
- generated_tokens 独立性（default_factory 验证）
- continuous batching 逐入逐出 trace
- has_unfinished 判断（含 waiting-only 场景）

### 3.6 在推理链路中的位置

```text
用户提交请求
  ↓
RequestState（request.py）
  ↓
FCFSScheduler: waiting / running / finished（fcfs.py）
  ↓
T2 分配 KV slot
  ↓
T4 BatchEngine 执行 prefill / decode
```

---

## T2: BatchedKVCache + SlotManager

> 状态：✅ 完成

### M2 vs M3 的 KV Cache 本质区别

M2 的 `LayerKVCache` 第一维是 `batch_size`，理论上 B>1 也能放多个请求。但它只有一个全局 `cur_len`，所以所有请求必须**锁步同步**（static batching）：

```text
M2 static batching：
  - 所有请求同时 prefill
  - 所有请求同时 decode（共享 cur_len）
  - 所有请求同时结束（短的 padding 等长的）
  - 不可能一个请求 seq_len=128，另一个 seq_len=64
```

M3 的 `BatchedLayerKVCache` 第一维从 batch 变成 slot，配合 per-slot 的 `seq_lens[S]`，每个请求独立进退（continuous batching）：

```text
M3 continuous batching：
  - 请求可以不同时 prefill（逐条进入）
  - 请求可以不同时结束（有的先到 EOS 先退出）
  - 退出后 slot 被新请求复用
  - 每个 slot 有独立的 seq_lens[s]
```

| | M2 KVCache | M3 BatchedKVCache |
|---|---|---|
| 第一维含义 | batch（同步组） | slot（独立请求） |
| 长度管理 | 全局 `cur_len`（int） | per-slot `seq_lens[S]`（tensor） |
| 请求独立性 | 锁步同步 | 独立进退 |
| 占用管理 | 无（整体 reset） | per-slot `occupied[S]` + SlotManager |

注意：`BatchedLayerKVCache` 和 `LayerKVCache` 的**数据结构完全相同**（都是 `k: Tensor, v: Tensor`），本质区别在 `BatchedKVCache` 这一层（per-slot 元数据管理）。单独定义新类是为了语义清晰和独立演进。

### max_num_slots 与 max_num_seqs 的关系

这两个是**同一个值**，在不同层表达不同语义：

```text
max_num_seqs = max_num_slots = 同一个数（比如 4）

Scheduler 层：max_num_seqs = 最多同时跑几个请求
KV Cache 层：max_num_slots = 预分配了几个 cache slot
```

它们必须相等：每个 running 请求需要恰好一个 KV slot，Scheduler 保证 `len(running) ≤ max_num_seqs`，所以 allocate 次数不会超过 `max_num_slots`，SlotManager 永远不会触发 RuntimeError。

```text
Scheduler.max_num_seqs  ──约束──→  最多 admit 几个请求
                                        ↓
                                  每个请求 allocate 一个 slot
                                        ↓
KVCache.max_num_slots   ──保证──→  slot 够用
```

实际使用时从同一个 config 参数读取：

```python
max_concurrent = config.max_num_seqs
scheduler = FCFSScheduler(max_num_seqs=max_concurrent)
kv_cache = BatchedKVCache.from_config(config, max_num_slots=max_concurrent, ...)
```

### 为什么 max_num_slots / max_seq_len 不在 ModelConfig 里

它们是**运行时参数**，不是模型参数：

```text
ModelConfig 描述模型架构（固定不变）：
  num_hidden_layers=28, hidden_size=1024, num_kv_heads=8, head_dim=64

max_num_slots / max_seq_len 描述推理配置（每次运行可变）：
  同时跑几个请求？→ max_num_slots = 4 还是 8
  最多生成多长序列？→ max_seq_len = 512 还是 1024
```

同一个 Qwen3-0.6B，不同场景可以用不同配置。和 M2 的 `KVCache.from_config()` 把 `batch_size` 和 `max_seq_len` 作为参数传入是同一设计。

### __init__ 与 from_config 的分工（工厂模式）

```python
# from_config: 高层接口，用户用这个
#   从 config 读模型参数 → 分配 tensor → 组装 layers → 调 __init__
cache = BatchedKVCache.from_config(config, max_num_slots=4, max_seq_len=512, ...)

# __init__: 低层接口，接收已构建的对象
#   只做赋值，不分配 tensor（测试时手动构建小 tensor 用）
cache = BatchedKVCache(layers=[...], max_seq_len=512, max_num_slots=4)
```

`__init__` 中的 `max_num_slots` 和 `max_seq_len` 也可以从 layers 推断（`layers[0].k.shape[0]` / `shape[2]`），显式传入更防御性，隐式推断更简洁。

### @classmethod 中的 `cls(...)` 用法

`from_config` 用 `@classmethod` 装饰，第一个参数 `cls` 是类本身（不是实例 `self`）。`cls(...)` 等价于调用该类的 `__init__` 构造实例：

```python
@classmethod
def from_config(cls, config, ...):
    # cls = BatchedKVCache（或子类）
    # cls(layers, ...) = BatchedKVCache(layers, ...)
    return cls(layers, max_seq_len, max_num_slots)
```

用 `cls` 而非硬编码类名是为了**支持继承**：子类调用 `from_config` 时返回子类实例而非父类实例。这是 Python 工厂方法的标准写法，M2 的 `KVCache.from_config()` 也用 `return cls(layers)`。

### 最终实现

**文件结构：**

```text
inferlite/model/
├── __init__.py              # 导出 BatchedKVCache, BatchedLayerKVCache, SlotManager
├── kv_cache.py              # M2（不动）
└── batched_kv_cache.py      # M3 新增
```

**三个类的职责：**

| 类 | 职责 | 核心字段 |
|---|---|---|
| `BatchedLayerKVCache` | 单层 KV 数据容器 | `k: Tensor [S, H_kv, L, D]`, `v: Tensor` |
| `SlotManager` | slot 分配/释放 | `free_slots: deque`, `req_to_slot: dict` |
| `BatchedKVCache` | 多层 cache + per-slot 元数据 | `layers`, `seq_lens`, `occupied`, `slot_manager` |

**SlotManager 接口：**

```python
sm.allocate(request_id) -> slot_id    # 分配，失败抛 ValueError/RuntimeError
sm.free(request_id) -> None           # 释放，失败抛 ValueError
sm.is_free(slot_id) -> bool           # 查询
```

**BatchedKVCache 接口：**

```python
cache = BatchedKVCache.from_config(config, max_num_slots, max_seq_len, dtype, device)
slot_id = cache.allocate_slot(request_id)   # 分配 + 设 occupied=True
cache.free_slot(request_id)                 # 释放 + 清 seq_lens/occupied
cache.reset_slots()                          # 全部清零（benchmark 用）
```

**seq_lens 的更新时机：**

`seq_lens` 是被动数据，BatchedKVCache 只负责分配/清零，中间的递增由 BatchEngine（T4）在推理循环中直接写：

```python
# prefill 完成后
cache.seq_lens[slot_id] = prompt_len

# 每步 decode 后
cache.seq_lens[slot_id] += 1
```

主流框架（vLLM, nano-vllm）也不包函数，直接在 Sequence 对象上追踪。

### 设计决策总结

| 决策 | 选择 | 理由 |
|---|---|---|
| SlotManager 数据结构 | `deque` + `dict` | deque O(1) popleft/append；dict O(1) 查 req→slot |
| 只保留 req_to_slot | 不要 slot_to_req | free() 传 request_id，不需要反查 |
| free_slot 不清 tensor | 只清元数据 | 和 M2 reset() 一致，下次 prefill 会覆盖 |
| 不继承 M2 KVCache | 独立类 | M2 是全局 cur_len，M3 是 per-slot seq_lens，语义不同 |
| torch.empty vs zeros | `empty` | k/v 总是 prefill 时覆盖写入，不需要初始化 |
| free_slot/allocate_slot 在 BatchedKVCache 上 | 对称封装 | 同时更新 SlotManager + occupied + seq_lens，调用方只调一个方法 |

### 测试覆盖

18 个单测覆盖 L0 全部 9 项：

| L0 项 | 测试 |
|---|---|
| cache shape [S, H_kv, L, D] | `test_from_config_shape` |
| dtype/device 一致 | `test_dtype_device` |
| allocate 从低 slot id 开始 | `test_allocate_order` |
| 超过容量抛 RuntimeError | `test_allocate_over_capacity` |
| free 后可复用 | `test_free_and_reuse` |
| duplicate request_id 抛 ValueError | `test_duplicate_request_id` |
| free 不存在抛 ValueError | `test_free_not_found`, `test_free_slot_not_found` |
| seq_lens 初始化/清零 | `test_seq_lens_init`, `test_free_slot_clears_metadata`, `test_reset_slots` |
| occupied mask 一致 | `test_occupied_init`, `test_allocate_slot`, `test_reset_slots` |

### T3 依赖

T3（BatchedAttention）需要：
- 从 `cache.seq_lens` 读取每个 slot 的有效长度，构造 attention mask
- 直接读 `cache.layers[i].k/v` 的 `[:, :, :seq_len, :]` 切片做 attention
- 写入新 KV 时用 `cache.layers[i].k[slot, :, pos, :] = new_k`（pos = seq_lens[slot]）

---

## T3: BatchedAttention

> 状态：✅ done

### 核心问题

M2 attention 的 cache 读写用全局 `cur_len`（所有请求同步）：

```python
# M2: 所有 batch row 写同一个 cache_position，读同一段历史
cache.k[:, :, cache_position : cache_position + seq_len, :] = k
k = cache.k[:, :, : cache_position + seq_len, :]
```

M3 需要 per-slot 独立位置：

```python
# M3: 每个 batch row 写自己 slot 的 position，从自己 slot gather 历史
for i in range(B):
    cache.k[slot_i, :, pos_i : pos_i + 1, :] = k[i]
k = cache.k[cache_slots, :, :max_len, :]  # gather
# + per-row mask（每个请求可见长度不同）
```

### 方案选择：扩展现有 GQAAttention vs 新建类

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| A: 扩展现有 forward | `isinstance` 分派 | 共享 q_proj/k_proj/o_proj 等权重 | forward 变长 |
| B: 新建 BatchedGQAAttention | 独立类 | 职责清晰 | 重复所有 projection 权重 |

**选择方案 A**：q_proj/k_proj/v_proj/o_proj/q_norm/k_norm/rotary_emb 全部相同，只是 cache 读写分支不同。用私有方法抽取 cache 逻辑保持可读性。

### 实现结构

forward 主流程不变（6 步），cache 读写按类型分派到私有方法：

```python
def forward(self, ..., layer_kv_cache=None,
            cache_position=0,              # M2
            cache_slots=None,              # M3: [B]
            cache_positions=None):         # M3: [B]
    # 1-3. projection + norm + RoPE（M1/M2/M3 通用）
    # 4. Cache 读写：
    if isinstance(layer_kv_cache, BatchedLayerKVCache):
        k, v = self._batched_cache_rw(...)
    elif layer_kv_cache is not None:
        k, v = self._single_cache_rw(...)
    # 5. repeat_kv + attention + mask
    if isinstance(layer_kv_cache, BatchedLayerKVCache):
        mask = self._build_batched_mask(...)
    # 6. o_proj
```

三个私有方法：

| 方法 | 职责 |
|---|---|
| `_single_cache_rw` | M2: 全局 cache_position 写入 + 切片读取 |
| `_batched_cache_rw` | M3: per-slot 写入 + gather |
| `_build_batched_mask` | M3: per-row visible mask |

### M3 prefill 策略

任务卡说"不做 prefill batching"。M3 prefill 仍然一条一条处理，**复用 `_batched_cache_rw`**（B=1 的特殊情况）。不能用 `_single_cache_rw`，因为 M3 cache 第一维是 slot 不是 batch。

### 与主流框架对比

| | inferlite M3 | nano-vllm | vLLM / SGLang |
|---|---|---|---|
| KV 读取 | Python gather | PyTorch index_select | CUDA kernel 内部 gather |
| mask | Python tensor | PyTorch mask | kernel 内按 context_len 裁切 |
| 性能 | O(B × max_len) materialize | 同上 | O(B × avg_len)，不 materialize |

我们的 gather + per-row mask 是教学版标准做法（nano-vllm 同），生产框架把 gather + mask 下沉到 CUDA kernel 优化。M4 PagedAttention 会进一步优化。

### M2 vs M3 的 "batch" 含义

M2 attention 张量本身也支持 `B>1`（`q: [B, n_heads, T, D]`），但 batch 维是**同步组**：

```text
M2: cache.k[:, :, cache_position:..., :] = k    ← 所有 B 行写同一个位置
    cache.k[:, :, :cur_len, :]                   ← 所有 B 行读同一段历史
    → B 个请求必须锁步：同一时刻进入、同一 prompt 长度、同一生成步数

M3: cache.k[slot_i, :, pos_i:..., :] = k[i]      ← 每行写自己的 slot + position
    cache.k[cache_slots, :, :max_len, :]          ← 每行从自己 slot gather
    → B 个请求独立：不同时刻进入、不同长度、不同生成步数
```

**T3 的核心不是"让 attention 支持 batch"（M2 已经支持），而是"让 batch 中每行独立访问自己的 KV 历史"。**

### gather 临时张量的内存浪费

M3 gather 需要创建固定大小的临时张量：

```python
max_len = int(cache_positions.max()) + 1   # 取所有请求中的最大位置
k = cache.k[cache_slots, :, :max_len, :]   # [B, H_kv, max_len, D]
```

假设 3 个请求位置分别是 128、64、300：

```text
gather 后：[3, H_kv, 301, D]

request 0 (pos=128): 有效 129 个 KV，后 172 个位置是垃圾数据（被 mask 掉）
request 1 (pos=64):  有效  65 个 KV，后 236 个位置是垃圾数据
request 2 (pos=300): 有效 301 个 KV，全部有效

浪费率 = (3×301 - 495) / (3×301) ≈ 45%
```

**位置差异越大，浪费越大。** 生产框架用自定义 kernel 避免这个问题（kernel 内部按 block_table 逐个 gather，不 materialize 整个 dense tensor）。教学版语义等价但多浪费临时内存。

### 改动范围

| 文件 | 改什么 |
|---|---|
| `attention.py` | 加 3 个私有方法 + 修改 forward 参数和 cache/mask 分支 |
| `qwen3.py` DecoderLayer | forward 加 `cache_slots`/`cache_positions`，透传 |
| `qwen3.py` Qwen3Model | forward 加参数，透传 |
| `test_batched_attention.py` | 10 个测试（8 个 L0 + 2 个 Model 级） |

### 最终实现

**attention.py** — 3 个私有方法 + forward 改造：

| 方法 | 职责 | 类型 |
|---|---|---|
| `_single_cache_rw` | M2：全局 cache_position 写入 + 切片读取（view） | `LayerKVCache → (k, v)` |
| `_batched_cache_rw` | M3：per-slot 写入 + gather 到 `[B, H_kv, max_len, D]` | `BatchedLayerKVCache → (k, v)` |
| `_build_batched_mask` | M3：per-row visible mask，padding 位置填 dtype min | `(scores, positions) → scores` |

forward 主流程 6 步不变，cache 读写和 mask 按 `isinstance` 分派。

**qwen3.py** — 参数透传：

- `DecoderLayer.forward()` 加 `cache_slots`/`cache_positions`，透传给 `self_attn`
- `Qwen3Model.forward()` 加 `cache_slots`/`cache_positions`，透传给每层
- `cache_position` 用 `isinstance(kv_cache, KVCache)` 判断，M3 时传 0（不使用）

### 设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 扩展 vs 新建类 | 扩展现有 GQAAttention | q/k/v/o_proj 权重共享，不需要重复 |
| cache 逻辑组织 | 私有方法抽取 | forward 主流程保持可读 |
| M3 prefill 路径 | 复用 `_batched_cache_rw`（B=1） | M3 cache 第一维是 slot，不能走 M2 的 `_single_cache_rw` |
| per-row mask 位置 | forward 内独立 `if` 分支 | 不依赖 `seq_len > 1`（M3 decode seq_len=1 也需执行） |
| cache 写入方式 | Python for 循环 | B=4~8 开销远小于矩阵乘，nano-vllm 同做法 |
| M3 `cache_position` 处理 | `isinstance(kv_cache, KVCache)` 判断 | `BatchedKVCache` 无 `cur_len` 属性 |

### 测试覆盖

| L0 项 | 测试名 | 验证内容 |
|---|---|---|
| 1 | `test_batched_decode_output_shape` | `[B, 1, 32]` shape 正确 |
| 2 | `test_cache_slot_write_position` | 只写 slot_i 的 pos_i，其他位置/slot 为零 |
| 3 | `test_no_cross_slot_attention` | 单条 vs 合批输出一致（atol=1e-5） |
| 4 | `test_mask_preserves_current_position` | 输出非全零，attend 到了自己 |
| 5 | `test_padding_positions_masked` | 输出无 NaN/Inf |
| 6 | `test_b1_equivalent_to_m2_decode` | B=1 batched ≈ M2 single（atol=1e-4） |
| 7 | `test_mixed_positions_equivalent_to_sequential` | 混合 batch ≈ 逐条 decode（atol=1e-4） |
| 8 | `test_gqa_repeat_kv_shape` | GQA heads 对齐无报错 |
| - | `test_model_batched_decode_shape` | Model 级 `[B, 1, 32]` shape |
| - | `test_model_m2_not_broken` | M2 KVCache 路径不受影响 |

### T4 依赖

T4 BatchEngine 需要：
1. 维护 `cache_slots`/`cache_positions` 并在每轮 decode 时传入 `model()`
2. 调用 `BatchedKVCache.allocate_slot`/`free_slot` 管理槽位
3. 递增 `BatchedKVCache.seq_lens` 更新每个请求的有效长度

---

## T4: BatchEngine

> 状态：进行中

### 核心问题

T1/T2/T3 分别解决了 scheduler、slot cache、batched attention。T4 要把它们串成 continuous batching 执行流：

```text
submit requests → prefill one by one → admit to running slots
→ batched decode one token → finished leave, waiting enter → loop
```

### seq_len 语义：与 M2 cur_len 和 nano-vllm 对齐

| 阶段 | seq_len | cache_positions | 含义 |
|---|---|---|---|
| prefill 前 | 0 | — | 无历史 |
| prefill 后 | prompt_len | — | KV[0..P-1] 已写入，下一步写 P |
| decode step 1 | prompt_len | prompt_len | 写 KV[P]，采样后 seq_len=P+1 |
| decode step N | prompt_len + N - 1 | prompt_len + N - 1 | 写 KV[P+N-1]，采样后 seq_len=P+N |

nano-vllm 的 `prepare_decode` 用 `positions.append(len(seq) - 1)` 也是同样的"下一个写入位置"语义。

### 主循环结构

```python
while scheduler.has_unfinished():
    # 1. finish done + free slots
    for req in get_finished_in_decode():
        scheduler.mark_finished(req)
        cache.free_slot(req.request_id)

    # 2. admit waiting + prefill one by one
    admitted = scheduler.admit_until_full()
    for req in admitted:
        slot = cache.allocate_slot(req.request_id)
        req.slot_id = slot
        prefill_one(req, slot)

    # 3. batched decode one step
    if not scheduler.running:
        break
    running = list(scheduler.running.values())
    cache_slots = [r.slot_id for r in running]
    cache_positions = [r.seq_len for r in running]
    logits = model(next_tokens, kv_cache=cache,
                   cache_slots=cache_slots, cache_positions=cache_positions)
    sampled = sampler(logits)

    # 4. update state
    for req, tok in zip(running, sampled):
        req.seq_len += 1
        req.generated_tokens.append(tok)
        cache.seq_lens[req.slot_id] = req.seq_len
        if is_finished(req, tok):
            mark_for_finish(req)
```

### 改动范围

| 文件 | 改什么 |
|---|---|
| `qwen3.py` Qwen3ForCausalLM | forward 加 `cache_slots`/`cache_positions`，透传到 model |
| `engine/batch_core.py` | 新建 BatchEngine + batch_generate |
| `engine/__init__.py` | 导出 BatchEngine, batch_generate |
| `test_batch_engine.py` | 10 个 L0 测试 |

---

## T5: E2E Correctness

> 待完成

---

## T6: Metrics & Benchmark

> 待完成

---

## T7: Docs & Tag

> 待完成
