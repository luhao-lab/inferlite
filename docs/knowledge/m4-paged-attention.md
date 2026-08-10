# inferlite M4：PagedAttention 完整设计

| 字段 | 内容 |
|---|---|
| 状态 | ✅ 完成（tag: `m4/paged-attention`，2026-08-11） |
| 前置 | M3 tag `m3/continuous-batching` |
| 后续 | M5 Prefix Caching |
| 测试 | 270 tests 全绿 |

---

## 摘要

M3 用 fixed-slot KV Cache 跑通了 continuous batching，但每个请求独占 `max_seq_len` 连续物理空间，短请求浪费严重。M4 引入 PagedAttention：把 KV Cache 从连续数组改为虚拟内存式分页管理——请求的逻辑 KV 切成固定大小 block，通过 block table 映射到非连续物理 block。

**M4 的核心收获：block pool + block table + paged scatter/gather + NaN 安全 + ForwardContext/CacheAdapter 统一架构。**

---

## 符号说明

| 符号 | 含义 | M4 典型值 |
|---|---|---|
| `block_size` | 每个物理 block 容纳的 token 数 | 16 / 32 |
| `num_blocks` | 物理 block 总数 | 可配置 |
| logical block | 请求内部的逻辑 block 编号 | `pos // block_size` |
| physical block | KV 池中的实际 block id | `0..num_blocks-1` |
| `block_table` | logical block → physical block 的映射 | `list[int]` |
| `ref_count` | 物理 block 的引用计数 | ≥0 |

---

## 1. M3 → M4 的关键变化

| 维度 | M3 fixed-slot | M4 paged |
|---|---|---|
| 内存单位 | slot | block |
| 请求→物理内存 | `request_id → slot_id` | `request_id → block_table → block_id` |
| KV layout | `[S, H, L, D]` | `[num_blocks, block_size, H, D]` |
| seq_len | `seq_lens[slot]` | `block_table.seq_len` |
| 写入 | `cache.k[slot, :, pos, :] = k` | scatter 到物理 block |
| 读取 | gather slot 的 `[0:seq_len]` | 按 block_table gather 多个 block |
| 释放 | 释放整个 slot | 每个 block refcount--，为 0 才释放 |
| 碎片 | 短请求浪费 `max_seq_len - actual_len` | 最多浪费 `block_size - 1` 个 token |

---

## 2. 与 vLLM / nano-vLLM 的异同

inferlite M4 是 **纯 PyTorch 教学版 PagedAttention**，只取核心内存管理思想：

| 维度 | inferlite M4 | nano-vLLM | vLLM |
|---|---|---|---|
| Attention 实现 | PyTorch gather 伪版 | Triton kernel + FlashAttention | 生产 kernel |
| Prefix Cache | ❌（留 M5） | ✅ hash + LRU | ✅ |
| CoW | ❌（留 M5） | ✅ | ✅ |
| Scheduler | 简化 FCFS | token budget | 完整 preemption |
| 目标 | 理解机制 | 接近可跑 | 生产 serving |

**明确不做**：Triton kernel（M9）、Prefix Cache（M5）、Chunked Prefill（M10）、OpenAI API（M6）。

---

## 3. ADR 决策

| ADR | 决策 | 理由 |
|---|---|---|
| 新建 PagedKVCache | 不改 BatchedKVCache | M3 保留做 oracle，回滚简单 |
| PyTorch gather 伪版 | 不写 Triton kernel | Mac/MPS 友好，先理解机制 |
| `block_size=16` | 不用生产的 256 | 短 prompt 跨 block，单测更有效 |
| 保留 refcount | 不做 CoW/hash | M4 只聚焦分页，prefix 留 M5 |
| 不做 chunked prefill | 保留 full prefill | 分页机制本身先验证 |

---

## 4. 三层职责

PagedKVCache 由三个组件协作：

| 组件 | 数量 | 管什么 | 不管什么 |
|---|---:|---|---|
| `BlockPool` | 全局 1 个 | 物理 block 分配/释放/ref_count | tensor、请求顺序 |
| `BlockTable` | 每请求 1 个 | logical block → physical block + `seq_len` | 空闲块、tensor |
| `PagedKVCache` | 全局 1 个 | 每层 K/V tensor、scatter/gather | 调度、采样 |

依赖方向：`BlockPool` 和 `BlockTable` 互不认识，通过 `PagedKVCache` 中转 `block_id`。

物理寻址：

```
layer.k.shape = [num_blocks, block_size, n_kv_heads, head_dim]
                          ^          ^
                      block_id     offset
```

也可以展平：`slot = block_id * block_size + offset`。

---

## 5. 请求生命周期

```
prefill:
  allocate_request(id, prompt_len)  → 分配 ceil(prompt_len/block_size) 个 block
  write_prefill(layer, ids, k, v)  → scatter K/V 到物理 block

decode:
  append_token(id)                 → 如当前 block 已满，分配新 block；seq_len += 1
  write_decode(layer, ids, k, v)  → 写 1 个 token 到 pos = seq_len - 1

finish:
  free_request(id)                 → 归还所有 block，删除 block table
```

关键约束：decode 必须先 `append_token()` 再 `write_decode()`，否则新 K/V 会覆盖历史。

---

## 6. Scatter/Gather 数据流

### 6.1 Prefill scatter 写入

以 `block_size=4`，两个请求为例：

```
request a: block_ids=[3,1], seq_len=5
request b: block_ids=[2],   seq_len=2
```

**Step 1：生成 slot_mapping**

对每个请求，用 block_ids 广播生成所有容量位置的 slot：

```
a: block_ids[:, None] * 4 + offsets[None, :]
   = [[12,13,14,15], [4,5,6,7]]
   → flatten()[:5] = [12, 13, 14, 15, 4]

b: → flatten()[:2] = [8, 9]

slot_mapping = [12, 13, 14, 15, 4, 8, 9]   # request 优先、pos 递增
```

**Step 2：从 padded batch 提取 flat K/V**

输入 `[B, n_kv, T_max, D]`（B=2, T_max=5），用 `seq_lens=[5,2]` 做 valid mask，boolean index 得到：

```
flat_k = [A0, A1, A2, A3, A4, B0, B1]   # 与 slot_mapping 一一对应
```

**Step 3：scatter**

```python
flat_cache_k[slot_mapping] = flat_k
```

连续源数据按不连续目标下标分散写入。前提：`slot_mapping` 无重复 slot（BlockPool 独占分配保证）。

### 6.2 Decode 写入

每轮每个请求只写 1 个 token，`slot_mapping` 长度为 B：

```python
slot = last_block_id * block_size + ((seq_len - 1) % block_size)
```

### 6.3 Gather 读取

**批量 gather**：block table padding 成矩阵后一次高级索引：

```python
block_table = [[7, 2],     # a: 2 blocks
               [5, 0]]     # b: 1 block, 0 是占位 padding

layer.k[block_table]       # → [B, max_blocks, block_size, n_kv, D]
  → reshape → [B, n_kv, L_pad, D]
```

`L_pad = max_blocks * block_size`，不是 `max(seq_len)`。短请求的 padding 位置含垃圾值。

### 6.4 NaN 安全

物理 tensor 用 `torch.empty` 创建，未写入位置可能含 NaN。仅做 score mask 不够：`0 × NaN = NaN`。

必须先用 `valid_lens` 清零无效 K/V：

```python
invalid = ~(positions[None, :] < valid_lens[:, None])[:, None, :, None]
k = k.masked_fill(invalid, 0)
v = v.masked_fill(invalid, 0)
```

score mask 管 attention 语义，K/V 清零管数值安全——两者缺一不可。

---

## 7. T7 vLLM V1 架构对齐

M4 实现过程中（T7）引入了 vLLM V1 的核心架构模式，消除了 M3/M4 之间 80% 的重复代码：

### 7.1 ForwardContext

cache 和 metadata 不经过模型参数传递：

```python
# 初始化：cache 绑定到每层 Attention
adapter.bind_kv_cache(model)

# 每次 forward 前：metadata 通过全局上下文设置
with set_forward_context(metadata):
    logits = model(input_ids, positions=positions)   # 只有 2 个参数
```

### 7.2 CacheAdapter Protocol

3 种 cache 实现统一接口：

```
CacheAdapter(Protocol):
  can_admit(prompt_len) → bool        # 容量检查
  allocate(req_id, prompt_len)        # 分配 cache
  free(req_id)                        # 释放 cache
  bind_kv_cache(model)                # 绑定到 Attention 层
  make_prefill_metadata(input, pos)   # 构造 AttentionMetadata
  make_decode_metadata(tokens, pos)   # 构造 AttentionMetadata
  prepare_decode(request_ids)         # decode 前同步状态
  set_seq_lens(requests)              # prefill 后同步 seq_lens
```

Engine 代码只通过这个接口操作 cache，不关心底层是 slot 还是 block。

### 7.3 统一 loop.py

M3/M4 共享 `batch_generate_loop()`：

```
while scheduler.has_unfinished():
  1. Admit:     can_admit → admit → allocate（逐条交替）
  2. Prefill:   _build_prefill_batch → make_prefill_metadata → forward → sample
  3. Decode:    prepare_decode → make_decode_metadata → forward → sample → update
  4. Finish:    完成 → free → 移出 running
```

### 7.4 Attention 两层拆分

- `Qwen3Attention`：projection → QK-norm → RoPE → 委托 Attention → o_proj
- `Attention`：cache RW（isinstance 分发 M2/M3/M4）→ GQA → causal mask → softmax

---

## 8. 最终架构

```
engine/（~800L，4 文件）
├── context.py    ForwardContext + AttentionMetadata + LLMModel Protocol
├── engine.py     M1/M2/M3/M4 generate 统一入口
├── loop.py       统一 batch 主循环（admit → prefill → decode → free）
└── metrics.py    性能指标采集

cache/（~1,250L，5 文件）
├── adapter.py    CacheAdapter Protocol + M2/M3/M4 三种 adapter
├── kv_cache.py   M2 单序列 LayerKVCache
├── batched_kv_cache.py  M3 固定 slot BatchedKVCache
├── block_pool.py        物理 block 分配池
└── paged_kv_cache.py    M4 PagedKVCache + BlockTable

model/attention.py
├── _single_cache_rw()   M2 路径
├── _batched_cache_rw()  M3 路径（prefill scatter + decode + gather）
└── _paged_cache_rw()    M4 路径（scatter + gather + NaN 安全）
```

---

## 9. 修复的隐藏 bug

| bug | 根因 | 修复 |
|-----|------|------|
| slot_mapping 始终指向最后分配的 slot | 批量 admit 后才逐个 prefill，`_current_request_ids[-1]` 总返回最后请求 | 改为逐条 admit-allocate-prefill 交替 |
| `_admit` 超额分配（3 请求占 2 slot） | `can_admit()` 只检查空闲不占用 | 同上，每条请求分配后才检查下一条 |
| sampler 3D 输入崩溃 | `logits[:, -1:, :]` 是 3D `[B,1,V]`，sampler 返回 3D | 改为 `logits[:, -1, :]`（2D `[B,V]`） |
| Metrics `total_output_tokens=0` | loop.py 未调 `record_step()` | 补充 `record_step`/`record_output_tokens`/`record_finished` |
| padded prefill cache 写入越界 | `k[i]` 是 padded 长度 T=4，但 `:plen` 切片只有 3 | 改为 `k[i, :, :plen, :]` 只写有效位置 |
| padded prefill attention 读到 padding | 不同 prompt 长度的请求 pad 到 max_len，padding 位置参与 attention | 新增 `valid_lens` mask 屏蔽 padding 位置 |

---

## 10. 与后续里程碑关系

| 里程碑 | M4 提供什么 | M4 不含什么 |
|---|---|---|
| M5 Prefix Cache | block table + refcount 基础 | hash、LRU、CoW |
| M9 Triton kernel | PyTorch gather 伪版做正确性 oracle | Triton/CUDA kernel |
| M10 Chunked Prefill | block table 可表达长 prompt 分块写入 | token budget、mixed scheduling |
| M6 API/SSE | 可降低长请求并发内存浪费 | HTTP server |

---

## 11. 测试覆盖

270 tests 全绿，关键测试：

- `test_block_pool.py` — 分配/释放/ref_count/double-free
- `test_paged_attention.py` — scatter/gather 正确性、NaN 安全、跨 block
- `test_paged_batch_engine.py` — paged batch 生命周期、EOS、block 耗尽
- `test_batch_generate.py` — serial vs batch `torch.equal` 等价
- `test_real_qwen3_batch_matches_serial` — 真实 Qwen3 模型端到端等价

---

## 12. 代码架构详解

> 本节从调用入口到模型 forward 逐层走读代码，帮助读者理解数据如何在层间流转。

### 12.1 入口：三条调用路径

用户通过 `cli.py` 或直接调用 `engine/engine.py` 中的三个函数启动推理：

```python
# M1/M2 单请求
engine = EngineCore(model, sampler)
result = generate(engine, input_ids, max_new_tokens=10, kv_cache=cache)

# M3 multi-request batched
results = batch_generate(model, sampler, prompts, max_num_slots=4, ...)

# M4 paged
results = batch_generate_paged(model, sampler, prompts, num_blocks=8, block_size=16, ...)
```

三条路径在 `engine/engine.py`（248L）中定义。`generate()` 处理 M1（无 cache）和 M2（单序列 cache）；`batch_generate()` 和 `batch_generate_paged()` 是 M3/M4 的薄包装，创建对应的 cache + adapter + scheduler 后委托给 `loop.batch_generate_loop()`。

**EngineCore.step()** 是最小推理单元：

```python
def step(self, input_ids):
    logits = self.model(input_ids, logits_to_keep=1)  # [B, 1, V]
    return self.sampler(logits[:, -1, :])              # [B, 1]
```

**generate() M2 路径**展示了 ForwardContext 的完整用法：

```python
adapter = SingleCacheAdapter(kv_cache)
adapter.bind_kv_cache(engine.model)        # 把 cache 注入每层 Attention.kv_cache

# Prefill
position_ids = torch.arange(T_p).unsqueeze(0)
metadata = adapter.make_prefill_metadata(input_ids, position_ids)
with set_forward_context(metadata):        # 全局上下文
    logits = engine.model(input_ids, positions=position_ids)

# Decode（循环）
pos = torch.tensor([[kv_cache.cur_len]])
metadata = adapter.make_decode_metadata(next_token, pos)
with set_forward_context(metadata):
    logits = engine.model(next_token, positions=pos, logits_to_keep=1)
```

### 12.2 统一主循环：loop.py

M3 和 M4 都进入 `batch_generate_loop()`（234L），这是整个引擎的核心：

```python
def batch_generate_loop(model, sampler, scheduler, adapter, prompts, ...):
    adapter.bind_kv_cache(model)

    while scheduler.has_unfinished():
        # ── 1. Admit + Allocate ──
        admitted = []
        while scheduler.waiting:
            req = scheduler.waiting[0]
            if not adapter.can_admit(req.prompt_ids.shape[1]):
                break
            scheduler.waiting.popleft()
            req.status = RequestStatus.RUNNING
            scheduler.running[req.request_id] = req
            adapter.allocate(req.request_id, prompt_len)
            admitted.append(req)

        # ── 2. Batched Prefill ──
        if admitted:
            input_ids, positions = _build_prefill_batch(admitted, device)  # padded batch
            metadata = adapter.make_prefill_metadata(input_ids, positions, request_ids=...)
            with set_forward_context(metadata):
                logits = model(input_ids, positions=positions)
            for i, req in enumerate(admitted):
                plen = req.prompt_ids.shape[1]
                req.last_token = sampler(logits[i, plen - 1, :].unsqueeze(0))
            adapter.set_seq_lens(admitted)

        # ── 3. Decode ──
        running = list(scheduler.running.values())
        adapter.prepare_decode([req.request_id for req in running])
        next_tokens, positions = _build_decode_batch(running, device)
        metadata = adapter.make_decode_metadata(next_tokens, positions)
        with set_forward_context(metadata):
            logits = model(next_tokens, positions=positions)
        sampled = sampler(logits[:, -1, :])

        # ── 4. Update + Finish ──
        for req, tok in zip(running, sampled):
            req.last_token = tok.unsqueeze(0)
            req.num_generated += 1
            req.seq_len += 1
            if req.num_generated >= req.max_new_tokens or (eos and tok == eos):
                scheduler.mark_finished(req)
                adapter.free(req.request_id)

    return collect_results(scheduler)
```

关键设计点：

1. **admit-allocate 交替**：每条请求先检查容量 `can_admit()`，再 `allocate()`，然后才检查下一条。这避免了批量 admit 后 allocate 发现容量不够的问题。

2. **batched prefill**：所有新请求 padded 到最长 prompt 长度，一次 forward 处理。每请求取 `logits[i, plen-1, :]` 采样（跳过 padding 位置）。

3. **adapter 多态**：loop.py 不区分 M3 slot 还是 M4 block，所有 cache 操作通过 adapter 接口。

### 12.3 CacheAdapter：统一 cache 差异

`cache/adapter.py`（315L）定义了 CacheAdapter Protocol 和三种实现。每个 adapter 封装了 cache 的创建、绑定、metadata 构造和生命周期管理。

**CacheAdapter Protocol**（8 个方法）：

```python
class CacheAdapter(Protocol):
    def can_admit(self, prompt_len: int) -> bool: ...
    def allocate(self, request_id: str, prompt_len: int) -> None: ...
    def free(self, request_id: str) -> None: ...
    def bind_kv_cache(self, model) -> None: ...
    def make_prefill_metadata(self, input_ids, positions, request_ids=None) -> AttentionMetadata: ...
    def make_decode_metadata(self, next_tokens, positions) -> AttentionMetadata: ...
    def prepare_decode(self, request_ids: list[str]) -> None: ...
    def set_seq_lens(self, requests) -> None: ...
```

**BatchedCacheAdapter**（M3）核心逻辑：

```python
def allocate(self, request_id, prompt_len):
    slot = self.cache.slot_manager.allocate_slot()           # 从空闲 slot 池取一个
    self.cache.slot_manager.req_to_slot[request_id] = slot   # 记录 request_id → slot

def bind_kv_cache(self, model):
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.attn.kv_cache = self.cache.layers[i]  # BatchedLayerKVCache

def make_prefill_metadata(self, input_ids, positions, request_ids=None):
    slots = [self.cache.slot_manager.req_to_slot[rid] for rid in request_ids]
    return AttentionMetadata(num_seqs=B, seq_lens=seq_lens, slot_mapping=slots)

def prepare_decode(self, request_ids):
    for rid in request_ids:
        slot = self.cache.slot_manager.req_to_slot[rid]
        self.cache.seq_lens[slot] += 1   # decode 前 seq_lens 自增
```

**PagedCacheAdapter**（M4）核心差异：

```python
def allocate(self, request_id, prompt_len):
    self.cache.allocate_request(request_id, prompt_len)  # 内部分配 ceil(plen/bs) 个 block

def prepare_decode(self, request_ids):
    for rid in request_ids:
        self.cache.append_token(rid)  # 如当前 block 满，分配新 block

def make_decode_metadata(self, next_tokens, positions):
    block_table = self.cache.gather_block_table(request_ids)  # [B, max_blocks]
    return AttentionMetadata(num_seqs=B, seq_lens=seq_lens, block_table=block_table)
```

关键区别：M3 用 `slot_mapping` 直接索引物理位置；M4 用 `block_table` 间接寻址。

### 12.4 Attention 层：cache RW 分发

`model/attention.py`（433L）中 `Attention.forward()` 是 cache 读写发生的地方：

```python
def forward(self, q, k, v):
    # 1. Cache RW（根据 kv_cache 类型分发）
    if self.kv_cache is None:
        pass                                        # M1: 无 cache
    elif isinstance(self.kv_cache, PagedKVCache):
        k, v, paged_valid_lens = self._paged_cache_rw(k, v, metadata)   # M4
    elif isinstance(self.kv_cache, BatchedLayerKVCache):
        k, v, cache_positions = self._batched_cache_rw(k, v, metadata)  # M3
    elif isinstance(self.kv_cache, LayerKVCache):
        k, v, cache_position = self._single_cache_rw(k, v, metadata)   # M2

    # 2. GQA repeat_kv
    k = repeat_kv(k, self.num_key_value_groups)

    # 3. Attention 计算
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
    # causal mask（seq_len > 1 时）
    # valid_lens mask（paged/batched 时屏蔽 padding）
    attn_weights = softmax(attn_weights, dim=-1, dtype=float32)
    return matmul(attn_weights, v)
```

**M3 _batched_cache_rw** 的 prefill 路径：

```python
# prefill: 每个请求写整段 KV 到自己的 slot
for i in range(B):
    slot = int(slot_mapping[i])
    plen = int(seq_lens[i])
    cache.k[slot, :, :plen, :] = k[i, :, :plen, :]  # 只写有效位置（非 padded）
    cache.v[slot, :, :plen, :] = v[i, :, :plen, :]

# gather: 读取所有 slot 的 KV 拼成 batch
k = cache.k[slot_mapping, :, :max_seq_lens, :]   # [B, n_kv, max_len, D]
return k, v, seq_lens  # seq_lens 触发 valid_lens mask
```

**M4 _paged_cache_rw** 的 scatter/gather：

```python
# scatter: 把 padded batch 的 K/V 写入物理 block
slot_mapping = paged_cache.make_slot_mapping(request_ids, seq_lens)
flat_k = flatten_valid_kv(k, seq_lens)             # [total_tokens, n_kv, D]
for layer_idx, layer_cache in enumerate(paged_cache.layers):
    flat_cache = layer_cache.k.view(-1, n_kv, D)   # 展平前两维
    flat_cache[slot_mapping] = flat_k               # scatter 到物理位置

# gather: 按 block table 读取 KV 拼成连续序列
block_table = metadata.block_table                 # [B, max_blocks]
gathered_k = layer_cache.k[block_table]            # [B, max_blocks, bs, n_kv, D]
k = gathered_k.reshape(B, max_blocks * bs, n_kv, D).transpose(1, 2)  # [B, n_kv, L_pad, D]

# NaN 安全：清零 padding 位置
k = k.masked_fill(~valid_mask, 0)
return k, v, valid_lens
```

### 12.5 ForwardContext：全局上下文传递

`engine/context.py`（110L）定义了两个核心数据结构和一个 context manager：

```python
@dataclass
class AttentionMetadata:
    num_seqs: int                       # batch 中请求数
    seq_lens: torch.Tensor              # [B] 每请求序列长度
    slot_mapping: torch.Tensor | None   # [B] M3 slot 映射
    block_table: torch.Tensor | None    # [B, max_blocks] M4 block 表

@dataclass
class ForwardContext:
    attn_metadata: AttentionMetadata

_forward_context: ForwardContext | None = None

@contextmanager
def set_forward_context(attn_metadata):
    global _forward_context
    _forward_context = ForwardContext(attn_metadata)
    try:
        yield
    finally:
        _forward_context = None
```

模型 forward 签名只有 `(input_ids, positions)`，cache 和 metadata 完全不经过参数传递。Attention 层通过 `get_forward_context().attn_metadata` 获取当前轮的元数据。

### 12.6 Scheduler + RequestState

`scheduler/request.py`（75L）定义请求生命周期：

```python
class RequestStatus(Enum):
    WAITING = 0    # 等待 admit
    RUNNING = 1    # 在 running 队列中
    FINISHED = 2   # 生成完毕

@dataclass
class RequestState:
    request_id: str
    prompt_ids: torch.Tensor      # [1, T_p]
    max_new_tokens: int
    eos_token_id: int | None
    last_token: torch.Tensor | None = None
    num_generated: int = 0
    generated_tokens: list = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    slot_id: int | None = None
    seq_len: int = 0
```

`scheduler/fcfs.py`（100L）实现先来先服务调度：

```python
class FCFSScheduler:
    def submit(self, req): self.waiting.append(req)
    def has_unfinished(self): return self.waiting or self.running
    def mark_finished(self, req):
        req.status = RequestStatus.FINISHED
        del self.running[req.request_id]
        self.finished[req.request_id] = req
```

FCFS 策略：队首请求不够容量就停止 admit，不跳过头部请求（对齐 vLLM V1 的简化版，不做 preemption）。

### 12.7 完整数据流追踪

一个请求从输入到输出的完整路径（M3 batched 为例）：

```
1. batch_generate()
   → 创建 BatchedKVCache([S, n_kv, max_seq_len, D])
   → 创建 BatchedCacheAdapter + FCFSScheduler
   → 提交 RequestState 到 waiting 队列

2. batch_generate_loop() 第一轮迭代
   → can_admit(5) = True → admit → allocate() 分配 slot=0
   → _build_prefill_batch() → input_ids=[1,1,2,3,5], positions=[0,1,2,3,4]
   → adapter.make_prefill_metadata() → AttentionMetadata(slot_mapping=[0], seq_lens=[5])
   → set_forward_context(metadata)

3. model(input_ids, positions=positions)
   → Qwen3ForCausalLM.forward()
     → 28 层 Qwen3DecoderLayer
       → Qwen3Attention: q_proj → q_norm → rope → Attention
         → Attention.forward(q, k, v)
           → _batched_cache_rw: cache.k[0, :, :5, :] = k[:, :5, :]  # 写 slot 0
           → gather: k = cache.k[[0], :, :5, :]                     # 读 slot 0
           → causal mask → softmax → matmul → output
       → SwiGLUMLP
     → lm_head → logits [1, 5, V]

4. sampler(logits[0, 4, :]) → next_token = 7
   → req.last_token = 7, req.num_generated = 1, req.seq_len = 5

5. 后续轮次 Decode
   → prepare_decode(["0"]) → cache.seq_lens[0] += 1  (now 6)
   → _build_decode_batch() → next_tokens=[[7]], positions=[[5]]
   → make_decode_metadata() → AttentionMetadata(slot_mapping=[0], seq_lens=[6])
   → model(next_tokens, positions)
     → Attention._batched_cache_rw (decode path):
       cache.k[0, :, 5, :] = k[0]  # 写位置 5
       k = cache.k[0, :, :6, :]    # 读 [0:6]
   → sampler → next_token = 42

6. req.num_generated >= max_new_tokens
   → mark_finished(req) → adapter.free("0") → slot 0 释放
```
