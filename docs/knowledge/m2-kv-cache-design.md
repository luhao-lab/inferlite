# inferlite M2 设计文档：KV Cache

> **状态**：✅ 已完成（2026-06-29）
> **作者**：luhao
> **基于**：M1 tag `m1/naive-forward`（95 单测全通过，数值对齐 transformers）

---

## 摘要

M1 每个 decode 步都重跑完整序列，计算量 O(N·T²)。M2 缓存 decode 阶段不变的 K/V 向量，单步降至 O(T)，效率提升约 T 倍（T=600 时提升 600×）。代价是预分配约 117 MB 显存，需同步支持 MPS/CUDA + bf16。

---

## 符号说明

本文统一使用以下符号，避免歧义：

| 符号 | 含义 | Qwen3-0.6B 典型值 |
|------|------|----------------|
| B | batch size（批大小） | 1（推理通常 B=1） |
| T_p | prompt 长度（prefill 阶段的 token 数） | 100～1024 |
| N | 最大生成 token 数 | 100～512 |
| T | 当前序列总长 = T_p + 已生成 token 数，最大值 T_p + N | ≤ L_max |
| L_max | `max_seq_len`，KV Cache 的最大容量，generate 前由用户指定 | 1024（默认） |
| L | num_hidden_layers，Transformer 层数 | 28 |
| H | hidden_size，隐藏层维度 | 1024 |
| H_q | num_attention_heads，Q 的注意力头数 | 16 |
| H_kv | num_key_value_heads，KV 的注意力头数（GQA < MHA） | 8 |
| D | head_dim = H / H_q，每个头的维度 | 64 |
| V | vocab_size，词表大小 | 151,936 |

> **GQA**（Grouped Query Attention）：H_kv < H_q，多个 Q head 共享同一组 K/V head。Qwen3-0.6B 中 H_q=16，H_kv=8，每 2 个 Q head 共享 1 对 KV。KV Cache 大小与 H_kv 成正比，GQA 直接减少 cache 显存。

---

## 1. 背景与问题

### 1.1 自回归推理的两个阶段

| 阶段 | 输入规模 | Q、K、V 形状（单层，GQA） | Attention 计算量 | 可缓存？ |
|------|---------|------------------------|----------------|---------|
| **Prefill** | T_p 个 token 并行 | Q: [B, H_q, T_p, D]，K/V: [B, H_kv, T_p, D] | O(T_p²)，T_p × T_p 矩阵乘 | 否——token 间两两计算，无法跳过 |
| **Decode**（每步） | 1 个 token | Q: [B, H_q, 1, D]，K/V: [B, H_kv, T, D]（T 含历史） | O(T)，1 × T 向量乘 | **是**——历史 K/V 不受新 token 影响 |

KV Cache 只能优化 decode 阶段，不影响 prefill。

### 1.2 M1 的浪费在哪里

M1 无 KV Cache，每个 decode 步都把**完整历史**过一遍：

```
decode 第 k 步（k = 1..N）：
  输入序列长度 = T_p + k
  每层 Attention：Q[B, H_q, T_p+k, D] @ K^T[B, H_q, T_p+k, D]，计算量 O((T_p+k)²)
```

N 步的总 Attention 计算量：

```
M1 总量 ≈ sum_{k=1}^{N} (T_p + k)² ≈ O(N · T²)    其中 T = T_p + N（最终序列长度）
```

**问题根源**：decode 第 k 步中，前 T_p + k - 1 个 token 的 K/V 向量与第 k-1 步计算结果**完全相同**（causal mask 保证），但 M1 每步都从头重算。

### 1.3 KV Cache 的效果

causal mask 的数学性质保证：位置 t 的 K/V 只依赖 t 及之前的 token，不受后续 token 影响。因此 prefill 后每个位置的 K/V 在整个 decode 过程中**永远不变**，可以缓存起来。

缓存 K/V 后，decode 第 k 步（k = 1..N）：

```
1. 只计算当前 1 个 token 的 Q：shape [B, H_q, 1, D]
2. 读取 cache 中前 T_p + k - 1 个 K/V：shape [B, H_kv, T_p+k-1, D]
3. Attention：Q[B, H_q, 1, D] @ K^T[B, H_q, T_p+k-1, D]，计算量 O(T)
4. 把当前 token 的 K/V 写入 cache 的第 T_p+k 个位置
```

N 步总计算量：

```
M2 总量 = O(N · T)    比 M1 的 O(N · T²) 少了 T 倍
```

**以 Qwen3-0.6B 为例**（T_p=100，N=500，T≈600）：

| 指标 | M1（无 cache） | M2（有 cache） |
|------|--------------|--------------|
| decode 单步 Attention | O(T²) ≈ 360,000 | O(T) ≈ 600 |
| N=500 步总量 | O(N·T²) ≈ 1.8 亿 | O(N·T) ≈ 30 万 |
| 理论加速比 | 1× | **600×** |

**实测结果**（Mac M3 Pro，MPS，bfloat16，gen_tokens=128，见 `bench/results/2026-06-29-m2-kv-cache-mps-bf16.md`）：

| prompt_tokens | M1 tok/s | M2 tok/s | 实测加速比 |
|--------------|---------|---------|----------|
| 32 | 13.8 | 24.9 | 1.80× |
| 64 | 12.7 | 28.9 | 2.27× |
| 128 | 9.6 | 25.7 | 2.67× |
| 256 | 6.2 | 24.1 | 3.91× |
| 512 | 3.3 | 24.1 | **7.36×** |

**理论 600× vs 实测 7.36× 的差距说明**：
- 理论值假设 Attention 是唯一瓶颈。实测中，每步 decode 的大头是把 28 层权重从内存读一遍（约数百 MB），这是 M1/M2 共有的 I/O 开销，KV Cache 无法消除。
- 对于 Qwen3-0.6B（0.6B 参数 × 2 bytes ≈ 1.2 GB），单步权重读取远比 Attention 计算耗时，导致实测加速比远低于理论上限。
- 权重越大、序列越长，实测加速比越高；M3 Continuous Batching 多请求共享权重读取后，收益会更显著。

### 1.4 代价：显存

KV Cache 为每层、每个位置预留 K/V 存储空间，总显存：

```
显存(bytes) = 2 × B × L × L_max × H_kv × D × dtype_bytes
              ↑K+V  ↑批   ↑层数  ↑最大长度  ↑KV头 ↑头维度
```

| 参数 | 符号 | Qwen3-0.6B 值 | 显存系数 | 说明 |
|------|------|-------------|--------|------|
| batch size | B | 1（推理默认） | ×1 | 见 §3 ADR-01 中 batch 讨论 |
| 层数 | L | 28 | ×28 | 每层独立缓存 |
| 最大序列长度 | L_max | 1024 | ×1024 | **最大影响因子**，线性 |
| KV head 数 | H_kv | 8 | ×8 | GQA 已比 MHA 减半 |
| 每头维度 | D | 64 | ×64 | 固定，由模型决定 |
| dtype | — | bf16=2B | ×2 | fp32 翻倍 |

```
Qwen3-0.6B，B=1，L_max=1024，bf16：
  2 × 1 × 28 × 1024 × 8 × 64 × 2 = 58 MB

若 L_max=4096：  58 × 4 = 233 MB
若用 MHA（H_kv=16）：58 × 2 = 117 MB
```

> 注：Qwen3-0.6B 的 `head_dim` 实为 64（H=1024，H_q=16，D=H/H_q=64），之前文档错写为 128，上方已更正。

### L_max（max_seq_len）如何确定？

L_max 是静态预分配方案的关键参数，**必须在 generate 之前由调用方指定**：

```
T_p + N ≤ L_max
   ↑          ↑
prompt长度  最大生成数   需满足此约束，否则 cache 越界
```

实践中有三种做法：

| 做法 | 适合场景 | 缺点 |
|------|---------|------|
| 固定 L_max（如 1024） | 大多数对话场景，默认推荐 | 超长 prompt 会报错 |
| L_max = T_p + max_new_tokens | 完全精确，零浪费 | 每次 generate 需要额外传参 |
| L_max = model.config.max_position_embeddings | 模型能支持的最大值 | 显存占用最大 |

M2 的策略：`from_config` 工厂方法接收 `max_seq_len` 参数，默认值 1024，可由 `--max-seq-len` CLI 参数覆盖。若实际 T > L_max，在写入时触发 IndexError（明确报错，优于静默截断）。

**结论**：`L_max` 是 KV Cache 显存的最大决定因素，需要在 generate 前由调用方根据使用场景权衡显存和灵活性。

---

## 2. 调研：主流框架怎么做

调研了三个层次的框架，重点理解三个问题：**内存如何分配**、**K/V 如何读写**、**position_ids 如何传递**。

> **参考代码**：[transformers `cache_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)（StaticCache/DynamicCache）、[transformers Qwen3 attention](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)（position_embeddings 统一计算）、[nano-vllm `attention.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nano_vllm/layers/attention.py)（PagedAttention + Triton kernel）。

### 2.1 transformers（单序列，纯 PyTorch）

transformers 提供两套 Cache，代表两种极端取舍：

**DynamicCache**（默认）— 动态增长，无需预知序列长度：

```python
# cache_utils.py - DynamicLayer.update
def update(self, key_states, value_states, *args, **kwargs):
    self.keys = torch.cat([self.keys, key_states], dim=-2)  # 每步 shape 增长
    self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values
```

- 优点：灵活，序列长度不需要提前知道
- 缺点：每步 `torch.cat` 申请新内存 + 拷贝旧数据，长序列时开销显著

**StaticCache** — 预分配，适合 `torch.compile` 和 cudagraph：

```python
# cache_utils.py - StaticLayer.update
def update(self, key_states, value_states, *args, **kwargs):
    kv_length = key_states.shape[-2]
    cache_position = torch.arange(kv_length) + self.cumulative_length  # 写入的绝对位置
    self.cumulative_length.add_(kv_length)
    self.keys.index_copy_(2, cache_position, key_states)   # 原地写入
    self.values.index_copy_(2, cache_position, value_states)
    return self.keys, self.values  # 返回完整 tensor（含尾部零值）
```

- 优点：decode 阶段零内存分配，内存地址固定（支持 cudagraph）
- 缺点：attention_mask 需要屏蔽尾部零值；MPS 上 `index_copy_` 可能不支持（有 fallback）
- 细节：`cumulative_length` 用 `torch.tensor([0])` 而非 `int`，让 torch.compile 追踪到固定地址避免触发重新编译

**Attention 接口**：Attention 接收整个 Cache 对象，通过 `self.layer_idx` 索引本层，调用 `cache.update(k, v, layer_idx)`。`position_embeddings`（cos/sin）在 `Qwen3Model.forward` 统一计算一次传入所有 28 层（inferlite M1 每层各算，多算了 27 次）。

**position_ids**：Model.forward 自动推算，调用方透明：
```python
past_seen_tokens = cache.get_seq_length()  # 读 cache 里已写入的 token 数
position_ids = torch.arange(seq_len) + past_seen_tokens  # decode 步 = [[cur_len]]
```

---

### 2.2 nano-vllm（分页内存，多序列并发）

nano-vllm 以 ~1200 行实现 vLLM 核心，定位是多序列 continuous batching，KV Cache 设计与 transformers 有根本差异。

**内存结构**：不按"每层一个连续 tensor"，而是用 `BlockManager` 管理固定大小的物理 block（每个 block = N 个 token 的空间），每个请求有自己的 `block_table`（逻辑→物理映射）。

**写入**：Triton kernel + `slot_mapping` 散列写入（PagedAttention），每个 token 写到分配到的物理 slot，不要求连续：

```python
# attention.py - 源码摘录
class Attention(nn.Module):
    def __init__(self, ...):
        self.k_cache = self.v_cache = torch.tensor([])  # 初始空，引擎启动时分配

    def forward(self, q, k, v):
        context = get_context()  # 全局 context，含 slot_mapping / block_tables / is_prefill
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # Triton kernel
        if context.is_prefill:
            o = flash_attn_varlen_func(...)
        else:
            o = flash_attn_with_kvcache(..., block_table=context.block_tables)
        return o
```

**设计思路**：Attention 不接收任何 cache 参数，所有调度信息通过全局 context 传递——这是为了让调度器统一管理多序列的内存分配，Attention 层不感知具体请求。

---

### 2.3 vLLM（生产级，PagedAttention 原型）

与 nano-vllm 同源，核心是 PagedAttention：把 KV Cache 的逻辑地址和物理地址解耦（类比操作系统虚拟内存），不同请求可以共享物理页（prefix caching），几乎消除碎片。KV Cache 的物理内存在服务启动时根据可用显存一次性全部分配完。

---

### 2.4 三框架对比

| 维度 | transformers StaticCache | nano-vllm / vLLM | inferlite M2 |
|------|------------------------|-----------------|--------------|
| 内存策略 | 预分配连续 tensor，max_len 固定 | 全局预分配，block 粒度动态分配 | 预分配连续 tensor，max_len 固定 |
| 写入方式 | `index_copy_` 原地写 | Triton kernel，slot 散列写 | 切片写入 `[:,pos:pos+T,:]`（直观） |
| 读取方式 | 返回完整 tensor（含零值尾部） | FlashAttn + block_table 间接读 | 切片读 `[:,:pos+T,:]`（有效范围） |
| 位置传递 | Model.forward 自动推算（隐式） | 调度器管理，context 传入（隐式） | generate loop 显式维护 `cur_len` |
| 多序列 | 需 padding mask | 天然支持 continuous batching | 不支持（M2 范围外） |
| 依赖 | 纯 PyTorch | Triton + FlashAttention | 纯 PyTorch |
| 适合场景 | 单序列 + torch.compile 优化 | 生产级多请求推理服务 | 单序列学习型实现 |
| **推理性能影响** | decode 零内存分配；支持 cudagraph/torch.compile；attention_mask 需处理尾部零值 | 吞吐量最高；Triton kernel 接近硬件上限；PagedAttention 消除显存碎片；需要 GPU（无 MPS 支持） | decode 零内存分配（切片不分配新内存）；不支持 cudagraph（切片边界动态）；MPS/CUDA 均兼容 |

**inferlite M2 选型定位**：与 transformers StaticCache 最接近，差异是——切片写入替代 `index_copy_`（MPS 兼容，更可读），`cur_len` 在 generate loop 显式维护（而非隐藏在 cache 对象内）。

---

## 3. 方案宏观拆解

> **先建立全局视图，再看细节决策。**

### 3.1 M2 = 四个子问题的组合

M2 的本质是把 M1 的 "每步 full forward" 改造成 "一次 prefill + N 次 decode"。这需要四个子问题协同：

```
子问题 A：KVCache 数据结构
  → 新建 kv_cache.py，定义 LayerKVCache + KVCache
  → 解决：cache 存哪里、怎么分配、怎么重置

子问题 B：Attention 接口扩展
  → 改 attention.py 的 GQAAttention.forward
  → 解决：cache 怎么读写、causal mask 怎么处理

子问题 C：Model 层信息传递
  → 改 qwen3.py 的 Qwen3Model.forward + layers.py 的 DecoderLayer
  → 解决：cache 怎么从 generate loop 传到每层 Attention

子问题 D：Generate Loop 拆分
  → 改 engine/core.py 的 generate() + engine/protocol.py
  → 解决：何时 prefill、何时 decode、cur_len 怎么推进
```

### 3.2 依赖关系与实施顺序

子问题之间存在明确的依赖关系，**必须从底层向上逐步实施**：

```
                  ┌─────────────────┐
                  │  子问题 A        │
                  │  kv_cache.py    │  ← 最先实现，无外部依赖
                  │  LayerKVCache   │
                  │  KVCache        │
                  └────────┬────────┘
                           │ 依赖：提供 LayerKVCache 类型
                           ▼
                  ┌─────────────────┐
                  │  子问题 B        │
                  │  attention.py   │  ← 只依赖 A，可独立测试
                  │  GQAAttention   │
                  └────────┬────────┘
                           │ 依赖：Attention 已支持 cache 参数
                           ▼
                  ┌─────────────────┐
                  │  子问题 C        │
                  │  qwen3.py       │  ← 依赖 A+B
                  │  layers.py      │
                  └────────┬────────┘
                           │ 依赖：整个 model 已支持 kv_cache 参数
                           ▼
                  ┌─────────────────┐
                  │  子问题 D        │
                  │  core.py        │  ← 最后实现，依赖 A+B+C
                  │  protocol.py    │
                  │  cli.py         │
                  └─────────────────┘
```

**为什么是这个顺序？**
- A 先于 B：Attention 需要知道 `LayerKVCache` 的类型才能写接口签名
- B 先于 C：Model 透传参数前需要知道 Attention 接受哪些参数
- C 先于 D：generate loop 调 `model(...)` 前需要 model 已接收 kv_cache

### 3.3 每个子问题影响的文件

| 子问题 | 文件 | 当前状态（M1） | M2 改动 |
|--------|------|--------------|--------|
| **A：数据结构** | `model/kv_cache.py`（新建） | 不存在 | 新增 `LayerKVCache`、`KVCache`、`from_config` |
| | `tests/unit/test_kv_cache.py`（新建） | 不存在 | 分配形状、reset、越界验证 |
| **B：Attention** | `model/attention.py` | `forward(hidden, position_ids)` | 新增 `position_embeddings`、`layer_kv_cache`、`cache_position` 参数；移除内部 rotary_emb；cache 读写；causal mask 改 `T>1` |
| | `tests/unit/test_attn_kv.py`（新建） | 不存在 | prefill/decode 两阶段数值验证 |
| **C：透传** | `model/qwen3.py` | 各层 attention 各自算 rotary_emb | Model 层统一计算 position_embeddings；透传 kv_cache 参数 |
| | `model/layers.py` | `DecoderLayer.forward(hidden, position_ids)` | 透传新增的 cache 参数 |
| **D：Generate** | `engine/protocol.py` | `__call__(input_ids, logits_to_keep)` | 新增 `position_ids`、`kv_cache` 可选参数 |
| | `engine/core.py` | `generate()` 每步 full forward | 拆 prefill/decode 分支；维护 `cur_len` |
| | `cli.py` | 无 device/dtype/max-seq-len 参数 | 新增 `--device`、`--dtype`、`--max-seq-len` |
| | `tests/unit/test_generate_kv.py`（新建） | 不存在 | 有 cache == 无 cache（`torch.equal`）验证 |

### 3.4 整体数据流：从 generate() 到 Attention

M2 完成后，一次 decode 步的数据流如下：

```
generate()                  core.py
  │  cur_token [B,1]
  │  position_ids [[cur_len]]
  │  kv_cache
  ▼
model(cur_token, position_ids, kv_cache)    qwen3.py / protocol.py
  │
  ├─ embed(cur_token) → hidden [B,1,H]
  │
  ├─ position_embeddings = rotary_emb(hidden, position_ids)  ← 只算一次
  │
  └─ for i in range(L):                                       L=28层
       layer[i](hidden, position_embeddings,
                layer_kv_cache=kv_cache.layers[i],           ← 本层cache
                cache_position=kv_cache.cur_len)
         │
         └─ GQAAttention.forward(...)                  attention.py
              │
              ├─ q/k/v projection
              ├─ q_norm / k_norm / RoPE
              ├─ 写入：kv.k[:,:,cur_len:cur_len+1,:] = k    ← 追加
              ├─ 读取：full_k = kv.k[:,:,:cur_len+1,:]     ← 读历史
              └─ Attention(q, full_k, full_v) → output

generate()
  ├─ kv_cache.cur_len += 1                              ← 显式更新
  └─ 采样下一个 token → 继续循环
```

---

## 4. 设计决策

> 每条决策遵循 ADR 格式：**背景 → 决策 → 理由 → 替代方案**。

---

### ADR-01 数据结构：list-of-LayerKVCache（静态预分配）

**背景**：需要在 prefill 后保存所有层的 K/V，decode 时逐步追加。

**决策**：每层一个 `LayerKVCache`（预分配的固定 tensor），`KVCache` 持有所有层。

```python
# inferlite/model/kv_cache.py（新建文件）
@dataclass
class LayerKVCache:
    k: torch.Tensor  # [B, n_kv_heads, max_seq_len, head_dim]
    v: torch.Tensor  # [B, n_kv_heads, max_seq_len, head_dim]

class KVCache:
    def __init__(self, layers: list[LayerKVCache]) -> None:
        self.layers = layers
        self.cur_len: int = 0

    @classmethod
    def from_config(cls, config, batch_size, max_seq_len, dtype, device) -> KVCache: ...

    def reset(self) -> None:
        self.cur_len = 0  # tensor 不清零，下次 prefill 会覆盖写入
```

**KV Cache 内存布局示意**（单层，B=1）：

```
预分配 tensor：shape [1, H_kv, L_max, D]
                     ├──────────────────────────────────────────────┐
位置索引：            0    1    2   ...  T_p-1  T_p  T_p+1 ... L_max-1
                     ├────────────────────┤──────┤──────┤──────────┤
prefill 后写入：      [K_0][K_1][K_2]...[K_Tp-1]  0     0     0  ...
                     └────────────────────┘  ↑                     ↑
                          已写入 T_p 个        cur_len=T_p       未使用

decode 第1步后：      [K_0][K_1]...[K_Tp-1][K_Tp] 0    0   ...
                                              ↑
                                         新写入，cur_len → T_p+1

decode 第k步读取：    full_k = k[:, :, :cur_len+1, :]
                              ↑─── 切片只取有效范围，不含尾部零值 ───┘
```

**理由**：静态预分配消除 decode 阶段的内存分配；每层独立，调试时可以单独检查。

**放弃的方案**：`torch.cat` 动态增长（DynamicCache 风格）——每步分配内存，隐藏实现行为，不适合学习框架。

#### batch=1 vs batch>1 的分配差异

KV Cache 的 shape 中 B 维度与 batch size 严格对齐：

```python
# B=1（单序列推理，M2 主要场景）
K shape: [1, H_kv, L_max, D]    # 58 MB（Qwen3-0.6B，bf16）

# B=4（等长批推理）
K shape: [4, H_kv, L_max, D]    # 58 × 4 = 232 MB

# B=8
K shape: [8, H_kv, L_max, D]    # 58 × 8 = 464 MB
```

**B=1 和 B>1 的关键区别**：

| 问题 | B=1 | B>1（等长） |
|------|-----|-----------|
| 显存 | 最小（B 是线性系数） | B 倍 |
| cur_len 管理 | 所有序列共享同一个 cur_len | 所有序列必须同时 prefill、同步推进 decode，共享 cur_len 合理 |
| causal mask | T×T 矩阵（prefill），skip（decode） | 同左 |
| padding | 不需要 | 需要（若 prompt 长度不等），此时需 attention_mask |
| M2 支持 | 完整支持 | **仅支持等长 batch**（不等长需 padding mask，M2 不做） |

**结论**：M2 `from_config` 接收 `batch_size` 参数，分配 `[B, H_kv, L_max, D]` 的 tensor。B=1 是默认和主要场景。不等长 batch 的 padding mask 支持留 M3+。

---

### ADR-02 cur_len：统一在 KVCache 层管理

**背景**：需要跟踪"cache 中已写入多少个 token"，用于确定写入位置和 decode 步的 position_ids。

**决策**：`KVCache.cur_len` 是唯一事实源，generate loop 显式更新。

```python
kv_cache.reset()

# prefill：一次处理完整 prompt
logits = model(prompt_ids, position_ids=..., kv_cache=kv_cache)
kv_cache.cur_len = T_prompt             # ← 显式更新

# decode loop
for step in range(max_new_tokens):
    pos = torch.tensor([[kv_cache.cur_len]])
    logits = model(next_token, position_ids=pos, kv_cache=kv_cache)
    kv_cache.cur_len += 1               # ← 显式更新
```

**理由**：状态变化在 generate loop 里一目了然，不需要 trace 进 cache 对象内部。所有层共享同一 `cur_len`，消除同步遗漏风险。

**与 transformers 的差异**：transformers 各层各自维护（StaticLayer: `cumulative_length`，DynamicLayer: `keys.shape[-2]`），对用户透明但状态分散——inferlite 有意让状态可见。

---

### ADR-03 Attention 接口：接收 LayerKVCache + cache_position

**背景**：Attention 需要读写 cache，但应尽量降低与 cache 全局状态的耦合。

**决策**：`GQAAttention.forward` 只接收本层的 `LayerKVCache` 和 `cache_position`（即 `cur_len` 的值）。

```python
# model/attention.py - GQAAttention.forward 新签名
def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],  # 由 Qwen3Model 统一计算
    layer_kv_cache: LayerKVCache | None = None,
    cache_position: int | None = None,     # = kv_cache.cur_len，由 generate loop 透传
) -> torch.Tensor:
    ...
    if layer_kv_cache is not None:
        # 写入当前 token(s)：切片赋值，原地写入，不分配新内存
        layer_kv_cache.k[:, :, cache_position:cache_position + T, :] = k
        layer_kv_cache.v[:, :, cache_position:cache_position + T, :] = v
        # 读取完整有效历史：切片读，只取 [0, cache_position+T) 范围
        k = layer_kv_cache.k[:, :, :cache_position + T, :]
        v = layer_kv_cache.v[:, :, :cache_position + T, :]
    ...
```

**Prefill vs Decode 时序对比**：

```
Prefill 阶段（T = T_p，cache_position = 0）：
─────────────────────────────────────────────
  输入：hidden [B, T_p, H]
  k/v projection → k: [B, H_kv, T_p, D]
  写入：kv.k[:, :, 0:T_p, :] = k        ← 一次写满 T_p 个位置
  读取：full_k = kv.k[:, :, :T_p, :]    ← 读回 T_p 个（就是刚写的）
  Attention：Q[T_p × T_p] 带 causal mask
  T > 1，构造 causal mask ✓

Decode 第1步（T = 1，cache_position = T_p）：
─────────────────────────────────────────────
  输入：hidden [B, 1, H]
  k/v projection → k: [B, H_kv, 1, D]
  写入：kv.k[:, :, T_p:T_p+1, :] = k   ← 追加到第 T_p 个位置
  读取：full_k = kv.k[:, :, :T_p+1, :] ← 读历史 T_p+1 个
  Attention：Q[1 × (T_p+1)] 无需 causal mask
  T == 1，跳过 causal mask ✓
```

**理由**：Attention 不感知"这是第几层"，降低耦合；切片写入比 `index_copy_` 更直观（读者直接看到"写哪里、读哪里"），MPS 兼容。单层 Attention 可以独立测试，不需要构造完整 KVCache。

**与 transformers 的差异**：transformers 传入整个 cache + layer_idx，update() 内部处理写入逻辑（封装更好，但对读者不透明）。

---

### ADR-04 position_ids：decode 步用绝对位置

**背景**：RoPE 需要 position_ids 来计算 cos/sin，decode 步每次只处理 1 个 token。

**决策**：decode 步 `position_ids = [[kv_cache.cur_len]]`，用绝对位置。

```python
pos = torch.tensor([[kv_cache.cur_len]], device=device)  # 绝对位置，不是 [[0]]
```

**理由**：RoPE 的正确性依赖 q_m 和 k_n 都是绝对位置，`q_m · k_n` 才能编码正确的相对距离 (m-n)。写 `[[0]]` 是沉默 bug：推理不报错，但每步都认为当前 token 在序列开头，RoPE 失效。

---

### ADR-05 causal mask：用 T > 1 判断

**背景**：prefill 需要 causal mask（prompt 内部的因果性），decode 步不需要（当前 token 只看历史，没有"未来"）。

**决策**：`if T > 1:` 构造 causal mask。

```python
# model/attention.py - GQAAttention.forward 中
if T > 1:
    causal_mask = torch.triu(torch.ones(T, T_k, dtype=torch.bool), diagonal=1)[None, None]
    attn_scores = attn_scores.masked_fill(causal_mask, -inf)
```

**四种情况验证**：

| 场景 | T | T > 1 | 需要 mask | 结论 |
|------|---|-------|---------|------|
| M1 full forward | prompt_len | ✓ | ✓ | 正确 |
| M2 prefill | prompt_len | ✓ | ✓ | 正确 |
| M2 decode | 1 | ✗ | ✗ | 正确 |
| 首步 decode | 1 | ✗ | ✗ | 正确 |

**放弃的方案**：`if layer_kv_cache is None`——prefill 时有 cache 但仍需 mask，此条件是错的。

---

### ADR-06 向后兼容：kv_cache=None 退化为 M1

**背景**：M1 的 95 个单测需要继续通过，同时 kv_cache=None 可以用作数值对齐基准。

**决策**：`kv_cache=None` 时走 M1 full forward 路径，不改动任何 M1 逻辑。

```python
# engine/core.py - generate() 改造后
def generate(..., kv_cache=None):
    if kv_cache is not None:
        # M2：prefill + decode loop
        logits = model(input_ids, position_ids=arange(T_p), kv_cache=kv_cache)
        kv_cache.cur_len = T_p
        for step in range(max_new_tokens):
            pos = torch.tensor([[kv_cache.cur_len]])
            next_token = engine.step_with_cache(cur_token, pos, kv_cache)
            kv_cache.cur_len += 1
            ...
    else:
        # M1：full forward（原逻辑不变）
        for _ in range(max_new_tokens):
            next_token = engine.step(input_ids)
            ...
```

**验收标准**：`generate(kv_cache=None)` 输出必须与 `generate(kv_cache=KVCache(...))` 完全一致（`torch.equal`）。

---

### ADR-07（顺带修复）position_embeddings 在 Qwen3Model 统一计算

**背景**：M1 中 `GQAAttention` 各自调用 `self.rotary_emb` 计算 cos/sin，28 层 = 28 次重复计算。

**决策**：把 `rotary_emb` 调用移到 `Qwen3Model.forward`，计算一次，传入所有层。

```python
# model/qwen3.py - Qwen3Model.forward 改造后
position_embeddings = self.rotary_emb(hidden_states, position_ids)  # 只算一次
for i, layer in enumerate(self.layers):
    hidden_states = layer(
        hidden_states,
        position_embeddings=position_embeddings,   # 透传
        layer_kv_cache=kv_cache.layers[i] if kv_cache else None,
        cache_position=kv_cache.cur_len if kv_cache else None,
    )
```

**理由**：与 transformers 对齐（transformers 也是在 Model 层统一计算），消除 27 次多余计算。

---

## 5. 影响范围

涉及文件的完整列表及改动说明见 §3.3，含新增文件（`model/kv_cache.py`、三个测试文件）和修改文件（`attention.py`、`qwen3.py`、`layers.py`、`protocol.py`、`core.py`、`cli.py`）。

额外产出：

| 文件 | 说明 |
|------|------|
| `docs/knowledge/m2-kv-cache-design.md` | 本设计文档 |
| `scripts/bench_kv_cache.py` | M1 vs M2 性能对比 benchmark 脚本 |
| `bench/results/2026-06-29-m2-kv-cache-mps-bf16.md` | 实测结果归档 |
| `docs/plan/m2-kv-cache-article.md`（待写） | 对外文章（知乎发布版） |

---

## 6. 测试策略

**核心正确性测试**（最高优先级）：

```python
def test_kv_cache_output_matches_no_cache():
    """有 cache 的输出必须与无 cache 完全一致。"""
    output_no_cache   = generate(engine, prompt, max_new_tokens=10, kv_cache=None)
    output_with_cache = generate(engine, prompt, max_new_tokens=10, kv_cache=kv_cache)
    assert torch.equal(output_no_cache, output_with_cache)
```

**测试矩阵**：

| 测试文件 | 核心验收点 |
|---------|-----------|
| `test_kv_cache.py` | from_config 形状正确，reset 后 cur_len=0 |
| `test_attention_kv.py` | prefill 写入正确；decode 步追加写入；causal mask T>1/T=1 分支 |
| `test_generate_kv.py` | **有 cache == 无 cache**（torch.equal）；EOS 停止 |
| 全部 M1 单测（95 个） | kv_cache=None 路径向后兼容 |

---

## 7. 实施计划

| 步骤 | 子问题 | 内容 | 验收 | 状态 |
|------|--------|------|------|------|
| S1 | A | 新建 `kv_cache.py` + `test_kv_cache.py` | 单测通过 | ✅ |
| S2 | B | 改 `attention.py`（接口 + cache 逻辑 + causal mask） | `test_attention_kv.py` 通过 | ✅ |
| S3 | C | 改 `qwen3.py`（position_embeddings 统一计算 + cache 透传） | 全部 95 个 M1 单测继续通过 | ✅ |
| S4 | D | 改 `protocol.py` + `core.py`（prefill/decode 拆分） | `test_generate_kv.py` 通过 | ✅ |
| S5 | D | 改 `cli.py`（device/dtype/max-seq-len 参数） | Qwen3-0.6B 在 MPS 上跑通 | ✅ |
| S6 | — | benchmark 脚本 + 结果归档 | `bench/results/` 有实测数据 | ✅ |
| S7 | — | tag `m2/kv-cache`，生成知乎版 | 文章发布 | ⬜ |

---

## 8. 范围外（M3+）

- PagedAttention / 动态内存管理（见 nano-vllm）
- Continuous batching
- top-k / top-p / temperature sampling
- 量化（int8/int4）
- padding mask（不等长 batch）

---

## 附录：技术设计文档写作框架

> 本节抽象本文的写作方法，供后续技术文档复用。

### 框架总览

好的技术设计文档回答三个问题：**为什么做**（Why）、**怎么想清楚**（How to think）、**怎么做**（How to do）。对应三条主干：

```
背景与问题  →  调研与选型  →  方案设计
（Why）        （How to think）  （How to do）
```

文档其余部分（测试、实施计划、范围外）是这三条主干的**验收与边界说明**，而非主体。

---

### 核心理念

**理念 1：先建立读者的"心智模型"，再讲细节**

每个技术文档都有一个核心认知门槛——读者不跨过去，后面的决策无从理解。本文的门槛是"KV Cache 为什么只优化 decode 而不优化 prefill"，所以 §1 花了大量篇幅在两个阶段的计算量分析上，而不是直接开始讲数据结构。

规律：**先建立读者需要的认知，再介绍你的设计**。

**理念 2：调研不是罗列，是为选型服务**

调研的终点是对比表。对比表的维度要与你即将做的设计决策直接挂钩（本文对比"内存分配策略""位置传递方式"，正是后续 ADR-01/ADR-02 的决策轴）。调研完读者应该能预测出你会选哪个方案，以及为什么。

规律：**调研 = 设计决策的前置论证，不是背景知识科普**。

**理念 3：宏观拆解先于微观决策**

设计决策（ADR）容易陷入"给答案没给思路"的陷阱——读者看完知道你选了什么，但不知道你是怎么把问题分解的。宏观拆解（§3）解决的正是这个问题：先把方案分成几个子问题，说清楚子问题之间的依赖关系，读者才能理解为什么要按这个顺序实施、为什么不能跳过某步。

规律：**先说"这个方案分几个子问题、依赖关系是什么"，再说每个子问题的具体决策**。

**理念 4：每个决策都要有可被证伪的"放弃方案"**

ADR 格式（背景 → 决策 → 理由 → 放弃的方案）强迫你说出"我考虑过哪些替代方案、为什么没选"。这做两件事：一是证明你确实想清楚了（而不是只想了一种），二是防止读者重新发明被否定过的轮子。

规律：**每个设计决策必须有至少一个被明确拒绝的替代方案**。

**理念 5：可视化是说清楚"时序"和"布局"的唯一手段**

文字描述流程容易产生歧义；数字描述内存布局让人脑筋打结。ASCII 图的作用不是好看，是把"动词"变成"图"——"追加写入"变成 `[K_0][K_1]...[K_Tp-1][K_Tp]`，"切片读有效范围"变成箭头标注。

规律：**凡是涉及时序（A 先于 B）或布局（某块内存长什么样），用图不用文字**。

---

### 文档结构模板

```
# [项目] [里程碑] 设计文档：[核心特性]

> 状态 / 作者 / 基于版本

## 摘要（3 句话）
  1. 当前问题是什么（量化）
  2. 方案是什么（一句话）
  3. 代价是什么（量化）

## 符号说明
  统一定义文档内所有符号，避免歧义

## 1. 背景与问题
  1.1 当前系统的工作方式（图）
  1.2 浪费/瓶颈在哪里（公式/量化）
  1.3 解决后的效果（对比表）
  1.4 引入的代价（量化）

## 2. 调研：主流方案怎么做
  > 开头标注参考代码/文档来源
  2.x 各方案（内存策略 + 读写方式 + 关键取舍）
  2.最后 对比表（维度与你的设计决策轴对齐）

## 3. 方案宏观拆解
  3.1 方案 = N 个子问题（列举 + 一句话说每个）
  3.2 依赖关系图（ASCII，说清楚实施顺序）
  3.3 文件影响表（子问题 / 文件 / 当前状态 / 改动）
  3.4 整体数据流（ASCII，从入口到核心逻辑）

## 4. 设计决策（ADR 格式）
  ### ADR-XX 标题
  **背景**：为什么需要做这个决策
  **决策**：选了什么（含代码片段）
  **理由**：为什么这样选（含可视化）
  **放弃的方案**：考虑过什么、为什么否定

## 5. 影响范围
  指向 §3.3，额外列出文档/发布产出

## 6. 测试策略
  核心正确性测试（最高优先级，用代码写出来）
  测试矩阵（文件 / 验收点）

## 7. 实施计划
  表格：步骤 / 对应子问题 / 内容 / 验收标准

## 8. 范围外
  明确列出"本次不做什么"，防止范围蔓延
```

---

### 写作检查清单

开始动笔前：
- [ ] 能否用一个公式/数字说清楚"现在有什么问题"？（如 O(N·T²) vs O(N·T)）
- [ ] 调研覆盖了至少 2 个同类方案，且知道每个方案的核心取舍？
- [ ] 能把方案分解成有明确依赖关系的 3~5 个子问题？

写完后自检：
- [ ] 每个 ADR 都有"放弃的方案"？
- [ ] 时序类/布局类内容有图辅助？
- [ ] 实施计划每步都有可执行的验收标准（不是"写完"，而是"XXX 测试通过"）？
- [ ] "范围外"明确列了，防止读者问"为什么没做 Y"？
