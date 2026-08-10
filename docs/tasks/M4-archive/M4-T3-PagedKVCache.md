# M4-T3 — PagedKVCache

> 实现多层分页 KV 容器：持有各层 K/V tensor，组合 BlockPool 与 BlockTable，负责请求生命周期与 KV 读写。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T3 |
| 里程碑 | M4 — PagedAttention |
| 状态 | ✅ done |
| 前置 | M4-T1 BlockPool ✅、M4-T2 BlockTable ✅ |
| 后续 | M4-T4 — PagedAttention |
| 估时 | 3h |
| 核心文件 | `inferlite/cache/paged_kv_cache.py`（与 T2 同文件） |
| 深入文档 | [`docs/knowledge/m4-paged-attention.md`](../../knowledge/m4-paged-attention.md) |

## 目标

### 要解决什么问题

T1 和 T2 都只处理整数：T1 管"块归谁"，T2 管"块的顺序"。真实的 K/V tensor 还没有出现过。T3 是**编号第一次变成内存地址**的地方。

M3 `BatchedKVCache` 的单层 shape 是：

```text
[max_num_slots, n_kv_heads, max_seq_len, head_dim]
      ↑                          ↑
   每请求一个 slot          每 slot 预留满长度
```

问题是 `max_seq_len` 必须按最坏情况预留。一个只生成 30 个 token 的请求，也占着 4096 个位置的空间。这是 M3 吞吐上不去的根本原因 —— 能同时在跑的请求数被内存卡死。

M4 改为：

```text
[num_blocks, block_size, n_kv_heads, head_dim]
      ↑           ↑
  全局块池    每块只放 block_size 个 token
```

请求按需申请块，用多少占多少。逻辑连续性由 T2 的 block table 维持。

### 做完是什么效果

```python
cache = PagedKVCache.from_config(
    config, num_blocks=64, block_size=16, dtype=torch.float32, device="cpu"
)

# prefill：两个请求已经分别 allocate，K/V 一次批量写入
cache.allocate_request("req-0", prompt_len=20)
cache.allocate_request("req-1", prompt_len=5)
cache.write_prefill(layer_idx=0, request_ids=["req-0", "req-1"], k=k_batch, v=v_batch)

# decode：先逐请求扩展逻辑长度，再一次批量写入本轮的 2 个 token
cache.append_token("req-0")
cache.append_token("req-1")
cache.write_decode(0, ["req-0", "req-1"], k_new_batch, v_new_batch)

# attention 读取：批量 gather，返回 padding 后的 K/V 和有效长度
k, v, valid_lens = cache.gather_kv(layer_idx=0, request_ids=["req-0", "req-1"])

cache.free_request("req-0")          # 归还 req-0 的全部块
```

### 不做什么

- 不改 attention 层。T3 只提供 `gather_kv`，接入 attention 是 T4。
- 不接 engine / scheduler。block admission 是 T5。
- 不做 prefix hash、LRU、CoW（M5）。
- 不修改 M3 `BatchedKVCache`，两条 cache 路径并存直到 T6 验证等价后。
- 不追求 kernel 级性能。消除 padding 浪费需要 kernel 内部按 block table 寻址，属 M9。

### 在推理链路中的位置

```text
M4-T5 BatchEngine（admission）
          |
          v
M4-T4 PagedAttention  <- 消费 gather_kv 的输出
          |
          v
M4-T3 PagedKVCache    <- 本任务：持有 tensor，组合下面两者
     |             |
     v             v
M4-T2 BlockTable   M4-T1 BlockPool
逻辑位置映射       物理 block 分配
```

T3 是**唯一同时认识 BlockPool 和 BlockTable 的类**。它们互不依赖，靠 T3 传递 `block_id` 协作。

### 设计原则

- **编号到地址只在这一层发生**。`block_id` 用作 tensor 第 0 维下标，其余全部交给 PyTorch。
- **生命周期与张量路径分离**。allocate / append / free 由 scheduler 按请求决定；大 K/V tensor 的读写必须是 batch 路径。
- **写入复刻 vLLM 数据流**。先从 BlockTable 生成扁平 `slot_mapping`，再一次 PyTorch scatter；M9 再替换为 kernel。
- **不持有请求语义**。只认 `request_id: str`，不耦合 `RequestState`。
- **M4 分配使用预检查**。单线程、无抢占时，预检查之后没有其他 allocator 介入；M5/M6 引入并发后必须升级事务回滚。
- **数值安全责任显式移交**。`gather_kv` 返回 `valid_lens`，调用方必须据此清零 padding（见 ADR-05）。

## 产出文件

- `inferlite/cache/paged_kv_cache.py::PagedLayerKVCache`
- `inferlite/cache/paged_kv_cache.py::PagedKVCache`
- `tests/unit/test_paged_kv_cache.py`

## 接口骨架

```python
@dataclass
class PagedLayerKVCache:
    """单层 paged KV 数据容器。

    k/v shape: [num_blocks, block_size, n_kv_heads, head_dim]
    """

    k: torch.Tensor
    v: torch.Tensor


class PagedKVCache:
    def __init__(
        self,
        layers: list[PagedLayerKVCache],
        block_pool: BlockPool,
        block_size: int,
    ) -> None: ...

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "PagedKVCache": ...

    # ── 请求生命周期 ──
    def allocate_request(self, request_id: str, prompt_len: int) -> None: ...
    def append_token(self, request_id: str) -> None: ...
    def free_request(self, request_id: str) -> None: ...

    # ── KV 写入（批量 slot scatter）──
    def write_prefill(
        self,
        layer_idx: int,
        request_ids: list[str],
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None: ...
    def write_decode(
        self,
        layer_idx: int,
        request_ids: list[str],
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None: ...

    # ── Attention 读取 ──
    def gather_kv(
        self, layer_idx: int, request_ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def gather_kv_single(
        self, layer_idx: int, request_id: str
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    # ── 容量查询（T5 admission 用）──
    @property
    def num_free_blocks(self) -> int: ...
    def can_allocate(self, prompt_len: int) -> bool: ...
    def seq_len_of(self, request_id: str) -> int: ...
```

相对早期草稿的五处变更：

| 变更 | 理由 |
|---|---|
| `gather_kv` 改批量签名并返回 `valid_lens` | 见 ADR-05。单请求签名会让 decode 每层跑 B 次 Python 循环 |
| 新增 `gather_kv_single` | 作为批量版本的测试 oracle，不走生产路径 |
| `may_append_block` → `append_token` | 原名只说"可能加块"，但它同时递增 `seq_len`。新名字覆盖完整语义 |
| 新增 `num_free_blocks` / `can_allocate` / `seq_len_of` | T5 admission 需要，避免 engine 直接摸 `cache.block_pool` 破坏封装 |
| `__init__` 与 `from_config` 分开 | 与 M3 `BatchedKVCache` 一致，测试可注入小 tensor 而不必造 config |

## ADR-05：paged gather 采用批量高级索引

### 背景

早期草稿的 `gather_kv(layer_idx, request_id)` 是单请求接口，实现方式是按 block 逐个切片再 `torch.cat`。decode 时每层每请求各调一次，Qwen3-0.6B 有 28 层，B 个请求就是 28B 次 Python 循环加 `cat`。

### 关键事实：gather 开销 M3 已经在付

M3 现有实现 `_batched_cache_rw`（`inferlite/model/attention.py:370`）：

```python
k = cache.k[cache_slots, :, :max_len, :]
```

`cache_slots` 是 tensor，这是**高级索引**，PyTorch 会复制数据而非返回 view。也就是说 M3 每层每个 decode 步已经做了一次完整 KV 拷贝。

所以"分页引入了 gather 开销"这个判断不成立。T3 要避免的不是 gather 本身，而是**额外**引入的 Python 循环和 `cat`。

### 决策

`gather_kv` 用一次两级高级索引完成整批读取：

```python
# 1. 各请求 block_ids padding 成 [B, max_num_blocks]，不足处填 0
#    （填 0 安全：这些位置会被 valid_lens 屏蔽）
block_table = ...                      # [B, max_num_blocks], dtype=long

# 2. 一次高级索引取出所有块
k = layer.k[block_table]               # [B, max_num_blocks, block_size, n_kv, D]

# 3. 合并块维与块内维，得到逻辑连续的序列维
b, nb, bs, h, d = k.shape
k = k.reshape(b, nb * bs, h, d)        # [B, L_pad, n_kv, D]

# 4. 转成 attention 需要的布局
k = k.transpose(1, 2)                  # [B, n_kv, L_pad, D]
```

零 Python 循环、零 `cat`，与 M3 同一量级。

### 代价对比

| | 草稿 per-request cat | ADR-05 批量索引 |
|---|---|---|
| 每层 Python 循环 | B 次 | 0 |
| `torch.cat` 调用 | B × num_blocks | 0 |
| 高级索引 | B 次小索引 | 1 次大索引 |
| 与 M3 开销量级 | 更差 | 相同 |
| padding 浪费 | 无 | 补到 `max_num_blocks × block_size` |

padding 浪费是新引入的：700 token 的请求与 30 token 的请求合批时，后者会被补到前者的块数。但这与 M3 补到 `max_len` 是同一性质的问题，且 block 粒度（16）比 M3 的 token 粒度更粗，浪费反而更少。彻底消除需要 kernel 内部按 block table 寻址，留 M9。

### 数值安全：必须复用 lessons L5 的结论

padding 区域的物理块可能从未写入，底层是 `torch.empty` 的未初始化内存，可能含 NaN/Inf。

**仅做 score mask 不够**：mask 能把 attention 概率压到 0，但 value 聚合 `0 × NaN = NaN` 仍会传播，最终 logits 全 NaN，argmax 恒返回 0。这正是 M3 修过的 bug（见 `docs/knowledge/lessons.md` L5）。

因此 `gather_kv` 必须返回 `valid_lens: [B]`，由 T4 据此清零 K/V。这条写入 T3 的 DoD 与 T4 的前置条件。

### 保留 per-request 参考实现

`gather_kv_single(layer_idx, request_id)` 逐位置读取，只用于教学与测试。T3 单测拿它当批量版本的 oracle —— 批量路径的 `reshape` / `transpose` 维度写错时，这条对比能立刻抓住。生产路径只走批量版本。

## 接口合同

### `from_config(config, num_blocks, block_size, dtype, device)`

- 创建 `BlockPool(num_blocks, block_size)` 和 `config.num_hidden_layers` 个 `PagedLayerKVCache`。
- 每层 K/V shape `[num_blocks, block_size, config.num_key_value_heads, config.head_dim]`。
- 用 `torch.empty` 预分配（与 M3 一致，不清零）。**注意由此产生的未初始化内存必须靠 `valid_lens` 屏蔽**。
- `num_blocks <= 0` / `block_size <= 0` 由 `BlockPool` 校验抛 `ValueError`。

### ADR-06：KV 写入采用批量 slot mapping scatter

vLLM V1 的 KV 写入不是逐请求调用：runner 把本轮所有 token 展平，生成每个 token 的物理 `slot_mapping`，再由 `reshape_and_cache` kernel 一次 scatter 到 paged KV cache。T3 采用相同数据流的 PyTorch 教学版，M9 再把 scatter 替换为 kernel。

#### slot 定义与 scatter

```text
slot = physical_block_id * block_size + block_offset
```

每层物理 tensor 的 shape 为 `[num_blocks, block_size, n_kv, D]`，可零拷贝 reshape 为 `[num_blocks * block_size, n_kv, D]`：

```python
flat_cache_k = layer.k.reshape(num_blocks * block_size, n_kv, head_dim)
flat_cache_k[slot_mapping] = flat_k
```

| 名称 | shape | 语义 |
|---|---|---|
| `slot_mapping` | `[total_valid_tokens]`，long | 每个有效 token 的物理写入位置 |
| `flat_k` / `flat_v` | `[total_valid_tokens, n_kv, D]` | 从 batch 输入滤出的有效 token KV |
| `flat_cache_k/v` | `[num_blocks * block_size, n_kv, D]` | 物理 K/V tensor 的 reshape view |

slot mapping 的生成允许用 Python 循环（只处理廉价整数）；大 tensor 搬运必须是一次高级索引 scatter。M4 不写 CUDA kernel。

#### 变长 prefill：有效长度只信任 BlockTable

严格前置：`allocate_request(request_id, prompt_len)` 已为每个请求建立 table，且 `table.seq_len == prompt_len`。因此 batch 的有效长度来自 table，**不额外传 `prompt_lens`**，避免同一长度出现两个事实来源：

```text
request_ids = ["a", "b"]
table.seq_len = [20, 5]
k/v shape = [2, n_kv, 20, D]

写入 k[0, :, 0:20, :] 与 k[1, :, 0:5, :]
绝不写 k[1, :, 5:20, :] 的 batch padding
```

M10 Chunked Prefill 中「本次写入长度」会与「累计 seq_len」不同，届时升级为 `write_tokens(request_ids, start_positions, token_lens, ...)`。

#### 接口合同

**`write_prefill(layer_idx, request_ids, k, v)`**：

- `k/v` shape `[B, n_kv_heads, T_max, head_dim]`。
- `request_ids` 非空、无重复，且 `len(request_ids) == B == k.shape[0] == v.shape[0]`；否则 `ValueError`。
- 每个 id 必须已注册；否则 `KeyError`。`layer_idx` 越界；`IndexError`。
- `T_max >= max(table.seq_len)`；否则 `ValueError`。
- heads / head_dim / dtype / device 必须与目标 layer cache 一致；否则 `ValueError`。
- 有效 token 总数 `sum(table.seq_len)` 必须等于 `slot_mapping.numel()`；每个有效 token 只写一次。

**`write_decode(layer_idx, request_ids, k, v)`**：

- `k/v` shape `[B, n_kv_heads, 1, head_dim]`，B 与 request_ids 同上。
- K/V token 维不为 1 时 `ValueError`。
- 调用方先对每个 id 调 `append_token()`；每个 table 的 `seq_len > 0`，否则 `RuntimeError`。
- slot mapping shape `[B]`，第 i 个 slot 是第 i 个 table 的 `seq_len - 1`。

两个 batch write 都不允许生产路径按 request 循环调用单请求写接口；不存在单请求 `write_prefill` / `write_decode` API。

### `allocate_request(request_id, prompt_len)`

- `request_id` 已存在时 `raise ValueError`（重复注册通常是上层状态机 bug）。
- `prompt_len <= 0` 时 `raise ValueError`。
- 计算 `num_needed = ceil(prompt_len / block_size)`。
- 逐个 `block_pool.allocate()`，`append_block` 到新建的 `BlockTable`，最后 `extend(prompt_len)`。
- 先调用 `block_pool.can_allocate(num_needed)` 预检查；不足时 `raise RuntimeError`，错误消息包含需要与可用的块数。
- M4 是单线程且无抢占，预检查到 allocate 循环之间没有其他 allocator 介入，因此正常路径无需 `try/except` 回滚。
- **升级条件**：M5/M6 若引入并发分配、抢占或可重入 callback，必须改为 `try/finally` 事务回滚；不可继续依赖预检查。

### `append_token(request_id)`

- decode 追加一个 token 前调用。
- `request_id` 未注册时 `raise KeyError`。
- 若 `table.needs_new_block()`，先 `block_pool.allocate()` 并 `append_block`；池空时 `raise RuntimeError`。
- 然后 `table.extend(1)`。
- 命名理由：它同时可能分配块并递增 `seq_len`，`may_append_block` 只描述了前半。

### `free_request(request_id)`

- `request_id` 未注册时 `raise KeyError`。
- 遍历 `table.block_ids` 逐个 `block_pool.free()`，然后从注册表移除。
- M4 无 CoW，一个块只属于一个请求，因此可以无条件释放。M5 有共享后要改成按 ref_count 判断。

### `gather_kv(layer_idx, request_ids) -> (k, v, valid_lens)`

- `request_ids` 为空列表时 `raise ValueError`。
- 任一 `request_id` 未注册 → `KeyError`。
- 返回：
  - `k` / `v`: `[B, n_kv_heads, L_pad, head_dim]`，`L_pad = max_num_blocks * block_size`。
  - `valid_lens`: `[B]`，`dtype=long`，值为各请求的 `seq_len`。
- `L_pad` 是**块对齐后**的长度，通常大于 `max(seq_len)`。调用方不得假设 `L_pad == max(seq_len)`。
- padding 区内容未定义，可能含 NaN。调用方必须用 `valid_lens` 清零 K/V 后再做 attention。

### `gather_kv_single(layer_idx, request_id) -> (k, v)`

- 返回 `[n_kv_heads, seq_len, head_dim]`，**无 padding**，长度精确等于 `seq_len`。
- 逐位置读取，仅供测试与教学。生产路径用批量版本。

### `num_free_blocks -> int`（property）

- 转发 `block_pool.num_free_blocks`。

### `can_allocate(prompt_len) -> bool`

- 判断能否容纳一个 `prompt_len` 长的新请求，即 `block_pool.can_allocate(ceil(prompt_len / block_size))`。
- `prompt_len <= 0` 时 `raise ValueError`。
- 存在的意义是让 T5 做 admission 时不必自己算块数、不必触碰 `block_pool`。

### `seq_len_of(request_id) -> int`

- 返回该请求当前 `seq_len`；未注册 → `KeyError`。
- 避免上层为读长度而直接访问 `block_tables`。

## 算法核心与不变量

### 三层索引的完整落地

```text
pos（逻辑位置）
  |  BlockTable.position_to_block()          T2
  v
(block_id, offset)
  |  layer.k[block_id, offset]               T3 ← 编号在这里变成地址
  v
真实内存
```

### 写入路径

```text
prefill: allocate_request(request, prompt_len)  -> 调度层逐请求分配 ceil(prompt_len/block_size) 块
         write_prefill(batch)                  -> 从所有 table 生成 slot_mapping，一次 scatter

decode:  append_token(request)                 -> 调度层逐请求补块并 seq_len += 1
         write_decode(batch)                   -> B 个 slot，一次 scatter
```

### 状态不变量

```text
每个已注册 request：0 < seq_len <= capacity（由 BlockTable 保证）
sum(len(table.block_ids)) + num_free_blocks == num_blocks   （无泄漏）
每个 block_id 至多属于一个 request（M4 无共享；M5 引入 ref_count > 1）
free_request 后：num_free_blocks 恢复到该请求分配前的值
```

第二条是最重要的对外不变量 —— 它等价于"没有块泄漏"，L0-8 专门测它。

## 实现步骤

1. 整理当前文件：标准库 / 第三方 / 本地 import 顺序，修正 `gather_kv_singe` 与 `request_ids` 拼写，补 `PagedKVCache` docstring；不实现单请求 write。
2. 实现 `_blocks_needed`、`allocate_request`、`append_token`、`free_request`；分配按预检查版实现，并在注释写清单线程前提。
3. 实现 layer / request / batch K/V shape 的共有校验 helper，避免两个 batch write 的合同漂移。
4. 实现 `_make_prefill_slot_mapping`，用小例子逐项与 `position_to_block` 手算值比对。
5. 实现 `write_prefill`：从 `[B, n_kv, T_max, D]` 滤出有效 token、生成 `flat_k/v`、一次 scatter。
6. 实现 `_make_decode_slot_mapping` 与 `write_decode`：每个请求一个 token、一次 scatter。
7. 实现 `gather_kv_single`（精确 oracle）。
8. 实现 `gather_kv`（ADR-05 批量 gather），逐请求与 single 对比。
9. 实现 `num_free_blocks` / `can_allocate` / `seq_len_of`，在 `cache/__init__.py` 导出新类，补全部注释。
10. 按 slot mapping → batch write → gather oracle → 全量回归的顺序验证。

## 复杂度

| 操作 | 时间复杂度 | 说明 |
|---|---:|---|
| `allocate_request` | O(num_needed) | 逐块 allocate |
| `append_token` | O(1) 均摊 | 多数步不分配 |
| `free_request` | O(num_blocks_held) | 逐块归还 |
| `write_prefill` | O(total_valid_tokens × n_kv × D) | Python 生成 slots + 一次 tensor scatter；不写 batch padding |
| `write_decode` | O(B × n_kv × D) | B 个 slot + 一次 tensor scatter |
| `gather_kv` | O(B × L_pad × n_kv × D) | 一次高级索引复制；与 M3 同量级 |
| `gather_kv_single` | O(seq_len) Python 循环 | 仅测试用 |

内存对比（Qwen3-0.6B：28 层、`n_kv=8`、`head_dim=128`、fp32）：

| 方案 | 单请求占用 | 说明 |
|---|---|---|
| M3 fixed-slot | `28 × 2 × 8 × 4096 × 128 × 4B ≈ 940 MB` | 按 `max_seq_len=4096` 预留 |
| M4 paged（30 token） | `28 × 2 × 8 × 32 × 128 × 4B ≈ 7.3 MB` | 按 2 块（32 位置）实占 |

这个约 128 倍的差距就是 M4 的收益来源，T6 benchmark 要实测验证。

## 与 vLLM 系实现的关系

| 实现 | 对应能力 | inferlite M4-T3 |
|---|---|---|
| vLLM `kv_cache` tensor `[2, num_blocks, block_size, n_kv, D]` | 物理 KV 存储 | 每层 `PagedLayerKVCache.k/v` 分开存 |
| vLLM `KVCacheManager.allocate_slots()` | 请求分配块 | `allocate_request()` / `append_token()` |
| vLLM `slot_mapping` | 展平的写入位置，交给 kernel | `_make_*_slot_mapping()` + PyTorch scatter |
| vLLM `reshape_and_cache` | batch KV scatter 写 cache | `flat_cache[slot_mapping] = flat_k/v`（教学版） |
| vLLM paged attention kernel | kernel 内按 block table 寻址，无 padding | `gather_kv()` 先物化再算，有 padding（M9 优化） |
| nano-vLLM `BlockManager` + `store_kvcache` | 块管理 + KV 写入 | T1 + T3 分开 |
| vLLM `free_block_queue` 回收 | 释放 | `free_request()` |

最大差异在读取：生产实现的 kernel 直接按 block table 寻址，**从不物化连续 KV**，因此没有 padding 浪费。inferlite T3 用"先 gather 成连续张量再走标准 attention"的伪版，牺牲性能换可读性，也让 T4 能复用 M3 的 attention 数学。这个取舍在 M9 写 kernel 时才会被消除。

## L0 测试清单

| # | 测什么 | 预期 |
|---|---|---|
| 1 | `from_config` | 层数、K/V shape、pool 容量正确 |
| 2 | batch prefill 跨多块 | `[20, 5]` 两请求单次 `write_prefill` 写入；各自逐位置读回正确，短请求 padding 未写入 |
| 3 | batch decode 跨块 | 两请求先分别 append、再单次 `write_decode`；写满请求补块，另一请求不补，KV 都正确 |
| 4 | `gather_kv_single` 正确性 | 与逐位置手工拼接一致 |
| 5 | 批量与单请求一致 | `gather_kv` 前 `seq_len` 段等于 `gather_kv_single` |
| 6 | `free_request` | 归还全部块，`num_free_blocks` 复原 |
| 7 | batch 写入隔离 | 请求 a/b 值分别为 1/2，单次 scatter 后 gather 各自纯净 |
| 8 | 分配失败无泄漏 | 块不足时抛 `RuntimeError` 且 `num_free_blocks` 不变 |
| 9 | 未注册 request | 各接口抛 `KeyError` |
| 10 | padding 区 NaN | 注入 NaN 后 `valid_lens` 足以让调用方清零 |
| 11 | prefill slot mapping | 长度等于 `sum(seq_lens)`，每个 slot 与手工 BlockTable 映射一致 |
| 12 | batch 输入合同 | B/id 不一致、重复 id、短 `T_max`、decode token 维非 1 均抛 `ValueError` |

### 测试命令

```bash
uv run pytest tests/unit/test_paged_kv_cache.py -q
```

<details>
<summary>展开查看测试场景细节</summary>

### L0-1 from_config

- 用 `ModelConfig.qwen3_06b()` 或小型 config。
- `len(cache.layers) == config.num_hidden_layers`。
- `cache.layers[0].k.shape == (num_blocks, block_size, n_kv_heads, head_dim)`。
- `cache.num_free_blocks == num_blocks`。

### L0-2 batch prefill 跨多块

- `block_size=16`；请求 a/b 的长度为 `[20, 5]`，分别先 `allocate_request`。
- K/V 输入 shape `[2, n_kv, 20, D]`，a 使用位置递增值，b 使用不同的可识别值。
- 一次 `write_prefill(0, ["a", "b"], k, v)` 后，分别按 BlockTable 逐位置读回比对。
- b 的输入 padding `[:, :, 5:20, :]` 不得写入 b 持有的任意物理块。

### L0-3 batch decode 跨块边界

- `block_size=4`；a 的 prefill 长度为 4（已写满），b 的长度为 3（还剩一位）。
- 两请求都 `append_token()`，再一次 `write_decode()`。
- a 应多拿一块且写新块 offset 0；b 不拿块且写旧块 offset 3。
- 两个输入 KV 在各自物理位置正确，且只有一次 batch scatter。


### L0-4 gather_kv_single 正确性

- prefill 20 个 token 后 `gather_kv_single` 返回 shape `[n_kv, 20, D]`。
- 与逐位置 `position_to_block` 手工取值拼出的张量 `torch.equal`。

### L0-5 批量与单请求一致

- 两个请求，`seq_len` 分别 20 和 5。
- `k_batch, v_batch, valid_lens = gather_kv(0, ["a", "b"])`。
- 对每个 i：`k_batch[i, :, :valid_lens[i], :]` 应等于 `gather_kv_single` 的结果。
- 断言 `valid_lens.tolist() == [20, 5]`。
- 断言 `k_batch.shape[2] == max_num_blocks * block_size`（不是 `max(seq_len)`）。

### L0-6 free_request 归还

- 记录分配前 `num_free_blocks`。
- `allocate_request` 后减少，`free_request` 后完全复原。
- 再次 `free_request` 同一 id 应抛 `KeyError`。

### L0-7 batch 写入隔离

- 请求 a、b 的 prefill 长度不同，输入 a 全 1.0、b 全 2.0。
- 单次 `write_prefill` scatter 后分别 `gather_kv_single`。
- 断言 a 的有效 KV 全为 1.0、b 的全为 2.0，无对方数据。
- 这条能抓住 slot mapping 把 batch 行和 request id 顺序错配的 bug。

### L0-8 分配失败无泄漏

- `num_blocks=2`、`block_size=4`，请求 `prompt_len=12`（需 3 块）。
- `allocate_request` 应抛 `RuntimeError`。
- **断言 `num_free_blocks` 仍为 2** —— 若实现未回滚，这里会变成 0 或 1。
- 断言该 `request_id` 未被注册（后续 `seq_len_of` 抛 `KeyError`）。

### L0-9 未注册 request

- 对空 cache 调用 `append_token` / `free_request` / `write_decode` / `gather_kv_single` / `seq_len_of` 均抛 `KeyError`。
- `gather_kv(0, [])` 抛 `ValueError`。

### L0-10 padding 区 NaN

- 两个请求 `seq_len` 差异较大（如 20 与 3），确保 gather padding 区存在。
- 用 `cache.layers[0].k.fill_(float("nan"))` 后通过 batch prefill 写入有效数据。
- `gather_kv` 返回的 `k[i, :, :valid_lens[i], :]` 必须全部 finite。
- padding 区允许含 NaN —— 这正是要求返回 `valid_lens` 的原因，T4 负责清零。
- 这条锁定 ADR-05 的数值安全契约，对齐 lessons L5。

### L0-11 prefill slot mapping

- 请求长度 `[5, 2]`，`block_size=4`，从 `_make_prefill_slot_mapping(["a", "b"])` 得到 slots。
- `slots.numel() == 7`。
- 逐个用 `position_to_block(pos)` 手算 `block_id * block_size + offset`，结果与 slots 完全一致。
- 这条把「block table 逻辑映射」和「扁平 scatter 地址」的连接锁死。

### L0-12 batch 输入合同

- `len(request_ids) != k.shape[0]` / `v.shape[0]` raise ValueError。
- request_ids 内重复 id raise ValueError。
- prefill `T_max < max(seq_len)` raise ValueError。
- decode token 维不是 1 raise ValueError。
- K/V heads、head_dim、dtype、device 与 cache 不一致 raise ValueError。

</details>

## DoD

- [x] `PagedLayerKVCache` / `PagedKVCache` 实现与本卡接口合同一致。
- [x] `uv run pytest tests/unit/test_paged_kv_cache.py -q` 全绿（9 passed）。
- [x] `uv run pytest tests/ -q` 全量回归通过（236 passed）。
- [x] `gather_kv` 无逐 token Python 循环、无 `torch.cat`（ADR-05）。
- [x] `gather_kv` 返回 `valid_lens`，且注释说明 padding 区可能含 NaN。
- [x] 批量 `gather_kv` 与 `gather_kv_single` 结果一致。
- [x] `write_prefill` / `write_decode` 均为 batch API，不存在生产路径的逐请求 T3 write 调用。
- [x] 两种 batch write 各只有一次 tensor scatter；不使用 `torch.cat`。
- [x] slot mapping 与 BlockTable 的逐位置手工映射一致；batch padding 不写入物理 cache。
- [x] batch 的 request_ids 去重、长度和 K/V shape / dtype / device 合同完整。
- [x] `allocate_request` 块不足时回滚已分配 block，不改变 pool 或 block_tables。
- [x] 不修改 M3 `BatchedKVCache`，不改 attention。
- [x] 不含 prefix hash / LRU / CoW（留 M5）。
- [x] 实现与测试文件均补齐详细注释（README 提交门禁）。
- [x] 末尾追加 `## 完成总结` 与 commit 号。

## 坑（按概率排序）

1. **batch 写入退回逐请求调用**：会让每层每步出现 B 次 Python 调度，把 gather 消掉的开销搬到 write 路径。必须生成扁平 slot mapping 后一次 scatter。
2. **slot mapping 顺序与 batch 行错配**：slot 的顺序必须和 `flat_k/v` 中 token 的顺序完全一致；错配不报错，但会把 a 的 KV 写进 b 的块。L0-7/L0-11 锁定。
3. **batch padding 被写入物理 cache**：短请求的 `k[i, :, seq_len_i:, :]` 可能是 NaN 或无意义值，写入后会污染后续读取。prefill 只 flatten 每行的有效范围。
4. **decode 忘记先 append_token**：slot 会指向上一个 token，静默覆盖旧 KV。契约是先对 batch 中每个 request append，再 batch write。
5. **重复 request id**：会让一次 scatter 出现同 slot 多次写入，顺序无业务语义，必须在入口拒绝。
6. **假设 `L_pad == max(seq_len)`**：`L_pad` 是块对齐长度，通常更大。调用方按 `max(seq_len)` 切片会漏数据或越界。
7. **padding 区 NaN 传播**：`gather_kv` 的 `L_pad` 区域可能是 `torch.empty` 的未初始化内存。只做 score mask 不够，`0 × NaN` 仍会污染 value 聚合（lessons L5）。必须靠 `valid_lens` 清零。
8. **预检查的并发假设失效**：M4 单线程下预检查可取代回滚；M5/M6 一旦有抢占或并发 allocator，必须升级事务回滚。
9. **`reshape` 与 `transpose` 顺序搞错**：`[B, nb, bs, h, d]` 必须先 `reshape` 合并 `nb×bs` 再 `transpose` 换 head 维。顺序颠倒不报错但语义全错 —— 需要 `gather_kv_single` 做 oracle。
10. **提前实现 M5**：不加 hash、LRU、CoW；`free_request` 在 M4 可无条件释放。

## 完成总结

M4-T3 已完成真实 K/V tensor 的分页容器。更完整的实现原理、slot mapping / scatter / gather 数据流和 vLLM 对照见 [`docs/knowledge/m4-paged-attention.md`](../../knowledge/m4-paged-attention.md)。T1 `BlockPool` 管全局 block 的所有权与回收，T2 `BlockTable` 管每请求逻辑位置到物理 block 的顺序映射；本任务首次将 `block_id` 用作物理 tensor 的第 0 维下标，完成从编号到真实 K/V 存储的连接。

### 主要实现

- `PagedLayerKVCache`：每层持有 `k/v: [num_blocks, block_size, n_kv_heads, head_dim]`。
- `PagedKVCache.from_config()`：以 `torch.empty` 创建全层物理 cache，保持未初始化 padding 风险可见，而非用 `zeros` 掩盖。
- 生命周期：`allocate_request` 在中途 allocation 失败时回滚已获得的 block；`append_token` 在 decode 边界按需加块；`free_request` 归还请求全部独占 block。
- 批量 prefill / decode 写入：从 BlockTable 生成 `slot = block_id * block_size + offset`，将 `[B, n_kv, T_max, D]` 的有效 token flatten 后，使用一次高级索引 scatter 写入 K 和 V。
- 批量读取：pad block table 后一次高级索引 gather，返回 `[B, n_kv, L_pad, D]` 与 `valid_lens`；单请求 `gather_kv_single` 作为无 padding oracle。

### 关键取舍

1. **slot mapping 只按请求循环，不逐 token 搬运 K/V**：每个请求通过广播生成整块 slot，batch K/V 用 boolean mask 一次过滤 padding，最终 K/V 各只做一次 scatter。此数据流与 vLLM V1 的 `slot_mapping + reshape_and_cache` 同构，M9 再替换为 kernel。
2. **保留分配回滚**：最终实现选择 `try/except` 归还半分配 block，而不是仅依赖 M4 单线程下的预检查；资源泄漏的风险高于多几行事务代码的复杂度。
3. **padding 责任显式转交 T4**：物理 cache 由 `torch.empty` 创建，gather 的 block 对齐 padding 可能含 NaN/Inf。`valid_lens` 是 T4 清零 K/V 并做 score mask 的唯一依据。
4. **M4 block 独占**：batch scatter 要求同一 batch 没有重复 slot；BlockPool 的独占分配和 request_ids 去重共同保证。M5 prefix cache 共享 block 后，写入前必须引入 CoW。

### 验证结果（2026-07-30）

```text
ruff check inferlite/cache/paged_kv_cache.py tests/unit/test_paged_kv_cache.py
All checks passed

pytest tests/unit/test_paged_kv_cache.py -q
9 passed

pytest tests/ -q
236 passed
```

新增测试覆盖：配置与池容量、分配/释放、分配失败无泄漏、slot mapping 手推、变长 batch prefill scatter、batch gather 与 single oracle 一致、跨块 batch decode、输入合同、未注册 request。

核心实现由作者手写；Agent 负责代码 review、补充合同校验与注释、单元测试、验证及任务卡收口。

最终代码提交：`6857c09 feat(kv-cache): add paged KV cache container (M4-T3 done)`。
