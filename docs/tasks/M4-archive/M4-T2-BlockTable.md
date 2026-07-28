# M4-T2 — BlockTable

> 实现单个请求的逻辑 token 位置到物理 block 的映射：请求看到连续地址，底层落在非连续物理 block 上。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T2 |
| 里程碑 | M4 — PagedAttention |
| 状态 | ⬜ pending |
| 前置 | M4-T1 — BlockPool ✅ |
| 后续 | M4-T3 — PagedKVCache |
| 估时 | 1h |
| 核心文件 | `inferlite/cache/paged_kv_cache.py`（与 T3 同文件） |

## 目标

### 要解决什么问题

T1 的 `BlockPool` 只回答"哪些物理 block 空闲"，它不知道任何请求的存在：

```text
BlockPool: physical_block_id -> ref_count
```

但 attention 需要按 token 位置读写 KV。请求视角是连续的逻辑地址 `pos = 0, 1, 2, ...`，而物理 block 由 pool 按空闲顺序分配，很可能是乱序、不连续的。T2 负责补上中间这一层翻译：

```text
BlockTable: pos -> (physical_block_id, block_offset)
```

这正是 PagedAttention 的核心间接层 —— 有了它，KV 才不需要为每个请求预留一整段连续 `max_seq_len` 空间（M3 `SlotManager` 的做法），碎片化的物理 block 也能拼出逻辑连续的序列。

### 做完是什么效果

```python
# block_size=16，pool 分配到的物理块恰好乱序
table = BlockTable(request_id="req-0", block_size=16)
table.append_block(7)
table.append_block(2)
assert table.capacity == 32

table.extend(20)              # 写入 20 个 token
assert table.seq_len == 20
assert table.num_full_blocks == 1
assert table.last_block_offset == 4

assert table.position_to_block(0) == (7, 0)     # 第 1 个逻辑 block -> 物理块 7
assert table.position_to_block(15) == (7, 15)
assert table.position_to_block(16) == (2, 0)    # 跨 block 边界 -> 物理块 2
assert table.position_to_block(19) == (2, 3)
```

### 不做什么

- 不持有或读写 K/V tensor；tensor 属于 T3 `PagedKVCache`。
- 不调用 `BlockPool`，也不持有它的引用。BlockTable 只记录"我被分到了哪些块"，块从哪来、何时归还由 T3 协调。
- 不做 prefix hash、`token_ids` 记录、LRU 或 CoW；这些属于 M5。
- 不耦合 `RequestState` / Sequence；`request_id` 只作为标识字符串。
- 不感知 `num_blocks` 总量，因此无法校验物理块 id 的上界（见接口合同）。

### 在推理链路中的位置

```text
M3 RequestState / Scheduler
          |
          v
M4-T3 PagedKVCache          <- 持有 tensor，协调下面两者
     |             |
     v             v
M4-T2 BlockTable   M4-T1 BlockPool ✅
逻辑位置映射       物理 block 分配
```

BlockTable 与 BlockPool 是**平级**的，互不依赖，都由 T3 组合使用。

### block_id 是编号，不是地址

一个容易误解的点：`block_id` 不是内存地址，也不是任何形式的指针。

T1 `BlockPool` 里没有任何 tensor、指针或地址，全部数据只有 `Block(block_id=i)` 和 `deque(range(num_blocks))` —— 它就是一个"0 到 `num_blocks-1` 的编号池，附带每个编号的引用计数"。

这里的 physical / logical 是**两层索引之间的相对关系**，不是"虚拟内存 vs 物理内存"那种硬件含义。`block_id` 之所以叫 physical，只因为它全局唯一、跨请求共享 —— 相对于每个请求各自从 0 开始编号的 logical block index 而言。

完整的三层索引模型：

```text
逻辑层：请求视角的 token 位置     pos = 0 .. seq_len-1
              |  BlockTable 翻译（T2）
              v
物理层：全局共享的 block 编号     block_id = 0 .. num_blocks-1
              |  PagedKVCache 索引（T3）
              v
真实内存：tensor 第 0 维下标      k[block_id, offset, head, dim]
```

编号变成真实地址发生在 T3：`block_id` 用作 `k: [num_blocks, block_size, n_kv_heads, head_dim]` 的第 0 维下标，真正的地址由 PyTorch 按 stride 算出。BlockPool 和 BlockTable 都只管到第一步之前。

这样分层有三个收益：

1. **同一个 pool 服务所有层**。Qwen3-0.6B 有 28 层，若 pool 管地址就要管 28×2 个 tensor。而 `block_id=5` 对 28 层都成立 —— 每层的 `k[5]` 是不同内存，但共用同一个编号，分配逻辑只需写一份。
2. **与 device / dtype 解耦**。CPU / MPS / CUDA、fp16 / bf16，映射代码完全不变，地址与对齐全归 PyTorch。
3. **让 M5 的引用计数有意义**。prefix caching 让两请求共享同一 `block_id`（`ref_count=2`），共享的是廉价整数，底层内存自然被共享，无需任何拷贝或地址运算。

vLLM 是同构设计：其 `BlockPool` 同样只管 `block_id` 和 `ref_cnt`，真实 `kv_cache` tensor 由 worker 持有，`block_id` 最终展平成 `slot_mapping` 交给 kernel 做地址计算。

对 T2 的直接影响：既然只处理编号，`BlockTable` 就不需要 import torch。如果实现时发现自己想 import，说明有逻辑串到 T3 去了。

### 设计原则

- **纯 Python，零依赖**：不 import torch，不 import BlockPool。这让它可以被单独测试，也让"位置映射算错"和"tensor 写错"两类 bug 互不掩盖。
- **写入口收在类内部**：`seq_len` 和 `block_ids` 的增长必须通过 `append_block()` / `extend()`，由类自己守住 `seq_len <= capacity`。
- **区分 logical 与 physical**：命名上严格区分逻辑 block 下标（`block_ids` 的索引）和物理 block id（`block_ids` 的值）。

## 产出文件

- `inferlite/cache/paged_kv_cache.py::BlockTable`
- `tests/unit/test_block_table.py`

## 接口骨架

```python
from dataclasses import dataclass, field


@dataclass
class BlockTable:
    """单个请求的逻辑 block 到物理 block 的映射。"""

    request_id: str
    block_size: int
    block_ids: list[int] = field(default_factory=list)
    seq_len: int = 0

    def __post_init__(self) -> None: ...

    # ── 只读视图 ──
    @property
    def num_blocks(self) -> int: ...
    @property
    def capacity(self) -> int: ...
    @property
    def num_full_blocks(self) -> int: ...
    @property
    def last_block_offset(self) -> int: ...

    # ── 查询 ──
    def needs_new_block(self) -> bool: ...
    def position_to_block(self, pos: int) -> tuple[int, int]: ...

    # ── 受控写入口 ──
    def append_block(self, physical_block_id: int) -> None: ...
    def extend(self, num_tokens: int) -> None: ...
```

相对早期草稿的三处实质变更：

| 变更 | 理由 |
|---|---|
| `block_size` 去掉默认值 `16` | T1 `BlockPool` 已持有 `block_size`。两处各带默认值，早晚会出现一边 16、一边 32 的不一致，且这种不一致表现为静默的位置错乱。强制由构造方传入 |
| 新增 `capacity` | 把 `len(block_ids) * block_size` 命名化，作为 `seq_len` 上界的单一事实来源，避免这个表达式散落在 `needs_new_block` / `extend` / T3 各处 |
| 新增 `append_block()` / `extend()` | 草稿里没有任何写方法，意味着 T3 的 `may_append_block()` 只能裸改 `block_ids.append(...)` 和 `seq_len += 1`，`seq_len <= capacity` 无人保证。一旦越界，`position_to_block` 会返回**别的请求**的物理块，KV 被静默覆盖 —— 这类 bug 比抛异常难查一个数量级 |

`block_ids` 必须用 `field(default_factory=list)`。写成 `block_ids: list[int] = []` 会让所有 BlockTable 实例共享同一个列表对象，多请求隔离直接失效。

M4 的 `BlockTable` 不预埋 `block_hashes` / `token_ids` 字段。M5 做 prefix caching 时再新增。

## 与 BlockPool 的关系

这一节回答两个容易混淆的问题：两个类各自负责什么，以及为什么它们在代码层面没有依赖。

### 职责切分

| | BlockPool（T1） | BlockTable（T2） |
|---|---|---|
| 数量 | 全局 **1 个** | 每个请求 **1 个** |
| 回答的问题 | 哪些块空闲？这块被几个人用着？ | 我这个请求的第 pos 个 token 在哪块的哪个位置？ |
| 核心数据 | `free_block_ids`、`ref_count` | `block_ids`（有序）、`seq_len` |
| 视角 | 全局资源账本（集合语义） | 单请求顺序记录（**序列语义**） |

一句话概括：**BlockPool 决定块归谁，BlockTable 决定块的顺序。**

`block_ids` 是 list 而不是 set，这是本质区别 —— 顺序本身就是信息。`block_ids=[7, 2]` 和 `[2, 7]` 对 BlockPool 完全等价（都是"7 和 2 被占用"），但对 BlockTable 是两个不同的序列：token 0 分别落在物理块 7 和物理块 2。

### 代码层面没有依赖

T2 不 import `BlockPool`、不持有它的引用、不调用它的任何方法。两者平级，由 T3 组合：

```text
        M4-T3 PagedKVCache        <- 唯一同时认识两者的人
        +--------+--------+
        v                 v
  BlockTable          BlockPool
（每请求一个）        （全局一个）
        +---  互不认识  ---+
```

协作不通过直接调用，而是通过 `block_id` 这个整数，T3 做中间人。以下是 T3 的伪代码示意（**下周 T3 的内容，T2 不实现**）：

```python
# prefill：向 pool 要块，把编号交给 table
def allocate_request(self, request_id, prompt_len):
    num_needed = ceil(prompt_len / self.block_size)
    if not self.block_pool.can_allocate(num_needed):    # 问 pool 够不够
        raise ...
    table = BlockTable(request_id, self.block_size)
    for _ in range(num_needed):
        table.append_block(self.block_pool.allocate())  # pool 发号，table 记顺序
    table.extend(prompt_len)

# decode：需要时补一块
def may_append_block(self, request_id):
    table = self.block_tables[request_id]
    if table.needs_new_block():                         # table 说不够了
        table.append_block(self.block_pool.allocate())  # 再去 pool 要
    table.extend(1)

# 释放：table 把编号还给 pool
def free_request(self, request_id):
    table = self.block_tables.pop(request_id)
    for block_id in table.block_ids:
        self.block_pool.free(block_id)

# 读写：table 翻译位置，编号当 tensor 下标
def write_decode(self, layer_idx, request_id, k, v):
    table = self.block_tables[request_id]
    block_id, offset = table.position_to_block(table.seq_len - 1)
    self.layers[layer_idx].k[block_id, offset] = k
```

信息始终单向流动：`pool.allocate()` 产出编号 → `table.append_block()` 消费编号；释放时 `table.block_ids` 提供编号 → `pool.free()` 消费编号。两边都只碰整数。

### 为什么刻意不让 BlockTable 依赖 BlockPool

这是设计决定，不是简化：

1. **两个关注点可独立演进**。M5 要给 pool 加 prefix hash 查找和 LRU 淘汰队列，pool 会复杂很多，而 BlockTable 的映射逻辑一行都不用改。
2. **分配失败的处理归上层**。若 table 自己去 pool 要块，"块不够了"就会在 table 内部抛出。但此时该抢占别的请求还是把当前请求踢回等待队列，是**调度决策**，BlockTable 无从判断。所以要块必须由能做决策的 T3/T5 发起。
3. **可独立测试**。T2 的单测直接构造 `block_ids=[7, 2, 5]` 即可，不需要真实分配、不需要 pool、不需要 torch。"映射算错"和"分配算错"两类 bug 互不掩盖。
4. **避免双向引用**。若 table 持有 pool，很容易演化成 table 析构时自动 `free`；而 M5 又需要 pool 知道哪些 table 引用了某块 —— 双向依赖一旦成型就很难拆。

### 走查示例：物理块复用

`block_size=4`、pool 共 3 块、两个请求 A / B：

```text
初始       pool: free=[0,1,2]  ref=[0,0,0]

A 到达，prompt_len=6，需 2 块
           pool.allocate() -> 0      free=[1,2]  ref=[1,0,0]
           pool.allocate() -> 1      free=[2]    ref=[1,1,0]
           A.block_ids=[0,1]  A.seq_len=6

B 到达，prompt_len=3，需 1 块
           pool.allocate() -> 2      free=[]     ref=[1,1,1]
           B.block_ids=[2]    B.seq_len=3

此时映射互不干扰：
  A.position_to_block(5) == (1, 1)   A 的第 6 个 token 在物理块 1 的 offset 1
  B.position_to_block(2) == (2, 2)   B 的第 3 个 token 在物理块 2 的 offset 2

A 结束释放
           pool.free(0); pool.free(1)   free=[0,1]  ref=[0,0,1]

B decode，seq_len 3 -> 4，恰好写满
           B.needs_new_block() -> True  （seq_len 4 >= capacity 4）
           pool.allocate() -> 0         <- 复用了 A 用过的块
           B.block_ids=[2,0]
  B.position_to_block(4) == (0, 0)
```

关键观察：B 最终的 `block_ids=[2, 0]` 逻辑上连续，物理编号却是倒序的。这正是 PagedAttention 消除内存碎片的机制，而实现它只需要"pool 管账本 + table 管顺序"这两个各自都很简单的部件。

对比 M3 `SlotManager`：它给每个请求预留一整段 `max_seq_len`，A 释放的空间只能整段复用。B 写到第 5 个 token 时若自己那段已满，没有任何办法借用 A 腾出的空间。

## 接口合同

### `__post_init__() -> None`

dataclass 的校验入口，负责拒绝构造出的非法初始状态：

- `block_size <= 0` 时 `raise ValueError`。
- `seq_len < 0` 时 `raise ValueError`。
- `seq_len > capacity` 时 `raise ValueError`（声称已有的 token 数超过已分配容量）。
- `block_ids` 中存在负数时 `raise ValueError`（理由见 `append_block`）。

### `num_blocks -> int`（property）

- 返回 `len(block_ids)`，即当前持有的物理 block 数。
- 注意这是"已分配的块数"，不是"已写满的块数"，后者是 `num_full_blocks`。

### `capacity -> int`（property）

- 返回 `len(block_ids) * block_size`，即当前已分配空间能容纳的 token 上限。
- `seq_len` 的合法上界，也是 `needs_new_block()` 的判据。

### `num_full_blocks -> int`（property）

- 返回 `seq_len // block_size`，已被写满的 block 数。
- `seq_len == 0` 时为 0；`seq_len == block_size` 时为 1。

### `last_block_offset -> int`（property）

- 返回 `seq_len % block_size`，最后一个 block 中已用的 token 数。
- 恰好写满时返回 0，而不是 `block_size`。这一点容易在 T3 写入时误用，需要配合 `needs_new_block()` 判断。

### `needs_new_block() -> bool`

- 语义：**再追加 1 个 token 之前，是否需要先分配新 block**。
- 实现：`return self.seq_len >= self.capacity`。
- `block_ids` 为空（`capacity == 0`）时返回 `True`，覆盖"请求刚创建、还没拿到任何块"的初始状态。

早期草稿写的是 `seq_len % block_size == 0 and seq_len > 0`，改掉的原因：那个式子只在"seq_len 恰好等于 capacity"时才与真实需求一致。当 prefill 一次性分配了 2 个块但只写了 20 个 token 时（`seq_len=20`、`capacity=32`），`20 % 16 != 0` 碰巧返回 False，看起来对；但若 `seq_len=32`、`capacity=48`（预分配了富余块），旧式子返回 True，会让 T3 多分配一个完全不需要的块。直接比较 `seq_len` 与 `capacity` 对所有中间状态都成立。

### `position_to_block(pos: int) -> tuple[int, int]`

- 逻辑位置 → `(physical_block_id, block_offset)`。
- 定义域为 `[0, seq_len)`：
  - `pos < 0` 时 `raise ValueError`。**必须显式拒绝**，否则 `block_ids[-1]` 会静默返回最后一个物理块，和 T1 的 `block_id=-1` 是同一类坑。
  - `pos >= seq_len` 时 `raise ValueError`。读未写入的位置在 T3 里会取到 `torch.empty` 的未初始化内存，可能含 NaN（参见 `lessons.md` L5）。
- 计算：

  ```python
  logical_block = pos // self.block_size
  block_offset = pos % self.block_size
  assert logical_block < len(self.block_ids)
  return self.block_ids[logical_block], block_offset
  ```

- `logical_block >= len(block_ids)` 用 `assert`：因为 `pos < seq_len <= capacity` 已经保证它不成立，触发就说明不变量被破坏了，属于代码 bug 而非调用方错误。这与 T1 的分层一致 —— 调用方可能触发的用显式异常，仅 bug 才触发的用 assert。

### `append_block(physical_block_id: int) -> None`

- 把一个物理 block 追加到 `block_ids` 末尾，`capacity` 相应增加 `block_size`。
- **不改动 `seq_len`**。分配空间和写入 token 是两件事，分开才能表达"已分配但未写满"。
- `physical_block_id < 0` 时 `raise ValueError`。
- 不校验上界：BlockTable 不知道 `num_blocks`，`physical_block_id < num_blocks` 由 `BlockPool` 在分配时保证。这个职责边界要在注释里写清，否则后面容易误加一个拿不到正确总量的假校验。

### `extend(num_tokens: int) -> None`

- 声明已写入 `num_tokens` 个 token，`seq_len += num_tokens`。
- `num_tokens < 0` 时 `raise ValueError`。
- `num_tokens == 0` 是合法 no-op（prefill 空 prompt 等边界）。
- `seq_len + num_tokens > capacity` 时 `raise RuntimeError`：容量不足说明调用方没有先 `append_block()`，是可预期的调用顺序错误，与 T1 `allocate()` 耗尽时抛 `RuntimeError` 的风格一致。
- prefill 用 `extend(prompt_len)`，decode 用 `extend(1)`。

## 算法核心与不变量

### 位置映射

```text
logical_block  = pos // block_size          # 第几个逻辑块
block_offset   = pos % block_size           # 块内第几个 token
physical_block = block_ids[logical_block]   # 查表得到物理块
结果           = (physical_block, block_offset)
```

`block_size` 通常取 2 的幂（16 / 32），除法和取模都很便宜。这里不做位运算优化 —— 可读性优先，且这不是热点。

### 状态不变量

```text
0 <= seq_len <= capacity == len(block_ids) * block_size
position_to_block 的合法定义域为 [0, seq_len)
num_full_blocks * block_size + last_block_offset == seq_len
所有 block_ids 元素 >= 0
```

- `append_block()` 只增 `capacity`，不动 `seq_len`。
- `extend()` 只增 `seq_len`，且不得越过 `capacity`。
- 两个写入口共同维持 `seq_len <= capacity`，这是整个类存在的意义。

## 实现步骤

1. 在 `inferlite/cache/paged_kv_cache.py` 新建文件，写文件级 docstring 说明本文件承载 T2 BlockTable 与 T3 PagedKVCache。
2. 定义 `BlockTable` dataclass 及四个字段，注意 `field(default_factory=list)`。
3. 实现 `__post_init__` 的构造校验。
4. 实现 `num_blocks` / `capacity` / `num_full_blocks` / `last_block_offset` 四个 property。
5. 实现 `needs_new_block()`，用 `seq_len >= capacity`。
6. 实现 `position_to_block()`，先校验定义域再做除法取模。
7. 实现 `append_block()` / `extend()`，守住容量不变量。
8. 在 `inferlite/cache/__init__.py` 导出 `BlockTable`。
9. 运行 T2 单测，再跑全量回归。

## 复杂度

| 操作 | 时间复杂度 | 原因 |
|---|---:|---|
| `position_to_block()` | O(1) | 一次整除 + 一次取模 + 一次 list 索引 |
| `append_block()` | O(1) 均摊 | `list.append` |
| `extend()` | O(1) | 整数加法 |
| 四个 property | O(1) | `len()` 与算术 |

BlockTable 的所有操作都是 O(1)，这是 PagedAttention 能做到低开销间接寻址的前提。

## 与 vLLM 系实现的关系

| 实现 | 对应能力 | inferlite M4-T2 |
|---|---|---|
| vLLM V1 `req_to_blocks` / `KVCacheBlocks` | 请求到物理块列表的映射 | `BlockTable.block_ids` |
| vLLM `BlockTable.append_row()` | 追加物理块到请求的表 | `append_block()`（单块教学版） |
| vLLM slot mapping 计算 | 逻辑位置 → 物理槽位，喂给 kernel | `position_to_block()`（返回二元组而非扁平 slot） |
| vLLM `num_computed_tokens` | 已计算 token 数 | `seq_len` |
| nano-vLLM `Sequence.block_table` | 请求持有的块列表 | 同能力，但独立成类而非挂在 Sequence 上 |
| vLLM `block_hashes` / prefix 复用 | 前缀缓存 | M5，T2 不预埋字段 |

一个差异值得注意：生产实现通常把 block table 组织成 `[max_num_reqs, max_num_blocks]` 的 GPU tensor，一次性喂给 kernel；inferlite 用 per-request 的 Python 对象，牺牲性能换可读性。T4 做 PyTorch gather 伪版时会体现这个取舍。

## L0 测试清单

| # | 测什么 | 预期 |
|---|---|---|
| 1 | 构造与初始状态 | `seq_len=0`、`capacity=0`、`num_blocks=0`，`needs_new_block()` 为 True |
| 2 | 边界映射 | `block_ids=[5,9]` 时 pos 0/15/16/31 映射到 `(5,0)/(5,15)/(9,0)/(9,15)` |
| 3 | 非连续物理块 | 物理块乱序（如 `[7,2,5]`）时仍按逻辑顺序翻译 |
| 4 | full_blocks / last_offset | `seq_len=0/1/16/17` 取值正确，且满足恒等式 |
| 5 | `needs_new_block` | 写满后为 True，`append_block` 后变 False |
| 6 | `extend` 容量守卫 | 超出 `capacity` 抛 `RuntimeError`；`extend(0)` 为 no-op；负数抛 `ValueError` |
| 7 | `position_to_block` 定义域 | `pos=-1` 与 `pos=seq_len` 抛 `ValueError` |
| 8 | 构造参数 | `block_size<=0`、`seq_len<0`、`seq_len>capacity`、负 `block_ids` 抛 `ValueError` |
| 9 | 多请求隔离 | 两个 BlockTable 各自 `append_block`/`extend` 互不影响 |

### 测试命令

```bash
uv run pytest tests/unit/test_block_table.py -q
```

<details>
<summary>展开查看测试场景细节</summary>

### L0-1 构造与初始状态

- `BlockTable(request_id="req-0", block_size=16)`。
- `block_ids == []`、`seq_len == 0`、`num_blocks == 0`、`capacity == 0`。
- `num_full_blocks == 0`、`last_block_offset == 0`。
- `needs_new_block() is True` —— 覆盖"还没拿到任何块"的初始状态，这是旧 `seq_len % block_size == 0 and seq_len > 0` 式子会答错的场景。

### L0-2 跨 block 边界映射

- `block_size=16`、`block_ids=[5, 9]`、`extend(32)`。
- `position_to_block(0) == (5, 0)`。
- `position_to_block(15) == (5, 15)` —— 第一个块的最后一个位置。
- `position_to_block(16) == (9, 0)` —— 跨越边界，切到第二个物理块。
- `position_to_block(31) == (9, 15)`。

### L0-3 非连续物理块

- `block_ids=[7, 2, 5]`（模拟 pool 乱序分配），`extend(48)`。
- `position_to_block(0) == (7, 0)`、`position_to_block(16) == (2, 0)`、`position_to_block(32) == (5, 0)`。
- 锁定的合同：逻辑顺序由 `block_ids` 的**下标**决定，与物理 id 的大小无关。这是 PagedAttention 允许物理碎片的关键。

### L0-4 num_full_blocks 与 last_block_offset

- `block_size=16`、`block_ids=[0, 1]`。
- `seq_len=0` → `(0, 0)`。
- `seq_len=1` → `(0, 1)`。
- `seq_len=16` → `(1, 0)` —— 恰好写满时 offset 回 0，不是 16。
- `seq_len=17` → `(1, 1)`。
- 每种情况都断言 `num_full_blocks * block_size + last_block_offset == seq_len`。

### L0-5 needs_new_block 状态转换

- `block_size=4`、`append_block(0)`，`capacity=4`。
- `extend(3)` 后 `needs_new_block() is False`（还差 1 个位置）。
- `extend(1)` 后 `seq_len == capacity == 4`，`needs_new_block() is True`。
- `append_block(1)` 后 `capacity=8`，`needs_new_block()` 回到 False。

### L0-6 extend 容量守卫

- `block_size=4`、单个块，`capacity=4`。
- `extend(5)` raise RuntimeError，且**失败后 `seq_len` 不变**（不能留下半改状态）。
- `extend(4)` 成功，`seq_len == 4`。
- `extend(0)` 后 `seq_len` 仍为 4。
- `extend(-1)` raise ValueError。

### L0-7 position_to_block 定义域

- `block_size=16`、`block_ids=[0, 1]`、`extend(20)`。
- `position_to_block(-1)` raise ValueError —— 若不校验，`block_ids[-1]` 会静默返回物理块 1。
- `position_to_block(20)` raise ValueError（等于 `seq_len`，尚未写入）。
- `position_to_block(19)` 正常返回。

### L0-8 构造参数校验

- `block_size=0` / `block_size=-1` raise ValueError。
- `seq_len=-1` raise ValueError。
- `block_ids=[0]`、`block_size=16`、`seq_len=17` raise ValueError（超出 capacity）。
- `block_ids=[-1]` raise ValueError。

### L0-9 多请求隔离

- 两个 BlockTable `req-0` / `req-1`，同一 `block_size`。
- 各自 `append_block` 不同物理块并 `extend`。
- 断言两者 `block_ids`、`seq_len` 与映射结果互不影响 —— 这条同时防住 dataclass 可变默认值共享列表的经典错误。

</details>

## DoD

- [ ] `BlockTable` 实现与任务卡接口合同一致。
- [ ] `uv run pytest tests/unit/test_block_table.py -q` 全绿。
- [ ] `uv run pytest tests/ -q` 全量回归通过。
- [ ] 满足不变量 `0 <= seq_len <= capacity`，且写入口是唯一修改途径。
- [ ] 调用方可能触发的错误用显式 `ValueError` / `RuntimeError`；仅内部不变量用 `assert`。
- [ ] 不 import torch，不依赖 `BlockPool` / attention / model。
- [ ] 不修改 M3 `BatchedKVCache`。
- [ ] 不含 prefix hash / `token_ids` / LRU / CoW 字段（留 M5）。
- [ ] 实现与测试文件均补齐详细注释（README 提交门禁）。
- [ ] 末尾追加 `## 完成总结` 与 commit 号。

## 坑（按概率排序）

1. **负数 `pos` 被当作反向索引**：`block_ids[-1]` 静默返回最后一个物理块，读到完全无关的 KV。所有位置入口先做显式定义域校验。
2. **`seq_len` 越过 `capacity`**：`position_to_block` 会索引到超出 `block_ids` 范围，或在预分配富余时写进相邻请求的物理块。后者是静默数据损坏，不会报错，只会让输出慢慢变得莫名其妙。必须由 `extend()` 守住。
3. **dataclass 可变默认值**：`block_ids: list[int] = []` 让所有实例共享同一列表，多请求彼此串数据。必须用 `field(default_factory=list)`。
4. **混淆 logical index 与 physical id**：`block_ids[2]` 是"第 3 个逻辑块的物理 id"，不是"物理块 2"。命名和注释都要显式区分。
5. **`last_block_offset` 在写满时返回 0**：误当成"最后一块还剩 0 个位置可用"就会漏分配。判断是否需要新块只用 `needs_new_block()`。
6. **给 `append_block` 加假的上界校验**：BlockTable 拿不到 `num_blocks`，任何自造上界都是错的。这不是遗漏，而是"不依赖 BlockPool"这一设计的直接后果 —— 上界由 `BlockPool.allocate()` 在发号时保证，它才是知道总量的那一方。
7. **提前实现 M5**：不加 hash、`token_ids`、LRU 或 CoW。
