"""M4 PagedAttention 的分页 KV Cache。

本文件承载两个协作但职责分离的组件：

  - BlockTable（T2）：单请求的 pos -> (physical_block_id, block_offset) 映射。
    纯 Python，不 import torch，不依赖 BlockPool。
  - PagedKVCache（T3，待实现）：持有各层 K/V tensor，组合 BlockPool 与
    BlockTable，负责请求生命周期和 KV 读写。

分层关系：BlockPool 管「块归谁」（全局一个），BlockTable 管「块的顺序」
（每请求一个）。两者互不认识，由 PagedKVCache 组合。

关于 block_id 的语义：它是一个**编号**，不是内存地址。真正的寻址发生在 T3，
届时 block_id 用作 k: [num_blocks, block_size, n_kv_heads, head_dim] 的第 0 维
下标。因此本文件只做整数运算，不需要 torch。
"""

from dataclasses import dataclass, field


@dataclass
class BlockTable:
    """单个请求的逻辑 block 到物理 block 的映射。

    请求看到连续的逻辑地址 pos = 0..seq_len-1，底层可以落在完全非连续、
    甚至倒序的物理 block 上。这层间接映射是 PagedAttention 能消除内存
    碎片的原因：不必为每个请求预留一整段连续的 max_seq_len 空间。

    block_ids 的**下标**是 logical block index，**值**是 physical block id，
    两者不可混淆。block_ids=[7, 2] 表示逻辑第 0 块落在物理块 7、逻辑第 1 块
    落在物理块 2。所以它必须是有序 list 而非 set —— 顺序本身就是信息。

    状态不变量：
    - 0 <= seq_len <= capacity == len(block_ids) * block_size
    - position_to_block 的合法定义域为 [0, seq_len)
    - num_full_blocks * block_size + last_block_offset == seq_len
    - block_ids 中所有元素非负

    seq_len 和 block_ids 应只通过 extend() / append_block() 修改：这两个写
    入口负责守住 seq_len <= capacity。字段本身未做私有化（保持 dataclass
    构造与测试的简洁），因此这是约定而非强制 —— 直接赋值 seq_len 会破坏
    不变量，进而让 position_to_block 返回其他请求的物理块。

    M4 不含 prefix caching，故不预埋 block_hashes / token_ids 字段（留 M5）。
    """

    request_id: str
    block_size: int
    # 逻辑块顺序 -> 物理块编号。必须用 default_factory，否则所有实例会共享
    # 同一个 list，多请求之间彼此串数据。
    block_ids: list[int] = field(default_factory=list)
    seq_len: int = 0

    def __post_init__(self) -> None:
        """校验构造出的初始状态。

        dataclass 自动生成的 __init__ 无法插入校验逻辑，__post_init__ 是它在
        所有字段赋值完成后调用的钩子，因此这里可以安全读取 self.capacity。

        这些都是调用方可能触发的参数错误，所以用显式 raise 而非 assert
        （python -O 会移除 assert，让校验在生产环境静默失效）。
        """
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.seq_len < 0:
            raise ValueError(f"seq_len must be non-negative, got {self.seq_len}")
        if any(bid < 0 for bid in self.block_ids):
            raise ValueError("block_ids must all be non-negative")
        # capacity 依赖 block_size，故必须放在 block_size 校验之后，否则
        # block_size=-4 会算出负 capacity，让这条检查失去意义。
        # 用 > 而非 >=：seq_len == capacity 表示「正好写满」，是合法状态。
        if self.seq_len > self.capacity:
            raise ValueError(f"seq_len {self.seq_len} exceeds capacity {self.capacity}")

    # ── 只读视图 ──
    # 这四个量都是 block_ids / seq_len 的派生值，故用 property 实时计算而非
    # 存成字段：存字段就会出现「两份事实」，append_block 后忘记同步就不一致。

    @property
    def num_blocks(self) -> int:
        """当前持有的物理 block 数（已分配，不代表已写满）。"""
        return len(self.block_ids)

    @property
    def capacity(self) -> int:
        """已分配空间能容纳的 token 上限，也是 seq_len 的合法上界。"""
        return self.block_size * self.num_blocks

    @property
    def num_full_blocks(self) -> int:
        """已被写满的 block 数。"""
        return self.seq_len // self.block_size

    @property
    def last_block_offset(self) -> int:
        """最后一个 block 中**已用**的 token 数。

        注意恰好写满时返回 0 而不是 block_size。误当成「还剩 0 个位置」
        就会漏分配，判断是否需要新块只用 needs_new_block()。
        """
        return self.seq_len % self.block_size

    # ── 查询 ──

    def needs_new_block(self) -> bool:
        """再追加 1 个 token 之前，是否需要先分配新 block。

        直接比较 seq_len 与 capacity，而不是 seq_len % block_size == 0：
        后者在 capacity == 0（请求刚创建、还没拿到任何块）时漏判返回 False，
        且在 prefill 预分配了富余块时误判返回 True，导致上层白白多要一块。
        """
        return self.seq_len >= self.capacity

    def position_to_block(self, pos: int) -> tuple[int, int]:
        """把逻辑 token 位置翻译成 (physical_block_id, block_offset)。"""
        # 必须显式拒绝越界，尤其是负数：pos=-1 会让 block_ids[-1] 静默返回
        # 最后一个物理块，不报错却读到无关数据。pos >= seq_len 则会读到从未
        # 写入的位置，在 T3 里是 torch.empty 的未初始化内存（可能含 NaN）。
        if not 0 <= pos < self.seq_len:
            raise ValueError(f"pos {pos} out of range [0, {self.seq_len})")
        logical_block = pos // self.block_size
        block_offset = pos % self.block_size
        # 内部不变量：pos < seq_len <= capacity == num_blocks * block_size
        # 已保证此条成立，触发则说明不变量维护逻辑有 bug，而非调用方错误，
        # 故用 assert 而非 raise。
        assert logical_block < self.num_blocks
        return self.block_ids[logical_block], block_offset

    # ── 受控写入口 ──

    def append_block(self, physical_block_id: int) -> None:
        """追加一个物理 block，capacity 增加 block_size。

        刻意**不改动 seq_len**：分配空间和写入 token 是两件事，分开才能
        表达「已分配但未写满」的中间状态。

        只校验非负，不校验上界 physical_block_id < num_blocks —— BlockTable
        刻意不依赖 BlockPool，因此拿不到 num_blocks。上界由 BlockPool 在
        allocate() 发号时保证，它才是知道总量的那一方。这不是遗漏。
        """
        if physical_block_id < 0:
            raise ValueError(f"physical_block_id must be non-negative, got {physical_block_id}")
        self.block_ids.append(physical_block_id)

    def extend(self, num_tokens: int) -> None:
        """声明已写入 num_tokens 个 token。prefill 传 prompt_len，decode 传 1。"""
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        # 容量不足说明调用方没有先 append_block()，属可预期的调用顺序错误，
        # 与 T1 BlockPool.allocate() 无空闲块时抛 RuntimeError 保持一致。
        # 先检查后修改，保证失败时 seq_len 不留半改状态。
        if self.seq_len + num_tokens > self.capacity:
            raise RuntimeError(
                f"cannot extend {num_tokens} tokens: "
                f"seq_len {self.seq_len} + {num_tokens} > capacity {self.capacity}"
            )
        self.seq_len += num_tokens
