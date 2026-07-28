"""M4-T2 BlockTable 单元测试。

本文件验证的是逻辑位置到物理 block 的**整数映射**，不涉及任何 K/V tensor：

    pos -> (physical_block_id, block_offset)

核心不变量：

    0 <= seq_len <= capacity == len(block_ids) * block_size

BlockTable 与 BlockPool 平级、互不依赖，所以这里可以直接构造任意
block_ids 而不需要真实分配 —— 这样「映射算错」和「分配算错」两类 bug
不会互相掩盖。M5 的 prefix hash / LRU / CoW 不在本文件范围内。
"""

import pytest

from inferlite.cache.paged_kv_cache import BlockTable


def test_initial_state_requires_new_block() -> None:
    """刚创建的 table 没有任何块，必须报告需要分配。"""
    table = BlockTable(request_id="req-0", block_size=16)

    assert table.block_ids == []
    assert table.seq_len == 0
    assert table.num_blocks == 0
    assert table.capacity == 0
    assert table.num_full_blocks == 0
    assert table.last_block_offset == 0

    # 关键用例：capacity == 0 时必须返回 True。
    # 旧公式 seq_len % block_size == 0 and seq_len > 0 在此处会漏判成 False，
    # 上层就不会分配首个块，随后 extend() 直接因容量不足失败。
    assert table.needs_new_block() is True


def test_position_mapping_across_block_boundary() -> None:
    """逻辑位置跨 block 边界时，必须切换到下一个物理块。"""
    table = BlockTable(request_id="req-0", block_size=16, block_ids=[5, 9])
    table.extend(32)

    assert table.position_to_block(0) == (5, 0)
    # 第一个块的最后一个位置，offset 取到 block_size - 1。
    assert table.position_to_block(15) == (5, 15)
    # 边界跨越点：pos=16 属于逻辑第 1 块，映射到物理块 9 的 offset 0。
    assert table.position_to_block(16) == (9, 0)
    assert table.position_to_block(31) == (9, 15)


def test_mapping_follows_logical_order_not_physical_id() -> None:
    """物理块乱序时，翻译顺序仍由 block_ids 的下标决定。"""
    # 模拟 BlockPool 在碎片化状态下分配出的乱序物理块。
    table = BlockTable(request_id="req-0", block_size=16, block_ids=[7, 2, 5])
    table.extend(48)

    # 这条断言锁定 PagedAttention 最本质的合同：逻辑顺序来自 block_ids 的
    # **下标**，与物理 id 的数值大小完全无关。正因如此，物理空间才可以碎片化。
    assert table.position_to_block(0) == (7, 0)
    assert table.position_to_block(16) == (2, 0)
    assert table.position_to_block(32) == (5, 0)

    # block_ids 是 list 而非 set：[7,2,5] 与 [2,5,7] 对 BlockPool 等价，
    # 但对 BlockTable 是两个完全不同的序列。
    assert table.block_ids == [7, 2, 5]


def test_full_blocks_and_last_offset_identity() -> None:
    """num_full_blocks 与 last_block_offset 必须恒等分解 seq_len。"""
    table = BlockTable(request_id="req-0", block_size=16, block_ids=[0, 1])

    # (seq_len, num_full_blocks, last_block_offset)
    expected = [(0, 0, 0), (1, 0, 1), (16, 1, 0), (17, 1, 1)]
    for seq_len, num_full, last_offset in expected:
        table.seq_len = seq_len
        assert table.num_full_blocks == num_full
        # seq_len=16 时 offset 为 0 而非 16：恰好写满的块「已用 0 个」是错觉，
        # 真实语义是「下一个 token 要写到新块的 offset 0」。T3 写入时不能靠
        # 这个值判断是否还有空位，那是 needs_new_block() 的职责。
        assert table.last_block_offset == last_offset
        assert table.num_full_blocks * table.block_size + table.last_block_offset == seq_len


def test_needs_new_block_transitions() -> None:
    """写满后需要新块，追加块后立刻不再需要。"""
    table = BlockTable(request_id="req-0", block_size=4)
    table.append_block(0)
    assert table.capacity == 4

    table.extend(3)
    # 还差 1 个位置，不需要新块。
    assert table.needs_new_block() is False

    table.extend(1)
    assert table.seq_len == table.capacity == 4
    assert table.needs_new_block() is True

    # append_block 只增 capacity 不动 seq_len，因此判断立即翻转。
    table.append_block(1)
    assert table.capacity == 8
    assert table.seq_len == 4
    assert table.needs_new_block() is False


def test_needs_new_block_with_preallocated_spare_block() -> None:
    """预分配富余块时不得误判为需要新块。"""
    # prefill 分配 3 块（capacity=48）但只写入 32 个 token：第 3 块整块空着。
    table = BlockTable(request_id="req-0", block_size=16, block_ids=[0, 1, 2])
    table.extend(32)

    # 旧公式 seq_len % block_size == 0 在此处为真，会误报需要新块，
    # 导致上层白白多分配一块，在 pool 紧张时触发不必要的分配失败。
    assert table.needs_new_block() is False


def test_extend_guards_capacity_without_partial_update() -> None:
    """extend 超出容量必须失败，且不留下半修改状态。"""
    table = BlockTable(request_id="req-0", block_size=4, block_ids=[0])

    # 容量不足是调用方漏了 append_block，属可预期的调用顺序错误，
    # 与 BlockPool.allocate() 耗尽时同样抛 RuntimeError。
    with pytest.raises(RuntimeError, match="capacity"):
        table.extend(5)
    # 失败后 seq_len 必须保持原值：若实现写成先加再检查，这里会变成 5，
    # 而 5 > capacity 会让后续 position_to_block 越界索引 block_ids。
    assert table.seq_len == 0

    table.extend(4)
    assert table.seq_len == 4
    # 空 prompt 等边界会传 0，应视为合法 no-op 而非错误。
    table.extend(0)
    assert table.seq_len == 4

    with pytest.raises(ValueError, match="non-negative"):
        table.extend(-1)


def test_position_to_block_rejects_out_of_domain() -> None:
    """position_to_block 的定义域严格是 [0, seq_len)。"""
    table = BlockTable(request_id="req-0", block_size=16, block_ids=[0, 1])
    table.extend(20)

    # -1 最危险：若不显式校验，block_ids[-1] 会静默返回物理块 1，
    # 不抛异常却读到无关数据（与 T1 的 block_id=-1 是同一类坑）。
    with pytest.raises(ValueError, match="out of range"):
        table.position_to_block(-1)
    # pos == seq_len 是尚未写入的位置。在 T3 里读到的是 torch.empty 的
    # 未初始化内存，可能含 NaN（见 lessons L5），必须提前拦住。
    with pytest.raises(ValueError, match="out of range"):
        table.position_to_block(20)

    # 最后一个已写入位置必须可读。
    assert table.position_to_block(19) == (1, 3)


def test_invalid_constructor_arguments() -> None:
    """拒绝没有物理意义的构造状态。"""
    # 一个 block 至少要能放一个 token，否则位置映射的除法没有意义。
    for block_size in (0, -1):
        with pytest.raises(ValueError, match="block_size"):
            BlockTable(request_id="req-0", block_size=block_size)

    with pytest.raises(ValueError, match="seq_len"):
        BlockTable(request_id="req-0", block_size=16, seq_len=-1)

    # 声称已有 17 个 token 但只分配了 1 块（capacity=16）：这个状态一旦成立，
    # position_to_block(16) 就会索引到不存在的逻辑块。
    with pytest.raises(ValueError, match="exceeds capacity"):
        BlockTable(request_id="req-0", block_size=16, block_ids=[0], seq_len=17)

    # seq_len == capacity 是「正好写满」，必须允许构造。
    table = BlockTable(request_id="req-0", block_size=16, block_ids=[0], seq_len=16)
    assert table.needs_new_block() is True

    # 负物理块 id 同样会被 Python 当作反向索引，构造期就要拒绝。
    with pytest.raises(ValueError, match="non-negative"):
        BlockTable(request_id="req-0", block_size=16, block_ids=[-1])
    with pytest.raises(ValueError, match="non-negative"):
        BlockTable(request_id="req-0", block_size=16).append_block(-1)


def test_multiple_requests_are_isolated() -> None:
    """不同请求的 block table 必须完全独立。"""
    table_a = BlockTable(request_id="req-a", block_size=4)
    table_b = BlockTable(request_id="req-b", block_size=4)

    table_a.append_block(0)
    table_a.append_block(1)
    table_a.extend(6)

    table_b.append_block(2)
    table_b.extend(3)

    # 这条同时防住 dataclass 可变默认值的经典错误：若 block_ids 写成
    # `= []` 而不是 field(default_factory=list)，两个实例会共享同一个
    # list，此处 table_b.block_ids 会变成 [0, 1, 2]。
    assert table_a.block_ids == [0, 1]
    assert table_b.block_ids == [2]
    assert table_a.seq_len == 6
    assert table_b.seq_len == 3

    assert table_a.position_to_block(5) == (1, 1)
    assert table_b.position_to_block(2) == (2, 2)
