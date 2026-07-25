"""M4-T1 BlockPool 单元测试。

本文件验证的是物理 block 元数据池的状态机，而不是 K/V tensor 内容：

    free:      ref_count == 0，block_id 位于 free_block_ids
    allocated: ref_count > 0，block_id 不位于 free_block_ids

M4 暂无 prefix caching 和 CoW，因此测试只覆盖分配、引用计数、释放、
容量检查和异常合同。hash/LRU/partial-hit CoW 属于 M5，不应提前混入。
"""

import pytest

from inferlite.cache.block_pool import BlockPool


def test_initialization_and_allocate_order() -> None:
    """初始化后全部 block 空闲，并按 deque 的 FIFO 顺序分配。"""
    pool = BlockPool(num_blocks=3, block_size=16)

    # block_size 在 T1 只作为 pool 配置保存；真正的位置映射由 T2 BlockTable 使用。
    assert pool.block_size == 16
    # free list 初始包含全部物理 id，低 id 先分配，便于调试和测试复现。
    assert list(pool.free_block_ids) == [0, 1, 2]

    assert [pool.allocate(), pool.allocate(), pool.allocate()] == [0, 1, 2]
    # allocate() 必须同时完成两个状态变化：从 free list 移除，并把 ref_count 置为 1。
    assert all(block.ref_count == 1 for block in pool.blocks)
    assert pool.num_free_blocks == 0


def test_exhaustion_and_can_allocate() -> None:
    """容量预检查和真实分配必须对“池耗尽”给出一致结果。"""
    pool = BlockPool(num_blocks=1, block_size=8)

    # scheduler 后续会用 can_allocate(n) 做 admission；它不能实际消耗 block。
    assert pool.can_allocate(1)
    pool.allocate()
    assert not pool.can_allocate(1)

    # can_allocate() 是正常路径的预检查，allocate() 仍需防御调用方漏检。
    with pytest.raises(RuntimeError, match="No free blocks"):
        pool.allocate()


def test_ref_count_release_and_reuse() -> None:
    """只有最后一个引用释放后，物理 block 才能回到 free list。"""
    pool = BlockPool(num_blocks=2, block_size=8)
    block_id = pool.allocate()  # block 0: ref 0 -> 1

    pool.inc_ref(block_id)  # 模拟未来共享场景：ref 1 -> 2
    pool.dec_ref(block_id)  # 仍有一个引用：ref 2 -> 1，不应释放
    assert pool.blocks[block_id].ref_count == 1
    assert pool.num_free_blocks == 1

    pool.free(block_id)  # 最后一个引用释放：ref 1 -> 0，追加到 deque 队尾
    assert pool.blocks[block_id].ref_count == 0
    assert pool.num_free_blocks == 2

    # 初始 free list 在分配 block 0 后剩 [1]；释放 0 后变成 [1, 0]。
    # 因此 FIFO 复用顺序是先拿 1，再拿刚释放的 0，而不是立即复用 0。
    assert pool.allocate() == 1
    assert pool.allocate() == block_id


def test_invalid_id_and_double_free_are_rejected() -> None:
    """非法 id 和生命周期错误必须显式失败，不能悄悄破坏状态。"""
    pool = BlockPool(num_blocks=2, block_size=8)

    # -1 尤其重要：若不显式校验，Python list 会把它解释为最后一个 block，
    # 从而无声地修改错误对象。2 则是标准的右边界越界。
    for block_id in (-1, 2):
        with pytest.raises(ValueError, match="out of range"):
            pool.inc_ref(block_id)
        with pytest.raises(ValueError, match="out of range"):
            pool.dec_ref(block_id)
        with pytest.raises(ValueError, match="out of range"):
            pool.free(block_id)

    # block 0 从未 allocate，ref_count 已是 0；再次释放属于 double free。
    with pytest.raises(RuntimeError, match="already 0"):
        pool.free(0)

    # 空闲 block 仍在 free list。若允许 inc_ref，它会同时处于“空闲”和“被引用”状态，
    # 下一次 allocate 可能把同一 physical block 再发给另一个调用方。
    with pytest.raises(RuntimeError, match="not allocated"):
        pool.inc_ref(0)


def test_invalid_constructor_and_capacity_arguments() -> None:
    """拒绝没有物理意义的容量参数，并锁定 can_allocate(0) 语义。"""
    # pool 至少要有一个 block；否则任何请求都不可能被调度。
    with pytest.raises(ValueError, match="num_blocks"):
        BlockPool(num_blocks=0, block_size=8)
    # 一个 block 至少容纳一个 token，block_size=0 无法进行位置映射。
    with pytest.raises(ValueError, match="block_size"):
        BlockPool(num_blocks=1, block_size=0)

    pool = BlockPool(num_blocks=1, block_size=8)
    # 分配 0 个 block 永远可满足，这使 required_blocks=0 的通用容量判断保持自然语义。
    assert pool.can_allocate(0)
    # 负需求没有意义；如果直接比较 len(free) >= -1 会错误返回 True。
    with pytest.raises(ValueError, match="non-negative"):
        pool.can_allocate(-1)
