import pytest

from inferlite.cache.block_pool import BlockPool


def test_initialization_and_allocate_order() -> None:
    pool = BlockPool(num_blocks=3, block_size=16)

    assert pool.block_size == 16
    assert list(pool.free_block_ids) == [0, 1, 2]
    assert [pool.allocate(), pool.allocate(), pool.allocate()] == [0, 1, 2]
    assert all(block.ref_count == 1 for block in pool.blocks)
    assert pool.num_free_blocks == 0


def test_exhaustion_and_can_allocate() -> None:
    pool = BlockPool(num_blocks=1, block_size=8)

    assert pool.can_allocate(1)
    pool.allocate()
    assert not pool.can_allocate(1)
    with pytest.raises(RuntimeError, match="No free blocks"):
        pool.allocate()


def test_ref_count_release_and_reuse() -> None:
    pool = BlockPool(num_blocks=2, block_size=8)
    block_id = pool.allocate()

    pool.inc_ref(block_id)
    pool.dec_ref(block_id)
    assert pool.blocks[block_id].ref_count == 1
    assert pool.num_free_blocks == 1

    pool.free(block_id)
    assert pool.blocks[block_id].ref_count == 0
    assert pool.num_free_blocks == 2
    assert pool.allocate() == 1
    assert pool.allocate() == block_id


def test_invalid_id_and_double_free_are_rejected() -> None:
    pool = BlockPool(num_blocks=2, block_size=8)

    for block_id in (-1, 2):
        with pytest.raises(ValueError, match="out of range"):
            pool.inc_ref(block_id)
        with pytest.raises(ValueError, match="out of range"):
            pool.dec_ref(block_id)
        with pytest.raises(ValueError, match="out of range"):
            pool.free(block_id)

    with pytest.raises(RuntimeError, match="already 0"):
        pool.free(0)

    with pytest.raises(RuntimeError, match="not allocated"):
        pool.inc_ref(0)


def test_invalid_constructor_and_capacity_arguments() -> None:
    with pytest.raises(ValueError, match="num_blocks"):
        BlockPool(num_blocks=0, block_size=8)
    with pytest.raises(ValueError, match="block_size"):
        BlockPool(num_blocks=1, block_size=0)

    pool = BlockPool(num_blocks=1, block_size=8)
    assert pool.can_allocate(0)
    with pytest.raises(ValueError, match="non-negative"):
        pool.can_allocate(-1)
