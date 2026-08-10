"""M4-T6 E2E: serial generate vs batch_generate_paged 语义等价测试。

核心命题：
    同一组请求逐条串行 generate（M2 KVCache 路径）的结果
    是否等价于 batch_generate_paged（M4 PagedKVCache 路径）的结果？

M4 只改了 KV cache 管理层（fixed-slot → paged block），
模型权重、attention 计算、采样逻辑完全不变，所以 token 序列应该完全一致。

与 M3 test_batch_generate.py 的区别：
    - M3 验证 BatchedKVCache（fixed-slot）路径
    - M4 验证 PagedKVCache（paged block）路径
    - M4 额外验证 block 分配/释放的正确性

运行：
    uv run pytest tests/e2e/test_paged_batch_generate.py -v
"""

import torch

from inferlite.cache.kv_cache import KVCache
from inferlite.cache.paged_kv_cache import PagedKVCache
from inferlite.config import ModelConfig
from inferlite.engine.engine import EngineCore, batch_generate_paged, generate
from inferlite.model.qwen3 import Qwen3ForCausalLM
from inferlite.sampler.greedy import GreedySampler


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


def _serial_generate(
    model: Qwen3ForCausalLM,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    config: ModelConfig,
    max_seq_len: int = 64,
    eos_token_id: int | None = None,
) -> list[torch.Tensor]:
    """逐条串行 generate，每个请求用独立的 M2 KVCache。"""
    sampler = GreedySampler()
    engine = EngineCore(model=model, sampler=sampler)
    results = []
    for prompt in prompts:
        cache = KVCache.from_config(
            config,
            batch_size=1,
            max_seq_len=max_seq_len,
            dtype=torch.float32,
            device="cpu",
        )
        with torch.no_grad():
            out = generate(
                engine,
                prompt.clone(),
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                kv_cache=cache,
            )
        results.append(out)
    return results


def _paged_generate(
    model: Qwen3ForCausalLM,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    config: ModelConfig,
    num_blocks: int = 32,
    block_size: int = 8,
    eos_token_id: int | None = None,
) -> list[torch.Tensor]:
    """M4 paged batch generate。"""
    sampler = GreedySampler()
    with torch.no_grad():
        return batch_generate_paged(
            model=model,
            sampler=sampler,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            num_blocks=num_blocks,
            block_size=block_size,
            config=config,
            eos_token_id=eos_token_id,
        )


# ---------------------------------------------------------------------------
# E2E-1: 基本等价 — 单请求 paged vs serial
# ---------------------------------------------------------------------------


def test_paged_matches_serial_single():
    """单请求：paged 输出应与 serial generate 完全一致。"""
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompts = [torch.tensor([[1, 2, 3]])]

    serial_results = _serial_generate(model, prompts, max_new_tokens=5, config=config)
    paged_results = _paged_generate(model, prompts, max_new_tokens=5, config=config)

    assert len(paged_results) == 1
    assert torch.equal(
        serial_results[0], paged_results[0]
    ), f"不匹配:\n  serial={serial_results[0].tolist()}\n  paged ={paged_results[0].tolist()}"


# ---------------------------------------------------------------------------
# E2E-2: 多请求等价
# ---------------------------------------------------------------------------


def test_paged_matches_serial_multi():
    """多请求：paged 批量输出应与逐条 serial 完全一致。"""
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompts = [
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[4, 5]]),
        torch.tensor([[6, 7, 8, 9]]),
    ]

    serial_results = _serial_generate(model, prompts, max_new_tokens=5, config=config)
    paged_results = _paged_generate(model, prompts, max_new_tokens=5, config=config)

    assert len(paged_results) == len(serial_results)
    for i, (serial_out, paged_out) in enumerate(zip(serial_results, paged_results, strict=False)):
        assert torch.equal(
            serial_out, paged_out
        ), f"请求 {i} 不匹配:\n  serial={serial_out.tolist()}\n  paged ={paged_out.tolist()}"


# ---------------------------------------------------------------------------
# E2E-3: 变长 prompt
# ---------------------------------------------------------------------------


def test_paged_matches_serial_variable_prompts():
    """不同长度的 prompt 不应影响每个请求的独立语义。"""
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompts = [
        torch.tensor([[1]]),
        torch.tensor([[2, 3, 4, 5, 6]]),
        torch.tensor([[7, 8]]),
    ]

    serial_results = _serial_generate(model, prompts, max_new_tokens=4, config=config)
    paged_results = _paged_generate(model, prompts, max_new_tokens=4, config=config)

    for i, (serial_out, paged_out) in enumerate(zip(serial_results, paged_results, strict=False)):
        assert torch.equal(
            serial_out, paged_out
        ), f"请求 {i} (prompt_len={prompts[i].shape[1]}) 不匹配"


# ---------------------------------------------------------------------------
# E2E-4: 跨 block 边界
# ---------------------------------------------------------------------------


def test_paged_cross_block_boundary():
    """prompt 或 output 跨越 block 边界时，token 序列仍应一致。

    block_size=4，prompt_len=5 → prefill 跨 2 个 block
    max_new_tokens=6 → decode 也会跨 block 边界
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompts = [torch.tensor([[10, 20, 30, 40, 50]])]  # len=5, block_size=4 → 2 blocks

    serial_results = _serial_generate(model, prompts, max_new_tokens=6, config=config)
    paged_results = _paged_generate(model, prompts, max_new_tokens=6, config=config, block_size=4)

    assert torch.equal(
        serial_results[0], paged_results[0]
    ), f"跨 block 边界不匹配:\n  serial={serial_results[0].tolist()}\n  paged ={paged_results[0].tolist()}"


# ---------------------------------------------------------------------------
# E2E-5: waiting 队列排空 — 请求数 > 并发容量
# ---------------------------------------------------------------------------


def test_paged_waiting_drain():
    """请求数 > 并发容量时，所有请求最终都应完成且与 serial 一致。

    5 个请求，每个 prompt_len=3 + max_new_tokens=3 → 总长 6
    block_size=4 → 每个请求需要 ceil(6/4) = 2 blocks
    2 并发 × 2 blocks = 至少 4 blocks，留余量给 num_blocks=10
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompts = [torch.tensor([[i, i + 1, i + 2]]) for i in range(5)]

    serial_results = _serial_generate(model, prompts, max_new_tokens=3, config=config)
    paged_results = _paged_generate(
        model, prompts, max_new_tokens=3, config=config, num_blocks=10, block_size=4
    )

    assert len(paged_results) == 5
    for i, (serial_out, paged_out) in enumerate(zip(serial_results, paged_results, strict=False)):
        assert torch.equal(serial_out, paged_out), f"请求 {i} 不匹配"


# ---------------------------------------------------------------------------
# E2E-6: block 释放验证
# ---------------------------------------------------------------------------


def test_paged_block_release():
    """所有请求完成后，block 应全部释放回 pool。

    直接构造 PagedKVCache，模拟 allocate + free 生命周期。
    """
    config = _tiny_config()
    num_blocks = 8
    block_size = 4
    cache = PagedKVCache.from_config(
        config,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
    )

    # 初始：所有 block 空闲
    assert cache.num_free_blocks == num_blocks

    # 分配 3 个请求
    cache.allocate_request("req-0", prompt_len=5)  # 5 tokens → 2 blocks
    cache.allocate_request("req-1", prompt_len=3)  # 3 tokens → 1 block
    cache.allocate_request("req-2", prompt_len=10)  # 10 tokens → 3 blocks
    assert cache.num_free_blocks == num_blocks - 6  # 2 + 1 + 3 = 6 blocks used

    # 释放全部
    cache.free_request("req-0")
    cache.free_request("req-1")
    cache.free_request("req-2")
    assert cache.num_free_blocks == num_blocks  # 全部归还


# ---------------------------------------------------------------------------
# E2E-7: 完整生命周期 — paged generate 后 block 全部归还
# ---------------------------------------------------------------------------


def test_paged_generate_frees_all_blocks():
    """batch_generate_paged 正常结束后，内部 PagedKVCache 应无泄漏。

    通过传入 metrics 间接验证：如果所有请求都正常 finish，
    说明每个请求都走了 free_request 路径（否则会卡死或报错）。
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    from inferlite.engine.metrics import MetricsCollector

    metrics = MetricsCollector()
    sampler = GreedySampler()
    prompts = [
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[4, 5]]),
        torch.tensor([[6, 7, 8, 9]]),
    ]

    with torch.no_grad():
        results = batch_generate_paged(
            model=model,
            sampler=sampler,
            prompts=prompts,
            max_new_tokens=4,
            num_blocks=8,
            block_size=4,
            config=config,
            metrics=metrics,
        )

    # 所有 3 个请求都应有结果
    assert len(results) == 3
    # 所有请求都应 finish（metrics 记录了 finished）
    summary = metrics.summary()
    assert summary["total_output_tokens"] > 0


# ---------------------------------------------------------------------------
# E2E-8: 多请求变长 + 跨 block + waiting drain 综合场景
# ---------------------------------------------------------------------------


def test_paged_comprehensive():
    """综合场景：变长 prompt + 跨 block 边界 + waiting 队列排空。

    prompt 长度: 1, 6, 2, 3；max_new_tokens=4
    最长总长: 6 + 4 = 10 → ceil(10/4) = 3 blocks/req
    4 请求 × 3 blocks = 12，给 num_blocks=16 留余量
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompts = [
        torch.tensor([[1]]),  # len=1
        torch.tensor([[2, 3, 4, 5, 6, 7]]),  # len=6, block_size=4 → 2 blocks
        torch.tensor([[8, 9]]),  # len=2
        torch.tensor([[10, 11, 12]]),  # len=3
    ]

    serial_results = _serial_generate(model, prompts, max_new_tokens=4, config=config)
    paged_results = _paged_generate(
        model, prompts, max_new_tokens=4, config=config, num_blocks=16, block_size=4
    )

    assert len(paged_results) == 4
    for i, (serial_out, paged_out) in enumerate(zip(serial_results, paged_results, strict=False)):
        assert torch.equal(
            serial_out, paged_out
        ), f"请求 {i} (prompt_len={prompts[i].shape[1]}) 不匹配"
