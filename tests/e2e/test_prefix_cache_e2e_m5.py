"""M5-T4 E2E: prefix cache 端到端测试。

验证 prefix cache 在完整生成流程中的正确性：
1. 相同 prompt 的两个请求，第二个应命中 prefix cache，输出与第一个完全一致
2. 相同前缀 + 不同后缀，前缀部分的 block 应被复用
3. 有 prefix cache 的 paged generate 与 serial generate 输出等价
4. M4 回归：prefix cache 不影响无共享前缀的请求

容量控制技巧：
    通过设置 num_blocks 限制并发，迫使请求串行执行。
    第一个请求完成后 blocks 进 LRU（带 hash），第二个请求才能命中 prefix cache。
    如果 num_blocks 够大，两个请求同时跑，就不会有 cache hit（各自独立分配）。

运行：
    uv run pytest tests/e2e/test_prefix_cache_e2e_m5.py -v
"""

import torch

from inferlite.cache.kv_cache import KVCache
from inferlite.cache.paged_kv_cache import PagedKVCache
from inferlite.config import ModelConfig
from inferlite.engine.engine import EngineCore, batch_generate_paged, generate
from inferlite.engine.loop import batch_generate_loop
from inferlite.model.qwen3 import Qwen3ForCausalLM
from inferlite.sampler.greedy import GreedySampler
from inferlite.scheduler.fcfs import FCFSScheduler
from inferlite.scheduler.request import RequestState


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
    """逐条串行 generate，每个请求用独立的 M2 KVCache（无 prefix cache）。"""
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


# ---------------------------------------------------------------------------
# E2E-1: 完全相同 prompt → prefix cache 命中 → 输出一致
# ---------------------------------------------------------------------------


def test_prefix_cache_full_hit_same_output():
    """两个完全相同 prompt 的请求，输出应完全一致。

    容量控制：prompt_len=12, block_size=4 → ceil(12/4)=3 blocks/request
    num_blocks=5 → req-0 用 3 blocks (2 free)，req-1 需要 3 blocks → 阻塞
    req-0 完成 → 3 blocks 进 LRU (带 hash) + 2 free = 5 → req-1 命中 prefix cache
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])  # 12 tokens = 3 blocks
    prompts = [prompt, prompt.clone()]

    with torch.no_grad():
        results = batch_generate_paged(
            model=model,
            sampler=GreedySampler(),
            prompts=prompts,
            max_new_tokens=4,  # 12+4=16 tokens → ceil(16/4)=4 blocks, 第4个在 decode 时分配
            num_blocks=5,  # 强制串行：req-0 用 3 后只剩 2，req-1 需要 3 → 阻塞
            block_size=4,
            config=config,
        )

    assert len(results) == 2
    assert torch.equal(results[0], results[1]), (
        f"prefix cache 命中时输出应一致:\n"
        f"  req-0={results[0].tolist()}\n"
        f"  req-1={results[1].tolist()}"
    )


# ---------------------------------------------------------------------------
# E2E-2: 部分前缀匹配 → 前缀 block 复用，后缀不同
# ---------------------------------------------------------------------------


def test_prefix_cache_partial_hit():
    """相同前缀 + 不同后缀，两个请求都应正确生成（与 serial 等价）。

    prompt_0 = [1..8, 50..53]  → blocks: [1..4], [5..8], [50..53]
    prompt_1 = [1..8, 90..93]  → blocks: [1..4], [5..8], [90..93]

    前缀 [1..8] 占 2 个满 block。req-0 跑完后这 2 个 block 带 hash 进 LRU。
    req-1 命中前 2 个 block（prefix cache），第 3 个 block 新分配。
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompts = [
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 50, 51, 52, 53]]),  # prefix [1..8] + suffix A
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 90, 91, 92, 93]]),  # prefix [1..8] + suffix B
    ]

    # serial baseline: 逐条独立生成
    serial_results = _serial_generate(model, prompts, max_new_tokens=4, config=config)

    # paged with prefix cache: num_blocks=5 强制串行
    # 12 tokens → ceil(12/4)=3 blocks/req, 5 blocks → 只能跑 1 个
    with torch.no_grad():
        paged_results = batch_generate_paged(
            model=model,
            sampler=GreedySampler(),
            prompts=prompts,
            max_new_tokens=4,
            num_blocks=5,
            block_size=4,
            config=config,
        )

    assert len(paged_results) == 2
    for i, (serial_out, paged_out) in enumerate(zip(serial_results, paged_results, strict=False)):
        assert torch.equal(serial_out, paged_out), (
            f"请求 {i} 不匹配:\n"
            f"  serial={serial_out.tolist()}\n"
            f"  paged ={paged_out.tolist()}"
        )


# ---------------------------------------------------------------------------
# E2E-3: prefix cache 不影响 serial 等价性
# ---------------------------------------------------------------------------


def test_prefix_cache_serial_equivalence():
    """有 prefix cache 的 paged generate 应与 serial generate 完全等价。

    两个请求共享前缀，但容量足够大让它们同时跑（无 cache hit）。
    即使如此，输出也应与 serial 一致（prefix cache 是透明的优化）。
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompts = [
        torch.tensor([[10, 20, 30, 40, 50]]),  # 5 tokens
        torch.tensor([[10, 20, 30, 40, 60]]),  # 共享前缀 [10,20,30,40]
    ]

    serial_results = _serial_generate(model, prompts, max_new_tokens=5, config=config)

    with torch.no_grad():
        paged_results = batch_generate_paged(
            model=model,
            sampler=GreedySampler(),
            prompts=prompts,
            max_new_tokens=5,
            num_blocks=16,  # 足够大，两个请求可以同时跑
            block_size=4,
            config=config,
        )

    assert len(paged_results) == 2
    for i, (serial_out, paged_out) in enumerate(zip(serial_results, paged_results, strict=False)):
        assert torch.equal(serial_out, paged_out), (
            f"请求 {i} serial vs paged 不匹配:\n"
            f"  serial={serial_out.tolist()}\n"
            f"  paged ={paged_out.tolist()}"
        )


# ---------------------------------------------------------------------------
# E2E-4: 无共享前缀 → M4 回归
# ---------------------------------------------------------------------------


def test_prefix_cache_no_shared_prefix_regression():
    """完全不同的 prompt，prefix cache 不应有任何影响。

    这是 M4 回归测试：即使 prefix cache 代码路径存在，
    无共享前缀的请求应与 serial 完全一致。
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompts = [
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[10, 20, 30]]),
        torch.tensor([[50, 60]]),
    ]

    serial_results = _serial_generate(model, prompts, max_new_tokens=4, config=config)

    with torch.no_grad():
        paged_results = batch_generate_paged(
            model=model,
            sampler=GreedySampler(),
            prompts=prompts,
            max_new_tokens=4,
            num_blocks=16,
            block_size=4,
            config=config,
        )

    assert len(paged_results) == 3
    for i, (serial_out, paged_out) in enumerate(zip(serial_results, paged_results, strict=False)):
        assert torch.equal(
            serial_out, paged_out
        ), f"请求 {i} 不匹配 (prompt_len={prompts[i].shape[1]})"


# ---------------------------------------------------------------------------
# E2E-5: 多轮复用 — 3 个相同 prompt 请求依次命中
# ---------------------------------------------------------------------------


def test_prefix_cache_multi_round_reuse():
    """3 个相同 prompt 请求，每个都应正确生成一致的结果。

    num_blocks=5 限制只能串行执行（每个请求需要 3 blocks），
    每个后续请求都应命中前一个的 prefix cache。
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    prompt = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]])  # 12 tokens
    prompts = [prompt, prompt.clone(), prompt.clone()]

    with torch.no_grad():
        results = batch_generate_paged(
            model=model,
            sampler=GreedySampler(),
            prompts=prompts,
            max_new_tokens=4,
            num_blocks=5,  # 强制串行：3 blocks/req, 5 blocks 只够 1 个
            block_size=4,
            config=config,
        )

    assert len(results) == 3
    # 所有 3 个请求输出应完全一致
    assert torch.equal(
        results[0], results[1]
    ), f"req-0 vs req-1 不一致:\n  {results[0].tolist()}\n  {results[1].tolist()}"
    assert torch.equal(
        results[1], results[2]
    ), f"req-1 vs req-2 不一致:\n  {results[1].tolist()}\n  {results[2].tolist()}"


# ---------------------------------------------------------------------------
# E2E-6: 机制验证 — spy 确认 hash_blocks 和 cache 命中实际发生
# ---------------------------------------------------------------------------


class _PagedKVCacheSpy(PagedKVCache):
    """记录 hash_blocks 调用次数的 PagedKVCache。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_blocks_count = 0

    def hash_blocks(self, request_id, token_ids):
        self.hash_blocks_count += 1
        super().hash_blocks(request_id, token_ids)


def _make_adapter_spy(cache):
    """创建带 allocate_with_cache 计数的 adapter（延迟 import 避免循环引用）。"""
    from inferlite.cache.adapter import PagedCacheAdapter  # noqa: 延迟 import

    class _Spy(PagedCacheAdapter):
        def __init__(self, c):
            super().__init__(c)
            self.allocate_with_cache_count = 0

        def allocate_with_cache(self, request_id, prompt_ids, num_cached):
            self.allocate_with_cache_count += 1
            super().allocate_with_cache(request_id, prompt_ids, num_cached)

    return _Spy(cache)


def test_prefix_cache_mechanism_verified():
    """直接验证 prefix cache 内部机制在 E2E 流程中被触发。

    用 spy adapter 包装 PagedCacheAdapter，确认：
    1. hash_blocks 被调用（在 loop.py 的 prefill/decode 后）
    2. allocate_with_cache 被调用（第二个请求命中 prefix cache）

    手动构造 scheduler + adapter + loop，绕过 batch_generate_paged 的封装。
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    model.eval()

    # 手动构造 PagedKVCache spy + adapter spy
    cache = _PagedKVCacheSpy.from_config(
        config, num_blocks=5, block_size=4, dtype=torch.float32, device="cpu"
    )
    adapter = _make_adapter_spy(cache)
    scheduler = FCFSScheduler(max_num_seqs=5)

    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])  # 12 tokens
    for i in range(2):
        scheduler.submit(
            RequestState(
                request_id=str(i),
                prompt_ids=prompt.clone(),
                max_new_tokens=4,
            )
        )

    with torch.no_grad():
        results = batch_generate_loop(
            model,
            GreedySampler(),
            scheduler,
            adapter,
            prompts=[prompt, prompt.clone()],
            max_new_tokens=4,
            device="cpu",
        )

    # 验证输出正确
    assert len(results) == 2
    assert torch.equal(results[0], results[1]), "相同 prompt 输出应一致"

    # 验证 hash_blocks 被调用（prefill 后 + 每步 decode 后）
    assert cache.hash_blocks_count > 0, f"hash_blocks 应被调用，实际 {cache.hash_blocks_count} 次"

    # 验证第二个请求命中了 prefix cache
    assert adapter.allocate_with_cache_count >= 1, (
        f"allocate_with_cache 应至少被调用 1 次（第二个请求命中），"
        f"实际 {adapter.allocate_with_cache_count} 次"
    )
