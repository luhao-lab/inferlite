"""M4-T5 Paged BatchEngine 单元测试。

测试 batch_generate_paged 的 engine 生命周期：
- block-aware admission（can_allocate 控制准入）
- batched prefill（多请求合并一次前向）
- decode append_token + paged decode
- free_request 释放 block
- EOS / max_new_tokens 终止条件

与 M3 test_batch_engine.py 的对应关系：
- M3 max_num_slots → M4 num_blocks × block_size
- M3 cache_slots / cache_positions → M4 paged_kv_cache / request_ids / is_prefill
- M3 allocate_slot / free_slot → M4 allocate_request / free_request / append_token

测试使用 FakePagedModel（满足 M4 扩展后的 LLMModel Protocol），
不依赖真实 Qwen3 模型，只验证 engine 逻辑。

运行：
  uv run pytest tests/unit/test_paged_batch_engine.py -v
"""

import torch

from inferlite.config import ModelConfig
from inferlite.engine.engine import batch_generate_paged
from inferlite.sampler.greedy import GreedySampler


def _tiny_config() -> ModelConfig:
    """最小模型配置，足够验证 engine 逻辑，不浪费计算。"""
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


class FakePagedModel:
    """满足 M4 LLMModel Protocol（含 paged_kv_cache/request_ids/is_prefill）的 fake model。

    行为：每次调用对所有 batch 行都让 greedy sampler 选中 token 7。
    prefill 时在最后一个位置放高分；decode 时只有一个位置。
    """

    def __init__(self, vocab_size: int, emit_token: int = 7) -> None:
        self.vocab_size = vocab_size
        self.emit_token = emit_token
        self.call_count = 0
        self.prefill_calls = 0
        self.decode_calls = 0
        self.batch_sizes: list[int] = []
        self.seen_request_ids: list[list[str] | None] = []
        self.seen_is_prefill: list[bool] = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        logits_to_keep: int | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: object = None,
        cache_slots: torch.Tensor | None = None,
        cache_positions: torch.Tensor | None = None,
        paged_kv_cache: object = None,
        request_ids: list[str] | None = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        self.call_count += 1
        self.batch_sizes.append(B)
        self.seen_request_ids.append(request_ids)
        self.seen_is_prefill.append(is_prefill)

        # 所有位置都给 emit_token 打高分，这样不管在 plen-1 还是 -1 取 logits 都能正确采样。
        # 变长 prefill 时 engine 在 logits[i, plen-1] 取 logits，不是 logits[i, -1]。
        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, :, self.emit_token] = 10.0

        if is_prefill:
            self.prefill_calls += 1
        else:
            self.decode_calls += 1

        return logits


class EosPagedModel:
    """在 decode 的第 eos_after 步返回 EOS token 得分最高。

    prefill 阶段固定返回 token 50（非 EOS），decode 阶段前 eos_after 步返回 50，
    之后返回 EOS token id。
    """

    def __init__(self, vocab_size: int, eos_token_id: int, eos_after: int) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.eos_after = eos_after
        self._decode_step = 0

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        logits_to_keep: int | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: object = None,
        cache_slots: torch.Tensor | None = None,
        cache_positions: torch.Tensor | None = None,
        paged_kv_cache: object = None,
        request_ids: list[str] | None = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        if is_prefill or T > 1:
            # prefill：选 token 50（非 EOS）
            logits[:, -1, 50] = 10.0
        else:
            # decode：前 eos_after 步选 token 50，之后选 EOS
            if self._decode_step < self.eos_after:
                logits[:, -1, 50] = 10.0
            else:
                logits[:, -1, self.eos_token_id] = 10.0
            self._decode_step += 1
        return logits


# ---------------------------------------------------------------------------
# T5-1: 单请求 paged generate 输出正确
# ---------------------------------------------------------------------------


def test_single_request_output():
    """单请求时 batch_generate_paged 应输出 prompt + generated tokens。"""
    config = _tiny_config()
    model = FakePagedModel(vocab_size=config.vocab_size, emit_token=7)
    sampler = GreedySampler()

    prompt = torch.tensor([[1, 2, 3]])
    results = batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=[prompt],
        max_new_tokens=5,
        num_blocks=16,
        block_size=4,
        config=config,
    )

    assert len(results) == 1
    # prompt(3) + generated(5) = 8 tokens, 全部生成 token 7
    expected = torch.tensor([[1, 2, 3, 7, 7, 7, 7, 7]])
    assert torch.equal(results[0], expected)


# ---------------------------------------------------------------------------
# T5-2: 多请求输出数量等于输入请求数
# ---------------------------------------------------------------------------


def test_multiple_requests_output_count():
    """输出数量应等于输入请求数。"""
    config = _tiny_config()
    model = FakePagedModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4, 5]]),
        torch.tensor([[6]]),
    ]
    results = batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        num_blocks=16,
        block_size=4,
        config=config,
    )

    assert len(results) == 3
    for r in results:
        assert r.shape[1] > 0  # 每个结果非空


# ---------------------------------------------------------------------------
# T5-3: batched prefill — 多个 admitted 请求合并为一次前向
# ---------------------------------------------------------------------------


def test_batched_prefill_single_forward():
    """多个请求同时 admit 时，prefill 应合并为一次 model forward（不是逐条）。"""
    config = _tiny_config()
    model = FakePagedModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    # 3 个请求，block 足够全部 admit
    prompts = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4, 5]]),
        torch.tensor([[6, 7]]),
    ]
    batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        num_blocks=32,  # 充足 block，全部一次 admit
        block_size=4,
        config=config,
    )

    # prefill 应该只有 1 次调用（batched），batch_size = 3
    assert model.prefill_calls == 1, f"Expected 1 prefill call, got {model.prefill_calls}"
    assert (
        model.batch_sizes[0] == 3
    ), f"Expected batch_size=3 for prefill, got {model.batch_sizes[0]}"


# ---------------------------------------------------------------------------
# T5-4: block-aware admission — block 不足时请求留在 waiting
# ---------------------------------------------------------------------------


def test_block_aware_admission():
    """block 不足时请求应留在 waiting，前序完成释放后继续进入。"""
    config = _tiny_config()
    model = FakePagedModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    # 4 个请求，每个 prompt_len=2 → ceil(2/4)=1 block per request
    # 但 max_new_tokens=5 → decode 需要更多 block
    # num_blocks=2 只够同时 admit 1-2 个请求
    prompts = [torch.tensor([[i, i + 1]]) for i in range(4)]
    results = batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        num_blocks=4,  # 有限 block
        block_size=4,
        config=config,
    )

    # 所有 4 个请求最终都应完成
    assert len(results) == 4


# ---------------------------------------------------------------------------
# T5-5: finished 释放 block — free_request 后 num_free_blocks 恢复
# ---------------------------------------------------------------------------


def test_finished_frees_blocks():
    """请求 finished 后 block 应被释放，后续请求能继续进入。

    用 max_num_seqs=1（num_blocks=1 的极端情况不行，因为 1 block 不够 prompt）
    改为 num_blocks 刚好够 1 个请求的 prompt + decode，验证串行执行。
    """
    config = _tiny_config()
    model = FakePagedModel(vocab_size=config.vocab_size, emit_token=7)
    sampler = GreedySampler()

    # prompt_len=3 → ceil(3/4)=1 block, max_new_tokens=2 → 最多 5 tokens → ceil(5/4)=2 blocks
    # num_blocks=2 只够 1 个请求完整执行
    prompts = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]
    results = batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=2,
        num_blocks=2,
        block_size=4,
        config=config,
    )

    assert len(results) == 2
    # 两个请求串行完成
    for r in results:
        assert r.shape[1] == 5  # prompt(3) + generated(2)


# ---------------------------------------------------------------------------
# T5-6: EOS 提前退出
# ---------------------------------------------------------------------------


def test_eos_early_stop():
    """EOS 应使请求提前结束，输出长度小于 prompt + max_new_tokens。"""
    config = _tiny_config()
    eos_id = 42
    model = EosPagedModel(vocab_size=config.vocab_size, eos_token_id=eos_id, eos_after=2)
    sampler = GreedySampler()

    prompt = torch.tensor([[1, 2]])
    results = batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=[prompt],
        max_new_tokens=10,
        num_blocks=16,
        block_size=4,
        config=config,
        eos_token_id=eos_id,
    )

    assert len(results) == 1
    # 输出长度应小于 prompt(2) + max_new_tokens(10)
    assert results[0].shape[1] < 12


# ---------------------------------------------------------------------------
# T5-7: max_new_tokens 到达即 finished
# ---------------------------------------------------------------------------


def test_max_new_tokens_termination():
    """达到 max_new_tokens 后请求应 finished，输出长度正确。"""
    config = _tiny_config()
    model = FakePagedModel(vocab_size=config.vocab_size, emit_token=7)
    sampler = GreedySampler()

    prompt = torch.tensor([[1, 2, 3]])
    results = batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=[prompt],
        max_new_tokens=4,
        num_blocks=16,
        block_size=4,
        config=config,
    )

    assert len(results) == 1
    # prompt(3) + generated(4) = 7 tokens
    assert results[0].shape[1] == 7


# ---------------------------------------------------------------------------
# T5-8: waiting queue 最终清空
# ---------------------------------------------------------------------------


def test_waiting_queue_drained():
    """所有请求最终都应 finished（waiting 清空）。"""
    config = _tiny_config()
    model = FakePagedModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [torch.tensor([[i, i + 1]]) for i in range(5)]
    results = batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        num_blocks=8,
        block_size=4,
        config=config,
    )

    assert len(results) == 5


# ---------------------------------------------------------------------------
# T5-9: request_ids 与 batch 行顺序一致
# ---------------------------------------------------------------------------


def test_request_ids_match_batch_order():
    """request_ids 顺序必须与 batch input 行顺序一致。

    FakePagedModel 记录每次调用看到的 request_ids，
    验证 prefill 时 request_ids 与 admitted 请求顺序匹配。
    """
    config = _tiny_config()
    model = FakePagedModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[10, 20]]),
        torch.tensor([[30, 40, 50]]),
    ]
    batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=2,
        num_blocks=16,
        block_size=4,
        config=config,
    )

    # prefill 调用（第一次）应该看到 request_ids = ["0", "1"]（按提交顺序）
    prefill_rids = model.seen_request_ids[0]
    assert prefill_rids is not None
    assert prefill_rids == ["0", "1"]


# ---------------------------------------------------------------------------
# T5-10: is_prefill 标志正确传递
# ---------------------------------------------------------------------------


def test_is_prefill_flag_correct():
    """prefill 调用 is_prefill=True，decode 调用 is_prefill=False。"""
    config = _tiny_config()
    model = FakePagedModel(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompts = [torch.tensor([[1, 2]])]
    batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=3,
        num_blocks=16,
        block_size=4,
        config=config,
    )

    # 第一次调用是 prefill (True)，后续是 decode (False)
    assert model.seen_is_prefill[0] is True
    for is_pf in model.seen_is_prefill[1:]:
        assert is_pf is False


# ---------------------------------------------------------------------------
# T5-11: 多请求 paged generate 与逐条输出 token 级等价
# ---------------------------------------------------------------------------


def test_paged_matches_sequential():
    """多请求 batched paged generate 应与逐条单请求 generate 输出 token 级等价。

    使用 FakePagedModel（所有请求输出相同 token），所以 batched 和 sequential
    应该产生完全相同的 token 序列。
    """
    config = _tiny_config()
    sampler = GreedySampler()

    prompts = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4, 5]]),
        torch.tensor([[6]]),
    ]

    # batched paged
    model_batched = FakePagedModel(vocab_size=config.vocab_size, emit_token=7)
    results_batched = batch_generate_paged(
        model=model_batched,
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=4,
        num_blocks=32,
        block_size=4,
        config=config,
    )

    # 逐条 sequential
    results_sequential = []
    for prompt in prompts:
        model_seq = FakePagedModel(vocab_size=config.vocab_size, emit_token=7)
        r = batch_generate_paged(
            model=model_seq,
            sampler=sampler,
            prompts=[prompt],
            max_new_tokens=4,
            num_blocks=32,
            block_size=4,
            config=config,
        )
        results_sequential.append(r[0])

    # token 级比较
    for i in range(len(prompts)):
        assert torch.equal(results_batched[i], results_sequential[i]), (
            f"Request {i}: batched={results_batched[i].tolist()}, "
            f"sequential={results_sequential[i].tolist()}"
        )


# ---------------------------------------------------------------------------
# T5-12: 跨 block decode — 长度正好跨 block 边界时输出正确
# ---------------------------------------------------------------------------


def test_cross_block_decode():
    """当生成 token 数使序列长度跨过 block 边界时，输出仍正确。

    block_size=4, prompt_len=3 → 占 1 block (positions 0-2)。
    decode 第 2 个 token 时 seq_len=5 → 需要第 2 个 block。
    """
    config = _tiny_config()
    model = FakePagedModel(vocab_size=config.vocab_size, emit_token=7)
    sampler = GreedySampler()

    prompt = torch.tensor([[1, 2, 3]])  # prompt_len=3
    results = batch_generate_paged(
        model=model,
        sampler=sampler,
        prompts=[prompt],
        max_new_tokens=5,  # 总长 3+5=8，跨 2 个 block（block_size=4）
        num_blocks=16,
        block_size=4,
        config=config,
    )

    assert len(results) == 1
    assert results[0].shape[1] == 8
    expected = torch.tensor([[1, 2, 3, 7, 7, 7, 7, 7]])
    assert torch.equal(results[0], expected)


# ---------------------------------------------------------------------------
# T5-13: M3 fixed-slot path 回归 — batch_generate 不受影响
# ---------------------------------------------------------------------------


def test_m3_batch_generate_regression():
    """M3 batch_generate 路径不受 paged 改动影响，继续正常工作。"""
    from inferlite.engine.engine import batch_generate

    config = _tiny_config()

    class FakeM3Model:
        def __init__(self, vocab_size: int) -> None:
            self.vocab_size = vocab_size
            self._step = 0

        def __call__(
            self,
            input_ids,
            *,
            logits_to_keep=None,
            position_ids=None,
            kv_cache=None,
            cache_slots=None,
            cache_positions=None,
            **kwargs,
        ):
            B, T = input_ids.shape
            logits = torch.zeros(B, T, self.vocab_size)
            logits[:, -1, 7] = 10.0
            return logits

    model = FakeM3Model(vocab_size=config.vocab_size)
    sampler = GreedySampler()

    prompt = torch.tensor([[1, 2]])
    results = batch_generate(
        model=model,
        sampler=sampler,
        prompts=[prompt],
        max_new_tokens=3,
        max_num_slots=2,
        config=config,
        max_seq_len=64,
    )

    assert len(results) == 1
    assert results[0].shape[1] == 5  # prompt(2) + generated(3)
