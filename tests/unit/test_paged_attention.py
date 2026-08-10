"""M4-T4 PagedAttention 单元测试。

覆盖 Qwen3Attention + PagedKVCache 的 M4 路径，以 M2 single cache 作为 oracle。

测试清单（对应任务卡 DoD）：
  1. B=1 prefill 输出 shape 正确 + 数值与 M2 oracle 对齐
  2. B=1 decode 输出 shape 正确 + 数值与 M2 oracle 对齐
  3. B>1 变长 decode：合批结果等价逐条 decode
  4. 跨 block 边界：prompt 超过 block_size 时 prefill 正确
  5. NaN padding 安全：block 对齐 padding 区不污染短请求输出
  6. 参数合同：block_table / layer_idx 缺失时校验
"""

import pytest
import torch

from inferlite.cache.kv_cache import LayerKVCache
from inferlite.cache.paged_kv_cache import PagedKVCache
from inferlite.config import ModelConfig
from inferlite.engine.context import AttentionMetadata, set_forward_context
from inferlite.model.attention import Qwen3Attention

# ---------------------------------------------------------------------------
# fixtures & helpers
# ---------------------------------------------------------------------------


def _tiny_config(num_hidden_layers: int = 2) -> ModelConfig:
    """足够小的两层配置，避免 Qwen3 全尺寸 tensor。"""
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
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


@pytest.fixture
def config() -> ModelConfig:
    return _tiny_config()


@pytest.fixture
def attn(config: ModelConfig) -> Qwen3Attention:
    return Qwen3Attention(config)


def _make_paged_cache(
    config: ModelConfig,
    num_blocks: int = 8,
    block_size: int = 4,
) -> PagedKVCache:
    return PagedKVCache.from_config(
        config,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
    )


def _make_single_cache(
    config: ModelConfig,
    max_seq_len: int = 32,
) -> LayerKVCache:
    """M2 oracle cache：单序列，无分页。"""
    return LayerKVCache(
        k=torch.zeros(1, config.num_key_value_heads, max_seq_len, config.head_dim),
        v=torch.zeros(1, config.num_key_value_heads, max_seq_len, config.head_dim),
    )


def _build_block_table(cache: PagedKVCache, request_ids: list[str]) -> torch.Tensor:
    """从 cache 的 block_tables 构造 padded block_table tensor。"""
    tables = [cache.block_tables[rid] for rid in request_ids]
    max_blocks = max(t.num_blocks for t in tables)
    block_table = torch.zeros(len(tables), max_blocks, dtype=torch.long)
    for i, t in enumerate(tables):
        block_table[i, : t.num_blocks] = torch.tensor(t.block_ids, dtype=torch.long)
    return block_table


def _call_paged(attn, hidden, pos_start, cache, request_id, layer_idx=0):
    """单请求 paged attention 调用（prefill 或 decode）。"""
    attn.attn.kv_cache = cache
    attn.attn.layer_idx = layer_idx

    table = cache.block_tables[request_id]
    seq_len = table.seq_len
    T = hidden.shape[1]

    block_table = _build_block_table(cache, [request_id])
    seq_lens = torch.tensor([seq_len], dtype=torch.long)
    position_ids = torch.arange(pos_start, pos_start + T).unsqueeze(0)

    metadata = AttentionMetadata(num_seqs=1, seq_lens=seq_lens, block_table=block_table)
    with set_forward_context(metadata):
        out = attn(position_ids, hidden)
    attn.attn.kv_cache = None
    return out


def _call_paged_batched(attn, hidden, cache, request_ids, layer_idx=0):
    """多请求 batched paged attention 调用。"""
    attn.attn.kv_cache = cache
    attn.attn.layer_idx = layer_idx

    block_table = _build_block_table(cache, request_ids)
    tables = [cache.block_tables[rid] for rid in request_ids]
    B = len(request_ids)
    T = hidden.shape[1]
    seq_lens = torch.tensor([t.seq_len for t in tables], dtype=torch.long)

    # position_ids: 每个请求从 0 到 T-1（prefill）或 seq_len-1（decode）
    if T > 1:
        # prefill: 所有请求共享 [0..T-1]
        position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    else:
        # decode: 每个请求在各自 seq_len-1 位置
        position_ids = (seq_lens - 1).unsqueeze(1)

    metadata = AttentionMetadata(num_seqs=B, seq_lens=seq_lens, block_table=block_table)
    with set_forward_context(metadata):
        out = attn(position_ids, hidden)
    attn.attn.kv_cache = None
    return out


def _call_m2(attn, hidden, pos_start, m2_cache, layer_idx=0):
    """M2 oracle 调用（单请求 LayerKVCache）。"""
    attn.attn.kv_cache = m2_cache
    attn.attn.layer_idx = layer_idx

    T = hidden.shape[1]
    seq_lens = torch.tensor([pos_start + T], dtype=torch.long)
    position_ids = torch.arange(pos_start, pos_start + T).unsqueeze(0)

    metadata = AttentionMetadata(num_seqs=1, seq_lens=seq_lens)
    with set_forward_context(metadata):
        out = attn(position_ids, hidden)
    attn.attn.kv_cache = None
    return out


# ===========================================================================
# 1. B=1 Prefill
# ===========================================================================


class TestPagedPrefill:
    """B=1 prefill：单请求整段 prompt 写入 paged cache。"""

    def test_prefill_output_shape(self, attn: Qwen3Attention, config: ModelConfig):
        """prefill 输出 shape 与 hidden_states 一致。"""
        cache = _make_paged_cache(config)
        prompt_len = 5
        cache.allocate_request("a", prompt_len)

        hidden = torch.randn(1, prompt_len, config.hidden_size)
        out = _call_paged(attn, hidden, pos_start=0, cache=cache, request_id="a")

        assert out.shape == (1, prompt_len, config.hidden_size)
        assert torch.isfinite(out).all()

    def test_prefill_matches_m2_oracle(self, attn: Qwen3Attention, config: ModelConfig):
        """M4 paged prefill 与 M2 single cache prefill 数值对齐。

        两者 cache 策略不同（分页 vs 连续），但 attention 数学完全相同：
        同样的 K/V、同样的 causal mask、同样的 output。
        """
        torch.manual_seed(42)
        prompt_len = 5

        # M4 paged path
        paged_cache = _make_paged_cache(config, block_size=4)
        paged_cache.allocate_request("a", prompt_len)
        hidden = torch.randn(1, prompt_len, config.hidden_size)
        out_paged = _call_paged(
            attn, hidden.clone(), pos_start=0, cache=paged_cache, request_id="a"
        )

        # M2 oracle path
        m2_cache = _make_single_cache(config, max_seq_len=32)
        out_m2 = _call_m2(attn, hidden.clone(), pos_start=0, m2_cache=m2_cache)

        assert torch.allclose(
            out_paged, out_m2, atol=1e-4
        ), f"max diff: {(out_paged - out_m2).abs().max()}"


# ===========================================================================
# 2. B=1 Decode
# ===========================================================================


class TestPagedDecode:
    """B=1 decode：单请求单 token decode 步进。"""

    def test_decode_output_shape(self, attn: Qwen3Attention, config: ModelConfig):
        """decode 输出 shape [1, 1, hidden_size]。"""
        cache = _make_paged_cache(config)
        prompt_len = 3
        cache.allocate_request("a", prompt_len)

        # 先 prefill
        hidden_prefill = torch.randn(1, prompt_len, config.hidden_size)
        _call_paged(attn, hidden_prefill, pos_start=0, cache=cache, request_id="a")

        # 再 decode 一步
        cache.append_token("a")
        hidden_decode = torch.randn(1, 1, config.hidden_size)
        out = _call_paged(attn, hidden_decode, pos_start=prompt_len, cache=cache, request_id="a")

        assert out.shape == (1, 1, config.hidden_size)
        assert torch.isfinite(out).all()

    def test_decode_matches_m2_oracle(self, attn: Qwen3Attention, config: ModelConfig):
        """M4 paged decode 与 M2 single cache decode 数值对齐。"""
        torch.manual_seed(42)
        prompt_len = 3

        hidden_prefill = torch.randn(1, prompt_len, config.hidden_size)
        hidden_decode = torch.randn(1, 1, config.hidden_size)

        # M4 paged path
        paged_cache = _make_paged_cache(config, block_size=4)
        paged_cache.allocate_request("a", prompt_len)
        _call_paged(attn, hidden_prefill.clone(), pos_start=0, cache=paged_cache, request_id="a")
        paged_cache.append_token("a")
        out_paged = _call_paged(
            attn, hidden_decode.clone(), pos_start=prompt_len, cache=paged_cache, request_id="a"
        )

        # M2 oracle path
        m2_cache = _make_single_cache(config, max_seq_len=32)
        _call_m2(attn, hidden_prefill.clone(), pos_start=0, m2_cache=m2_cache)
        out_m2 = _call_m2(attn, hidden_decode.clone(), pos_start=prompt_len, m2_cache=m2_cache)

        assert torch.allclose(
            out_paged, out_m2, atol=1e-4
        ), f"max diff: {(out_paged - out_m2).abs().max()}"


# ===========================================================================
# 3. B>1 变长 Decode
# ===========================================================================


class TestPagedBatchDecode:
    """B>1 变长 decode：多请求合批，长度不同。"""

    def test_batch_decode_equivalent_to_sequential(self, attn: Qwen3Attention, config: ModelConfig):
        """合批 paged decode 结果等价逐条 paged decode。"""
        torch.manual_seed(42)
        lengths = [3, 7]
        request_ids = ["a", "b"]

        # 准备 prefill 数据（pad 到 t_max）
        t_max = max(lengths)
        hidden_prefill = torch.randn(2, t_max, config.hidden_size)

        # 合批 prefill
        paged_cache = _make_paged_cache(config, num_blocks=16, block_size=4)
        for rid, plen in zip(request_ids, lengths, strict=False):
            paged_cache.allocate_request(rid, plen)
        _call_paged_batched(attn, hidden_prefill.clone(), paged_cache, request_ids)

        # 合批 decode 一步
        for rid in request_ids:
            paged_cache.append_token(rid)
        hidden_decode = torch.randn(2, 1, config.hidden_size)
        out_batch = _call_paged_batched(attn, hidden_decode.clone(), paged_cache, request_ids)

        # 逐条 oracle
        for i, (rid, plen) in enumerate(zip(request_ids, lengths, strict=False)):
            single_cache = _make_paged_cache(config, num_blocks=16, block_size=4)
            single_cache.allocate_request(rid, plen)
            single_hidden = hidden_prefill[i : i + 1, :plen, :].clone()
            _call_paged(attn, single_hidden, pos_start=0, cache=single_cache, request_id=rid)
            single_cache.append_token(rid)
            out_single = _call_paged(
                attn,
                hidden_decode[i : i + 1].clone(),
                pos_start=plen,
                cache=single_cache,
                request_id=rid,
            )
            assert torch.allclose(
                out_single, out_batch[i : i + 1], atol=1e-4
            ), f"request {rid} (len={plen}) mismatch"


# ===========================================================================
# 4. 跨 Block 边界
# ===========================================================================


class TestCrossBlockBoundary:
    """跨 block 边界：prompt 超过 block_size 时 prefill/decode 正确。"""

    def test_prefill_across_blocks(self, attn: Qwen3Attention, config: ModelConfig):
        """prompt 跨多个 block 时 prefill 数值与 M2 oracle 对齐。"""
        torch.manual_seed(42)
        block_size = 4
        prompt_len = 10  # 需要 3 个 block：[0..3], [4..7], [8..9]

        paged_cache = _make_paged_cache(config, num_blocks=8, block_size=block_size)
        paged_cache.allocate_request("a", prompt_len)
        assert paged_cache.block_tables["a"].num_blocks == 3

        hidden = torch.randn(1, prompt_len, config.hidden_size)
        out_paged = _call_paged(
            attn, hidden.clone(), pos_start=0, cache=paged_cache, request_id="a"
        )

        # M2 oracle
        m2_cache = _make_single_cache(config, max_seq_len=32)
        out_m2 = _call_m2(attn, hidden.clone(), pos_start=0, m2_cache=m2_cache)

        assert torch.allclose(
            out_paged, out_m2, atol=1e-4
        ), f"max diff: {(out_paged - out_m2).abs().max()}"

    def test_decode_after_cross_block_prefill(self, attn: Qwen3Attention, config: ModelConfig):
        """跨 block prefill 后 decode 一步，数值与 M2 oracle 对齐。"""
        torch.manual_seed(42)
        block_size = 4
        prompt_len = 9  # 3 个 block

        hidden_prefill = torch.randn(1, prompt_len, config.hidden_size)
        hidden_decode = torch.randn(1, 1, config.hidden_size)

        # M4 paged
        paged_cache = _make_paged_cache(config, num_blocks=8, block_size=block_size)
        paged_cache.allocate_request("a", prompt_len)
        _call_paged(attn, hidden_prefill.clone(), pos_start=0, cache=paged_cache, request_id="a")
        paged_cache.append_token("a")
        assert paged_cache.block_tables["a"].num_blocks == 3

        out_paged = _call_paged(
            attn, hidden_decode.clone(), pos_start=prompt_len, cache=paged_cache, request_id="a"
        )

        # M2 oracle
        m2_cache = _make_single_cache(config, max_seq_len=32)
        _call_m2(attn, hidden_prefill.clone(), pos_start=0, m2_cache=m2_cache)
        out_m2 = _call_m2(attn, hidden_decode.clone(), pos_start=prompt_len, m2_cache=m2_cache)

        assert torch.allclose(
            out_paged, out_m2, atol=1e-4
        ), f"max diff: {(out_paged - out_m2).abs().max()}"


# ===========================================================================
# 5. NaN Padding 安全
# ===========================================================================


class TestNaNSafety:
    """block 对齐 padding 区即使含 NaN，也不污染输出。"""

    def test_nan_padding_does_not_contaminate(self, attn: Qwen3Attention, config: ModelConfig):
        """短请求的 block padding 区显式填 NaN，输出仍然有限。"""
        torch.manual_seed(42)
        block_size = 4

        # 请求 a: 3 tokens (1 block, 1 padding position)
        # 请求 b: 7 tokens (2 blocks, 1 padding position)
        cache = _make_paged_cache(config, num_blocks=16, block_size=block_size)
        cache.allocate_request("a", 3)
        cache.allocate_request("b", 7)

        # 预填 NaN 到所有 block 的 padding 位置
        for layer in cache.layers:
            layer.k.fill_(float("nan"))
            layer.v.fill_(float("nan"))

        # prefill（pad 到 t_max=7）
        t_max = 7
        hidden = torch.randn(2, t_max, config.hidden_size)
        out = _call_paged_batched(attn, hidden, cache, ["a", "b"])

        assert torch.isfinite(out).all(), "NaN leaked from block padding"

    def test_nan_in_unused_blocks(self, attn: Qwen3Attention, config: ModelConfig):
        """gather_kv 中 block_table padding 的无效 block 含 NaN，不影响输出。"""
        torch.manual_seed(42)
        block_size = 4

        cache = _make_paged_cache(config, num_blocks=16, block_size=block_size)
        cache.allocate_request("a", 3)  # 1 block
        cache.allocate_request("b", 10)  # 3 blocks

        # block 0 的 padding 区域显式填 NaN
        cache.layers[0].k[0, 3:, :, :] = float("nan")
        cache.layers[0].v[0, 3:, :, :] = float("nan")

        t_max = 10
        hidden = torch.randn(2, t_max, config.hidden_size)
        out = _call_paged_batched(attn, hidden, cache, ["a", "b"])

        assert torch.isfinite(out).all(), "NaN leaked from unused block positions"


# ===========================================================================
# 6. 参数合同
# ===========================================================================


class TestParameterContracts:
    """M4 路径的参数校验。"""

    def test_missing_block_table_raises(self, attn: Qwen3Attention, config: ModelConfig):
        """metadata.block_table=None 时，paged path 应报错。"""
        cache = _make_paged_cache(config)
        cache.allocate_request("a", 3)
        hidden = torch.randn(1, 3, config.hidden_size)

        attn.attn.kv_cache = cache
        attn.attn.layer_idx = 0
        # block_table=None → _paged_cache_rw 访问 metadata.block_table 时应 TypeError/AttributeError
        metadata = AttentionMetadata(num_seqs=1, seq_lens=torch.tensor([3]))
        with pytest.raises((TypeError, AttributeError)):
            with set_forward_context(metadata):
                attn(torch.arange(3).unsqueeze(0), hidden)
        attn.attn.kv_cache = None

    def test_missing_layer_idx_raises(self, attn: Qwen3Attention, config: ModelConfig):
        """layer_idx=None 时，paged path 应报错。"""
        cache = _make_paged_cache(config)
        cache.allocate_request("a", 3)
        hidden = torch.randn(1, 3, config.hidden_size)

        attn.attn.kv_cache = cache
        attn.attn.layer_idx = None  # 未设置 layer_idx
        block_table = _build_block_table(cache, ["a"])
        metadata = AttentionMetadata(
            num_seqs=1, seq_lens=torch.tensor([3]), block_table=block_table
        )
        with pytest.raises(TypeError):
            with set_forward_context(metadata):
                attn(torch.arange(3).unsqueeze(0), hidden)
        attn.attn.kv_cache = None

    def test_kv_cache_written_after_prefill(self, config: ModelConfig):
        """prefill 后 cache 中确实写入了 K/V 数据。"""
        attn = Qwen3Attention(config)
        cache = _make_paged_cache(config)
        prompt_len = 5
        cache.allocate_request("a", prompt_len)

        hidden = torch.randn(1, prompt_len, config.hidden_size)
        _call_paged(attn, hidden, pos_start=0, cache=cache, request_id="a")

        # gather 回 oracle 数据，确认非空
        k, v = cache.gather_kv_single(0, "a")
        assert k.shape == (config.num_key_value_heads, prompt_len, config.head_dim)
        assert k.abs().sum() > 0
        assert v.abs().sum() > 0

    def test_kv_cache_written_after_decode(self, config: ModelConfig):
        """decode 一步后 cache 中新增了 K/V 数据。"""
        attn = Qwen3Attention(config)
        cache = _make_paged_cache(config)
        prompt_len = 3
        cache.allocate_request("a", prompt_len)

        # prefill
        hidden = torch.randn(1, prompt_len, config.hidden_size)
        _call_paged(attn, hidden, pos_start=0, cache=cache, request_id="a")

        # decode 一步
        cache.append_token("a")
        hidden_dec = torch.randn(1, 1, config.hidden_size)
        _call_paged(attn, hidden_dec, pos_start=prompt_len, cache=cache, request_id="a")

        # 验证 seq_len 增加了
        assert cache.seq_len_of("a") == prompt_len + 1
        k, v = cache.gather_kv_single(0, "a")
        assert k.shape == (config.num_key_value_heads, prompt_len + 1, config.head_dim)
