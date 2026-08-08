"""M2-T2 Qwen3Attention KV Cache 接口测试。

测试分两大类：
1. Cache 读写正确性：写入槽位、追加位置、历史拼接
2. 数值等价性：有 cache 的输出 == 无 cache 的 full attention（误差 < 1e-5）
"""

import pytest
import torch

from inferlite.cache.kv_cache import KVCache
from inferlite.config import ModelConfig
from inferlite.engine.forward_context import AttentionMetadata, set_forward_context
from inferlite.model.attention import Qwen3Attention

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 1
MAX_SEQ = 128


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig.qwen3_0_6b()


@pytest.fixture
def attn(config: ModelConfig) -> Qwen3Attention:
    torch.manual_seed(42)
    return Qwen3Attention(config).eval()


@pytest.fixture
def cache(config: ModelConfig) -> KVCache:
    return KVCache.from_config(
        config,
        batch_size=BATCH,
        max_seq_len=MAX_SEQ,
        dtype=torch.float32,
        device="cpu",
    )


def _call_attn(
    attn: Qwen3Attention,
    x: torch.Tensor,
    pos_start: int,
    layer_kv_cache=None,
    cache_position: int = 0,
) -> torch.Tensor:
    """Helper：用新签名调用 Qwen3Attention，自动设置 ForwardContext。

    Args:
        attn: Qwen3Attention 实例
        x: [B, T, H] hidden states
        pos_start: position_ids 起始位置
        layer_kv_cache: 当前层 LayerKVCache（None 走 M1）
        cache_position: 写入 cache 的起始位置
    """
    B, T, _ = x.shape
    position_ids = torch.arange(pos_start, pos_start + T).unsqueeze(0).expand(B, -1)

    if layer_kv_cache is not None:
        # 绑定 cache 到 Attention 层
        attn.attn.kv_cache = layer_kv_cache
        # 构造 metadata
        if cache_position == 0:
            seq_lens = torch.tensor([T], device=x.device)
        else:
            seq_lens = torch.tensor([cache_position + T], device=x.device)
        metadata = AttentionMetadata(num_seqs=1, seq_lens=seq_lens)
        with set_forward_context(metadata):
            out = attn(position_ids, x)
        # 调用后恢复（避免影响后续 M1 调用）
        attn.attn.kv_cache = None
        return out
    else:
        return attn(position_ids, x)


# ---------------------------------------------------------------------------
# Case 1: prefill 后 cache.k/v[:, :, :T_p, :] 已被写入
# ---------------------------------------------------------------------------


def test_prefill_writes_cache(attn: Qwen3Attention, cache: KVCache, config: ModelConfig) -> None:
    T_p = 5
    x = torch.randn(BATCH, T_p, config.hidden_size)
    with torch.no_grad():
        _call_attn(attn, x, pos_start=0, layer_kv_cache=cache.layers[0], cache_position=0)

    # prefill 写完后，槽位 [0:T_p] 应全部非零
    assert cache.layers[0].k[:, :, :T_p, :].abs().sum() > 0
    assert cache.layers[0].v[:, :, :T_p, :].abs().sum() > 0
    # 槽位 [T_p:] 应仍为零（未写入）
    assert cache.layers[0].k[:, :, T_p:, :].abs().sum() == 0


# ---------------------------------------------------------------------------
# Case 2: decode 步写在 cache_position 位置，不覆盖已有历史
# ---------------------------------------------------------------------------


def test_decode_appends_at_position(
    attn: Qwen3Attention, cache: KVCache, config: ModelConfig
) -> None:
    T_p = 4
    x_p = torch.randn(BATCH, T_p, config.hidden_size)
    with torch.no_grad():
        _call_attn(attn, x_p, pos_start=0, layer_kv_cache=cache.layers[0], cache_position=0)

    # 记录 prefill 写入的 k（后续验证没被覆盖）
    k_prefill = cache.layers[0].k[:, :, :T_p, :].clone()

    # decode 写入前，位置 T_p 应为零
    assert cache.layers[0].k[:, :, T_p, :].abs().sum() == 0

    x_d = torch.randn(BATCH, 1, config.hidden_size)
    with torch.no_grad():
        _call_attn(attn, x_d, pos_start=T_p, layer_kv_cache=cache.layers[0], cache_position=T_p)

    # decode 写入后，位置 T_p 应非零
    assert cache.layers[0].k[:, :, T_p, :].abs().sum() > 0
    # prefill 写入的历史未被覆盖
    assert torch.equal(cache.layers[0].k[:, :, :T_p, :], k_prefill)


# ---------------------------------------------------------------------------
# Case 3: 有 cache 的 decode 步输出 == 无 cache 的 full attention（< 1e-5）
# ---------------------------------------------------------------------------


def test_cache_output_equals_full_attention(
    attn: Qwen3Attention, cache: KVCache, config: ModelConfig
) -> None:
    T = 6
    T_p = 5
    torch.manual_seed(0)
    x = torch.randn(BATCH, T, config.hidden_size)

    # M1 路径：全量 input 一次跑完
    with torch.no_grad():
        out_full = _call_attn(attn, x, pos_start=0)

    # M2 路径：prefill T_p 个 token
    with torch.no_grad():
        _call_attn(attn, x[:, :T_p], pos_start=0, layer_kv_cache=cache.layers[0], cache_position=0)

    # M2 路径：decode 第 T_p 个 token
    with torch.no_grad():
        out_decode = _call_attn(
            attn, x[:, T_p:], pos_start=T_p, layer_kv_cache=cache.layers[0], cache_position=T_p
        )

    # decode 步的输出应与 full attention 的对应位置一致
    assert torch.allclose(
        out_decode, out_full[:, T_p:], atol=1e-5
    ), f"max diff: {(out_decode - out_full[:, T_p:]).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# Case 4: kv_cache=None 时行为与 M1 完全一致（兼容路径）
# ---------------------------------------------------------------------------


def test_m1_compat_no_cache(attn: Qwen3Attention, config: ModelConfig) -> None:
    T = 4
    x = torch.randn(BATCH, T, config.hidden_size)
    with torch.no_grad():
        out1 = _call_attn(attn, x, pos_start=0)
        out2 = _call_attn(attn, x, pos_start=0)
    # 同输入两次调用结果完全相同（无随机性）
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Case 5: prefill 时有 causal mask（T > 1），decode 步无 causal mask（T = 1）
# ---------------------------------------------------------------------------


def test_decode_output_is_finite(attn: Qwen3Attention, cache: KVCache, config: ModelConfig) -> None:
    """decode 步 T=1 不构建 causal mask，输出不含 nan/inf。"""
    T_p = 3
    x_p = torch.randn(BATCH, T_p, config.hidden_size)
    with torch.no_grad():
        _call_attn(attn, x_p, pos_start=0, layer_kv_cache=cache.layers[0], cache_position=0)

    x_d = torch.randn(BATCH, 1, config.hidden_size)
    with torch.no_grad():
        out = _call_attn(
            attn, x_d, pos_start=T_p, layer_kv_cache=cache.layers[0], cache_position=T_p
        )

    assert torch.isfinite(out).all(), "decode 步输出含 nan/inf，可能 causal mask 误加"


# ---------------------------------------------------------------------------
# Case 6: cache 只影响 K/V，不改变 Q（Q 永远来自当前 token）
# ---------------------------------------------------------------------------


def test_q_is_from_current_token_only(
    attn: Qwen3Attention, cache: KVCache, config: ModelConfig
) -> None:
    """同一个 decode 输入，不管 cache 里有什么，输出的 shape 都是 [B, 1, H]。"""
    x_d = torch.randn(BATCH, 1, config.hidden_size)

    # cache 空（cur_len=0）
    with torch.no_grad():
        out_empty = _call_attn(
            attn, x_d, pos_start=0, layer_kv_cache=cache.layers[0], cache_position=0
        )
    assert out_empty.shape == (BATCH, 1, config.hidden_size)

    # cache 有历史
    cache.reset()
    x_p = torch.randn(BATCH, 5, config.hidden_size)
    with torch.no_grad():
        _call_attn(attn, x_p, pos_start=0, layer_kv_cache=cache.layers[0], cache_position=0)
        out_with_hist = _call_attn(
            attn, x_d, pos_start=5, layer_kv_cache=cache.layers[0], cache_position=5
        )
    assert out_with_hist.shape == (BATCH, 1, config.hidden_size)
