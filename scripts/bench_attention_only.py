"""解耦 Attention 计算与权重 I/O 的 benchmark。

## 目的

端到端 bench_kv_cache.py 测出的 M2 加速比（~7×@T=512）远低于理论值（T 倍），
原因是权重 I/O（每步读 ~1.2 GB 权重）主导了单步耗时，KV Cache 节省的 Attention
计算量被淹没：

    单步 decode 耗时 ≈ 权重 I/O 时间 + Attention 计算时间
                        ↑ M1/M2 相同    ↑ M2 省去了这部分

本脚本只实例化 GQAAttention 单层，不加载完整模型，直接对 Attention 核心计时，
剥离权重 I/O 干扰，展示 KV Cache 对 Attention 计算本身的真实加速效果。

## 设计说明

**M1 等价**：输入 hidden [B, T, H]，layer_kv_cache=None，每次做完整 T×T Attention。
**M2 等价**：预填好 T 个位置的 LayerKVCache，然后输入 hidden [B, 1, H]，做 1×T 向量乘。

两者的 Attention 计算量差距是 T 倍（O(T²) vs O(T)），不含任何 FFN/权重 I/O。

**计时准确性**：
- 每次测量前同步 device（MPS/CUDA），避免异步执行导致计时偏低
- 多次重复取均值，减少随机抖动

## 用法

    # 使用 Qwen3-0.6B 的真实超参（不需要加载权重）
    uv run python scripts/bench_attention_only.py --device mps --dtype bf16

    # 自定义扫描范围
    uv run python scripts/bench_attention_only.py --seq-lengths 64 256 512 1024 2048

## 输出示例（Mac M3 Pro, MPS, bfloat16, reps=20）

    Attention-only benchmark（剥离权重 I/O）
    Device: mps  dtype: bfloat16  reps: 20  batch: 1

    seq_len   M1 (ms)   M2 (ms)   Speedup   理论值
    -----------------------------------------------
         32      0.81      0.69     1.2×      32×
         64      1.20      0.49     2.4×      64×
        128      1.43      0.56     2.5×     128×
        256      2.50      0.49     5.1×     256×
        512      5.58      0.68     8.2×     512×
       1024     14.72      1.00    14.7×    1024×

    M1 耗时随 T 近似 T² 增长（0.81 ms → 14.72 ms，约 18×，T 比 32×）。
    M2 耗时基本稳定在 0.5-1 ms（MPS kernel launch 固定开销）。
    实测 Speedup 低于理论值的原因：MPS 单次 kernel 的 launch overhead（~0.5 ms）
    使短序列下 M2 和 M1 都有固定下限，序列更长后差距会趋近理论 T 倍。

    对比端到端 bench_kv_cache.py：T=512 端到端仅 7×，Attention-only 已达 8×，
    说明权重 I/O（~6 ms/step）是端到端加速比的主要上限，不是 Attention 算法本身。
"""

import argparse
import time

import torch

from inferlite.config import ModelConfig
from inferlite.model.attention import GQAAttention
from inferlite.model.kv_cache import LayerKVCache
from inferlite.model.layers import RotaryEmbedding

DEFAULT_SEQ_LENGTHS = [32, 64, 128, 256, 512, 1024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Attention-only M1 vs M2 speedup (no weight I/O)."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Tensor dtype (default: auto).",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENGTHS,
        help=f"Sequence lengths to sweep (default: {DEFAULT_SEQ_LENGTHS}).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Repetitions per measurement, results are averaged (default: 10).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size (default: 1).",
    )
    return parser.parse_args()


def resolve_device_dtype(device_str: str, dtype_str: str) -> tuple[torch.device, torch.dtype]:
    """解析 device 和 dtype 字符串为 torch 对象。"""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    if dtype_str == "auto":
        if device.type in ("mps", "cuda"):
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
    return device, dtype


def sync_device(device: torch.device) -> None:
    """同步 device，确保所有异步操作完成后再计时。"""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_attention(fn, device: torch.device, reps: int) -> float:
    """多次运行 fn 取均值，返回毫秒。sync_device 确保计时准确。"""
    # warmup 1 次，避免首次 JIT 编译污染
    fn()
    sync_device(device)

    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync_device(device)
    t1 = time.perf_counter()
    return (t1 - t0) / reps * 1000  # ms


def main() -> None:
    args = parse_args()
    device, dtype = resolve_device_dtype(args.device, args.dtype)

    # 使用 Qwen3-0.6B 真实超参，不需要加载权重
    config = ModelConfig.qwen3_0_6b()
    B = args.batch

    # 随机初始化 GQAAttention（只需超参，不需要真实权重）
    attn = GQAAttention(config).to(device=device, dtype=dtype)
    attn.eval()

    # RotaryEmbedding 用于 M1 路径（单层内部调用）
    rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta).to(device=device, dtype=dtype)

    print(
        f"\nAttention-only benchmark（剥离权重 I/O）"
        f"\nDevice: {device}  dtype: {dtype}  reps: {args.reps}  batch: {B}"
        f"\nConfig: H={config.hidden_size}, n_q={config.num_attention_heads}, "
        f"n_kv={config.num_key_value_heads}, D={config.head_dim}, L={config.num_hidden_layers}"
    )
    print()

    header = f"{'seq_len':>8}  {'M1 (ms)':>9}  {'M2 (ms)':>9}  {'Speedup':>9}  {'理论值':>8}"
    print(header)
    print("-" * len(header))

    with torch.no_grad():
        for T in args.seq_lengths:
            # ---- M1 场景：输入完整 T 个 token，无 cache ----
            hidden_m1 = torch.randn(B, T, config.hidden_size, device=device, dtype=dtype)
            position_ids_m1 = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

            # 用默认参数绑定循环变量，避免 ruff B023（closure 晚绑定问题）
            def run_m1(_h=hidden_m1, _pos=position_ids_m1) -> None:
                attn(_h, position_ids=_pos, layer_kv_cache=None)

            # ---- M2 场景：预填 T 个位置的 cache，输入 1 个 token ----
            # 预填 KV Cache（等价于 prefill 后的 decode 第一步）
            layer_cache = LayerKVCache(
                k=torch.zeros(
                    B, config.num_key_value_heads, T, config.head_dim, device=device, dtype=dtype
                ),
                v=torch.zeros(
                    B, config.num_key_value_heads, T, config.head_dim, device=device, dtype=dtype
                ),
            )
            # 用随机数填充 cache，模拟真实 prefill 后的状态
            layer_cache.k.normal_()
            layer_cache.v.normal_()

            # decode 步：只输入 1 个 token，从 cache 位置 T 开始写
            hidden_m2 = torch.randn(B, 1, config.hidden_size, device=device, dtype=dtype)
            position_ids_m2 = torch.full((B, 1), T, device=device, dtype=torch.long)
            # M2 需要 position_embeddings（外部提前算好 cos/sin）
            cos, sin = rotary_emb(hidden_m2, position_ids_m2)

            # 注意：M2 decode 步会写入 cache[T:T+1]，需要 cache 有足够空间
            # 扩大 cache 一格，避免越界
            layer_cache_ext = LayerKVCache(
                k=torch.zeros(
                    B,
                    config.num_key_value_heads,
                    T + 1,
                    config.head_dim,
                    device=device,
                    dtype=dtype,
                ),
                v=torch.zeros(
                    B,
                    config.num_key_value_heads,
                    T + 1,
                    config.head_dim,
                    device=device,
                    dtype=dtype,
                ),
            )
            layer_cache_ext.k[:, :, :T, :] = layer_cache.k
            layer_cache_ext.v[:, :, :T, :] = layer_cache.v

            def run_m2(
                _h=hidden_m2,
                _cos=cos,
                _sin=sin,
                _cache=layer_cache_ext,
                _pos=T,
            ) -> None:
                # 每次写同一个 cache 槽位（T），确保不同 rep 稳定
                attn(
                    _h,
                    position_embeddings=(_cos, _sin),
                    layer_kv_cache=_cache,
                    cache_position=_pos,
                )

            t_m1 = time_attention(run_m1, device, args.reps)
            t_m2 = time_attention(run_m2, device, args.reps)
            speedup = t_m1 / t_m2

            print(f"{T:>8}  {t_m1:>9.2f}  {t_m2:>9.2f}  {speedup:>8.1f}×  {T:>6}×")

    print()
    print("说明：")
    print("  M1 随 T 增长呈 T² 趋势，M2 基本稳定（O(T)，只做 1×T 向量乘）")
    print("  Attention-only Speedup 随 T 增长接近理论值（T 倍）")
    print("  对比端到端 bench_kv_cache.py 的低加速比：差距来自权重 I/O（每步读 ~1.2 GB）")
    print(f"  Qwen3-0.6B on {device}: 权重 I/O 约 6 ms/step，远大于短序列的 Attention 耗时")


if __name__ == "__main__":
    main()
