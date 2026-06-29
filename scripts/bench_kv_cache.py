"""M2 KV Cache 性能对比 benchmark。

## 目的

在不同 prompt 长度下对比 M1（无 cache）和 M2（有 cache）两种模式的推理速度，
展示 KV Cache 加速效果随序列长度增长的变化趋势，验证 O(T²) → O(T) 的理论优化。

## 设计说明

**为什么不放进单测？**
- 单测要求无副作用、秒级完成、不依赖真实大模型权重
- benchmark 需要加载真实权重（几 GB）+ 足够长的序列才能拉开差距
- 两者目的不同：单测验正确性，benchmark 量化性能
- 结果归档到 bench/results/ 供文章引用

**测量方式**
- 用 pad_token 构造固定长度的合成 prompt（控制变量，排除 tokenizer 影响）
- 每次 run_once 包含完整 generate loop，用 time.perf_counter 计时
- Warmup 用最短序列，让 MPS/CUDA 完成 JIT 编译，避免首次调用开销污染数据
- gen_tokens 固定，让 M1/M2 在相同输出长度下比较

**局限性**
- 单条请求（batch_size=1），无法体现 M3 Continuous Batching 的收益
- 合成 pad prompt 非真实分布，实际推理时 Attention 计算量可能不同
- 每个 prompt_length 只跑一次，有随机波动；生产 benchmark 应多次取平均

## 用法

    # 默认扫描 32/64/128/256/512 五个 prompt 长度
    uv run python scripts/bench_kv_cache.py --model-dir <path>

    # MPS + bf16（Mac 上推荐）
    uv run python scripts/bench_kv_cache.py --model-dir <path> --device mps --dtype bf16

    # 自定义扫描长度和生成长度
    uv run python scripts/bench_kv_cache.py --model-dir <path> \\
        --prompt-lengths 64 256 512 1024 --gen-tokens 128

## 输出示例（Qwen3-0.6B, Mac M3 Pro, MPS bf16, gen_tokens=128）

    prompt_tokens   M1 tok/s   M2 tok/s   Speedup
    -----------------------------------------------
               32       13.8       24.9     1.80x
               64       12.7       28.9     2.27x
              128        9.6       25.7     2.67x
              256        6.2       24.1     3.91x
              512        3.3       24.1     7.36x

    说明：
      M1 随 prompt 增长而变慢（每步重算所有历史 token 的 Attention，O(T²)）
      M2 基本保持稳定（decode 阶段每步只算 1 个 token，O(T)）

完整结果归档：bench/results/2026-06-29-m2-kv-cache-mps-bf16.md
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from inferlite.cli import resolve_device_dtype
from inferlite.engine import EngineCore, generate
from inferlite.model.kv_cache import KVCache
from inferlite.model.weights import load_causal_lm_from_hf
from inferlite.sampler import GreedySampler

# 默认扫描的 prompt 长度梯度
DEFAULT_PROMPT_LENGTHS = [32, 64, 128, 256, 512]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark M1 vs M2 generate speed across different prompt lengths."
    )
    parser.add_argument("--model-dir", required=True, help="Local HF/ModelScope model directory.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Model dtype (default: auto).",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_PROMPT_LENGTHS,
        help=(
            f"List of prompt token counts to sweep (default: {DEFAULT_PROMPT_LENGTHS}). "
            "Example: --prompt-lengths 32 128 512"
        ),
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate for each run (default: 64).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs before timing (default: 1).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max sequence length for KV cache pre-allocation (default: 2048).",
    )
    return parser.parse_args()


def run_once(
    engine: EngineCore,
    input_ids: torch.Tensor,
    gen_tokens: int,
    eos_token_id: int | None,
    kv_cache: KVCache | None,
) -> float:
    """运行一次 generate，返回耗时（秒）。

    kv_cache=None 走 M1 路径（每步 full forward）；
    kv_cache 非 None 走 M2 路径（prefill + decode loop）。
    generate 内部会调用 kv_cache.reset()，每次调用都是从头跑，不受上次状态影响。
    """
    with torch.no_grad():
        t0 = time.perf_counter()
        generate(
            engine,
            input_ids.clone(),
            max_new_tokens=gen_tokens,
            eos_token_id=eos_token_id,
            kv_cache=kv_cache,
        )
        t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    args = parse_args()
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    device, dtype = resolve_device_dtype(args.device, args.dtype)

    print(f"\nLoading model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    model = load_causal_lm_from_hf(model_dir)
    model.to(device, dtype=dtype)
    model.eval()

    sampler = GreedySampler()
    engine = EngineCore(model, sampler)

    # 最大 prompt 长度需要小于 max_seq_len - gen_tokens：
    # KV Cache 预分配了 max_seq_len 个槽位，prefill 写 prompt_len，
    # decode 再写 gen_tokens，总写入不能超过 max_seq_len。
    max_prompt = args.max_seq_len - args.gen_tokens
    prompt_lengths = [p for p in args.prompt_lengths if p <= max_prompt]
    if len(prompt_lengths) < len(args.prompt_lengths):
        skipped = [p for p in args.prompt_lengths if p > max_prompt]
        print(
            f"[warn] skipping prompt lengths {skipped} (exceed max_seq_len - gen_tokens = {max_prompt})"
        )

    kv_cache = KVCache.from_config(
        model.config,
        batch_size=1,
        max_seq_len=args.max_seq_len,
        dtype=dtype,
        device=device,
    )

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    eos_token_id = tokenizer.eos_token_id

    print(
        f"Device: {device}  |  dtype: {dtype}  |  gen_tokens: {args.gen_tokens}  |  warmup: {args.warmup}"
    )

    # Warmup：用最短序列跑 warmup 次，让 MPS/CUDA 完成 JIT 编译。
    # 不 warmup 的话第一次调用会包含编译开销，导致数据偏高。
    # M1/M2 两种模式都要 warmup，确保两者都处于稳定状态。
    warmup_ids = torch.full((1, prompt_lengths[0]), pad_id, dtype=torch.long, device=device)
    print("\nWarming up ...")
    for _ in range(args.warmup):
        run_once(engine, warmup_ids, args.gen_tokens, eos_token_id, kv_cache=None)
        run_once(engine, warmup_ids, args.gen_tokens, eos_token_id, kv_cache=kv_cache)

    # 表头
    print(f"\n{'prompt_tokens':>13}  {'M1 tok/s':>9}  {'M2 tok/s':>9}  {'Speedup':>8}")
    print("-" * 47)

    results = []
    for prompt_len in prompt_lengths:
        input_ids = torch.full((1, prompt_len), pad_id, dtype=torch.long, device=device)

        t_m1 = run_once(engine, input_ids, args.gen_tokens, eos_token_id, kv_cache=None)
        t_m2 = run_once(engine, input_ids, args.gen_tokens, eos_token_id, kv_cache=kv_cache)

        tps_m1 = args.gen_tokens / t_m1
        tps_m2 = args.gen_tokens / t_m2
        speedup = t_m1 / t_m2

        results.append((prompt_len, tps_m1, tps_m2, speedup))
        print(f"{prompt_len:>13}  {tps_m1:>9.1f}  {tps_m2:>9.1f}  {speedup:>7.2f}x")

    print()
    print("说明：")
    print("  M1 随 prompt 增长而变慢（每步重算所有历史 token 的 Attention，O(T²)）")
    print("  M2 基本保持稳定（decode 阶段每步只算 1 个 token，O(T)）")


if __name__ == "__main__":
    main()
