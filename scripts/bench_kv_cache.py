"""M2 KV Cache 性能对比 benchmark。

对比同一个 prompt 在 M1（无 cache）和 M2（有 cache）两种模式下的推理速度。

用法：
    uv run python scripts/bench_kv_cache.py --model-dir <path>
    uv run python scripts/bench_kv_cache.py --model-dir <path> --device mps --dtype bf16
    uv run python scripts/bench_kv_cache.py --model-dir <path> --prompt-tokens 64 --gen-tokens 128

输出示例：
    Device: mps  |  dtype: bfloat16
    Prompt tokens: 64  |  Generate tokens: 128  |  Warmup: 1

    Mode        Time(s)    tok/s   Speedup
    M1 (no kv)   12.34    10.4x    1.00x
    M2 (kv)       2.11    60.7x    5.84x
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark M1 vs M2 generate speed.")
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
        "--prompt-tokens",
        type=int,
        default=32,
        help="Number of prompt tokens (default: 32). Uses a synthetic repeated token sequence.",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate (default: 64).",
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
        default=1024,
        help="Max sequence length for KV cache (default: 1024).",
    )
    return parser.parse_args()


def run_once(
    engine: EngineCore,
    input_ids: torch.Tensor,
    gen_tokens: int,
    eos_token_id: int | None,
    kv_cache: KVCache | None,
) -> float:
    """运行一次 generate，返回耗时（秒）。"""
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

    # 用 tokenizer 的 pad_token_id 构造一个固定长度的合成 prompt
    # 目的是控制 prompt_tokens 数量精确，方便对比
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    input_ids = torch.full((1, args.prompt_tokens), pad_id, dtype=torch.long, device=device)

    kv_cache = KVCache.from_config(
        model.config,
        batch_size=1,
        max_seq_len=args.max_seq_len,
        dtype=dtype,
        device=device,
    )

    eos_token_id = tokenizer.eos_token_id

    print(f"Device: {device}  |  dtype: {dtype}")
    print(
        f"Prompt tokens: {args.prompt_tokens}  |  "
        f"Generate tokens: {args.gen_tokens}  |  "
        f"Warmup: {args.warmup}"
    )

    # Warmup
    print("\nWarming up ...")
    for _ in range(args.warmup):
        run_once(engine, input_ids, args.gen_tokens, eos_token_id, kv_cache=None)
        run_once(engine, input_ids, args.gen_tokens, eos_token_id, kv_cache=kv_cache)

    # Timing
    print("Timing ...")
    t_m1 = run_once(engine, input_ids, args.gen_tokens, eos_token_id, kv_cache=None)
    t_m2 = run_once(engine, input_ids, args.gen_tokens, eos_token_id, kv_cache=kv_cache)

    tps_m1 = args.gen_tokens / t_m1
    tps_m2 = args.gen_tokens / t_m2
    speedup = t_m1 / t_m2

    print(f"\n{'Mode':<16} {'Time(s)':>8} {'tok/s':>8} {'Speedup':>8}")
    print("-" * 44)
    print(f"{'M1 (no cache)':<16} {t_m1:>8.2f} {tps_m1:>8.1f} {'1.00x':>8}")
    print(f"{'M2 (kv cache)':<16} {t_m2:>8.2f} {tps_m2:>8.1f} {speedup:>7.2f}x")
    print()


if __name__ == "__main__":
    main()
