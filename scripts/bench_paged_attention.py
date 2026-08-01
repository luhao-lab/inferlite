"""M4 PagedAttention benchmark。

对比 M3 fixed-slot 和 M4 paged block 两种 KV cache 管理方式，
量化 block 分配、内部碎片和吞吐差异，验证 paged 按需分配的内存收益。

## 目的

M3 fixed-slot 每个请求预留 max_seq_len 连续空间，即使短请求也占满一个 slot。
M4 paged block 按需分配，浪费最多 block_size - 1 个 token（最后一个 block 的碎片）。

本脚本量化：
  - block 分配数量 vs fixed-slot 预留
  - 内部碎片（capacity - used）
  - paged vs fixed-slot 的容量比
  - 吞吐/延迟（仅参考，PyTorch gather 伪版可能更慢）

## 用法

    # 基础对比
    uv run python scripts/bench_paged_attention.py \\
        --model-dir ~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B

    # MPS + bf16
    uv run python scripts/bench_paged_attention.py \\
        --model-dir <path> --device mps --dtype bf16 \\
        --num-requests 8 --block-size 16 --num-blocks 128

    # block_size 扫描
    uv run python scripts/bench_paged_attention.py \\
        --model-dir <path> --block-size-list 8 16 32

## 输出示例

    ============================================================
    M4 PagedAttention Benchmark
    ============================================================
    requests: 8, max_new_tokens: 32, prompt_len: 32, block_size: 16

    ── A. Fixed-slot baseline ──
      capacity_tokens: 4096  (8 slots × 512 max_seq_len)
      used_tokens:     512   (8 × (32 + 32))
      waste_tokens:    3584

    ── B. Paged blocks ──
      allocated_blocks: 24
      capacity_tokens:  384   (24 × 16)
      used_tokens:      512   (wait... used > capacity?)
      internal_frag:    32    (每请求最多 block_size - 1)

    ── Memory comparison ──
      fixed_slot_capacity: 4096
      paged_capacity:      384
      ratio:               0.09x (paged 只需 9% 空间)

完整结果归档：bench/results/YYYY-MM-DD-m4-paged-attention.md
"""

import argparse
import time

import torch
from transformers import AutoTokenizer

from inferlite.cache.kv_cache import KVCache
from inferlite.cli import resolve_device_dtype
from inferlite.engine.core import EngineCore, generate
from inferlite.engine.metrics import MetricsCollector
from inferlite.engine.paged_core import batch_generate_paged
from inferlite.model.weights import load_causal_lm_from_hf
from inferlite.sampler import GreedySampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark M4 PagedAttention.")
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
    parser.add_argument("--num-requests", type=int, default=8, help="Number of requests.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens per request.")
    parser.add_argument("--prompt-len", type=int, default=32, help="Prompt length.")
    parser.add_argument("--block-size", type=int, default=16, help="Block size in tokens.")
    parser.add_argument("--num-blocks", type=int, default=128, help="Total physical blocks.")
    parser.add_argument(
        "--block-size-list",
        type=int,
        nargs="+",
        help="Sweep multiple block_size values (e.g., --block-size-list 8 16 32).",
    )
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max seq len for fixed-slot.")
    parser.add_argument(
        "--max-num-slots", type=int, default=8, help="Slots for fixed-slot baseline."
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs.")
    return parser.parse_args()


def make_prompts(
    num_requests: int, prompt_len: int, pad_token_id: int, device
) -> list[torch.Tensor]:
    """生成固定长度的 pad prompt。"""
    return [
        torch.full((1, prompt_len), pad_token_id, dtype=torch.long, device=device)
        for _ in range(num_requests)
    ]


def make_variable_prompts(num_requests: int, pad_token_id: int, device) -> list[torch.Tensor]:
    """生成长度方差的 prompt（模拟真实场景）。"""
    import random

    random.seed(42)
    lengths = [random.randint(8, 64) for _ in range(num_requests)]
    return [torch.full((1, ln), pad_token_id, dtype=torch.long, device=device) for ln in lengths]


def compute_block_metrics(
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    block_size: int,
    num_blocks: int,
) -> dict:
    """理论计算 paged block 使用指标（不实际运行）。

    假设每个请求生成 max_new_tokens 个 token，
    总长度 = prompt_len + max_new_tokens。
    """
    total_used = 0
    total_blocks_needed = 0
    for prompt in prompts:
        total_len = prompt.shape[1] + max_new_tokens
        total_used += total_len
        blocks = (total_len + block_size - 1) // block_size
        total_blocks_needed += blocks

    capacity_tokens = total_blocks_needed * block_size
    internal_frag = capacity_tokens - total_used

    return {
        "total_used_tokens": total_used,
        "total_blocks_needed": total_blocks_needed,
        "capacity_tokens": capacity_tokens,
        "internal_fragmentation": internal_frag,
        "fragmentation_ratio": internal_frag / capacity_tokens if capacity_tokens > 0 else 0,
    }


def compute_fixed_slot_metrics(
    num_requests: int,
    max_seq_len: int,
    max_num_slots: int,
) -> dict:
    """Fixed-slot 的容量指标。"""
    capacity = max_num_slots * max_seq_len
    return {"capacity_tokens": capacity, "max_num_slots": max_num_slots, "max_seq_len": max_seq_len}


def bench_fixed_slot(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    config,
    max_seq_len,
    max_num_slots,
    device,
    dtype,
):
    """Fixed-slot baseline：M3 batch_generate 或 serial generate。"""
    from inferlite.engine.batch_core import batch_generate

    sampler = GreedySampler()
    metrics = MetricsCollector()
    metrics.max_num_slots = max_num_slots
    eos_token_id = tokenizer.eos_token_id

    start = time.perf_counter()
    with torch.no_grad():
        batch_generate(
            model=model,
            sampler=sampler,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            max_num_slots=max_num_slots,
            config=config,
            max_seq_len=max_seq_len,
            eos_token_id=eos_token_id,
            device=device,
            dtype=dtype,
            metrics=metrics,
        )
    total_ms = (time.perf_counter() - start) * 1000

    summary = metrics.summary()
    summary["total_ms"] = total_ms
    return summary


def bench_paged(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    num_blocks,
    block_size,
    config,
    device,
    dtype,
):
    """M4 paged batch generate。"""
    sampler = GreedySampler()
    metrics = MetricsCollector()
    eos_token_id = tokenizer.eos_token_id

    start = time.perf_counter()
    with torch.no_grad():
        batch_generate_paged(
            model=model,
            sampler=sampler,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            num_blocks=num_blocks,
            block_size=block_size,
            config=config,
            eos_token_id=eos_token_id,
            device=device,
            dtype=dtype,
            metrics=metrics,
        )
    total_ms = (time.perf_counter() - start) * 1000

    summary = metrics.summary()
    summary["total_ms"] = total_ms
    return summary


def print_header(args):
    print()
    print("=" * 60)
    print("M4 PagedAttention Benchmark")
    print("=" * 60)
    print(f"requests: {args.num_requests}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"prompt_len: {args.prompt_len}")
    print(f"block_size: {args.block_size}")
    print(f"num_blocks: {args.num_blocks}")
    print(f"model: {args.model_dir}")
    print(f"device/dtype: {args.device}/{args.dtype}")
    print()


def main():
    args = parse_args()
    device, dtype = resolve_device_dtype(args.device, args.dtype)

    # 加载模型
    print("Loading model...")
    model = load_causal_lm_from_hf(args.model_dir)
    config = model.config
    model.to(device=device, dtype=dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 构造 prompts
    prompts = make_prompts(args.num_requests, args.prompt_len, pad_token_id, device)

    print_header(args)

    # Warmup
    if args.warmup > 0:
        print("Warmup...")
        warmup_cache = KVCache.from_config(
            config,
            batch_size=1,
            max_seq_len=args.max_seq_len,
            dtype=dtype,
            device=device,
        )
        warmup_engine = EngineCore(model=model, sampler=GreedySampler())
        with torch.no_grad():
            _ = generate(
                warmup_engine,
                prompts[0].clone(),
                max_new_tokens=4,
                kv_cache=warmup_cache,
            )
        print()

    # ── Fixed-slot baseline ──
    print("Running fixed-slot baseline...")
    fixed_summary = bench_fixed_slot(
        model,
        tokenizer,
        prompts,
        args.max_new_tokens,
        config,
        args.max_seq_len,
        args.max_num_slots,
        device,
        dtype,
    )
    fixed_metrics = compute_fixed_slot_metrics(
        args.num_requests,
        args.max_seq_len,
        args.max_num_slots,
    )

    print(f"{'─' * 40}")
    print("A. Fixed-slot baseline")
    print(f"{'─' * 40}")
    print(
        f"  capacity_tokens:   {fixed_metrics['capacity_tokens']}  ({args.max_num_slots} slots × {args.max_seq_len})"
    )
    print(f"  total_ms:          {fixed_summary['total_ms']:.2f}")
    print(f"  output_tokens/s:   {fixed_summary.get('output_tokens_per_s', 0):.2f}")
    print()

    # ── Paged blocks ──
    block_size_list = args.block_size_list or [args.block_size]
    for block_size in block_size_list:
        print(f"Running paged blocks (block_size={block_size})...")
        paged_summary = bench_paged(
            model,
            tokenizer,
            prompts,
            args.max_new_tokens,
            args.num_blocks,
            block_size,
            config,
            device,
            dtype,
        )
        paged_metrics = compute_block_metrics(
            prompts,
            args.max_new_tokens,
            block_size,
            args.num_blocks,
        )

        print(f"{'─' * 40}")
        print(f"B. Paged blocks (block_size={block_size})")
        print(f"{'─' * 40}")
        print(f"  blocks_needed:     {paged_metrics['total_blocks_needed']}")
        print(
            f"  capacity_tokens:   {paged_metrics['capacity_tokens']}  ({paged_metrics['total_blocks_needed']} × {block_size})"
        )
        print(f"  used_tokens:       {paged_metrics['total_used_tokens']}")
        print(
            f"  internal_frag:     {paged_metrics['internal_fragmentation']}  ({paged_metrics['fragmentation_ratio']:.1%})"
        )
        print(f"  total_ms:          {paged_summary['total_ms']:.2f}")
        print(f"  output_tokens/s:   {paged_summary.get('output_tokens_per_s', 0):.2f}")
        print()

        # ── Memory comparison ──
        ratio = paged_metrics["capacity_tokens"] / fixed_metrics["capacity_tokens"]
        print(f"{'─' * 40}")
        print("Memory comparison")
        print(f"{'─' * 40}")
        print(f"  fixed_slot_capacity:  {fixed_metrics['capacity_tokens']}")
        print(f"  paged_capacity:       {paged_metrics['capacity_tokens']}")
        print(f"  paged/fixed ratio:    {ratio:.2%}  (paged 只需 {ratio:.0%} 空间)")
        print()

        # ── Throughput comparison ──
        fixed_tps = fixed_summary.get("output_tokens_per_s", 0)
        paged_tps = paged_summary.get("output_tokens_per_s", 0)
        tps_ratio = paged_tps / fixed_tps if fixed_tps > 0 else 0
        print(f"{'─' * 40}")
        print("Throughput comparison (reference only)")
        print(f"{'─' * 40}")
        print(f"  fixed_slot:  {fixed_tps:.2f} tok/s")
        print(f"  paged:       {paged_tps:.2f} tok/s")
        print(f"  ratio:       {tps_ratio:.2f}x")
        print("  (PyTorch gather 伪版可能更慢，M9 kernel 后消除)")
        print()


if __name__ == "__main__":
    main()
