"""M1-M5 统一 benchmark。

同一模型、同一 prompt、同一生成参数，对比 5 个里程碑的吞吐和延迟。

用法:
    uv run python scripts/bench_all_milestones.py \
        --model-dir ~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B \
        --device mps --dtype bf16 \
        --num-requests 4 --max-new-tokens 16 --prompt-len 32
"""

import argparse
import time

import torch
from transformers import AutoTokenizer

from inferlite.cache.kv_cache import KVCache
from inferlite.cli import resolve_device_dtype
from inferlite.engine.engine import EngineCore, batch_generate, batch_generate_paged, generate
from inferlite.engine.metrics import MetricsCollector
from inferlite.model.weights import load_causal_lm_from_hf
from inferlite.sampler import GreedySampler


def parse_args():
    p = argparse.ArgumentParser(description="M1-M5 unified benchmark.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--num-requests", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--num-blocks", type=int, default=128)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=1)
    return p.parse_args()


def bench_m1(model, tokenizer, prompts, max_new_tokens):
    """M1: 无 cache，逐条 serial。"""
    engine = EngineCore(model=model, sampler=GreedySampler())
    eos = tokenizer.eos_token_id

    start = time.perf_counter()
    results = []
    with torch.no_grad():
        for prompt in prompts:
            out = generate(engine, prompt.clone(), max_new_tokens, eos_token_id=eos, kv_cache=None)
            results.append(out)
    total_ms = (time.perf_counter() - start) * 1000

    total_tokens = sum(r.shape[1] - p.shape[1] for r, p in zip(results, prompts, strict=False))
    return {
        "total_ms": total_ms,
        "output_tokens": total_tokens,
        "tok_s": total_tokens / (total_ms / 1000),
    }


def bench_m2(model, tokenizer, prompts, max_new_tokens, config, dtype, device, max_seq_len):
    """M2: KV cache，逐条 serial。"""
    engine = EngineCore(model=model, sampler=GreedySampler())
    eos = tokenizer.eos_token_id

    start = time.perf_counter()
    results = []
    with torch.no_grad():
        for prompt in prompts:
            cache = KVCache.from_config(
                config, batch_size=1, max_seq_len=max_seq_len, dtype=dtype, device=device
            )
            out = generate(engine, prompt.clone(), max_new_tokens, eos_token_id=eos, kv_cache=cache)
            results.append(out)
    total_ms = (time.perf_counter() - start) * 1000

    total_tokens = sum(r.shape[1] - p.shape[1] for r, p in zip(results, prompts, strict=False))
    return {
        "total_ms": total_ms,
        "output_tokens": total_tokens,
        "tok_s": total_tokens / (total_ms / 1000),
    }


def bench_m3(
    model, tokenizer, prompts, max_new_tokens, config, max_seq_len, max_num_slots, eos, device
):
    """M3: batched slot continuous batching。"""
    metrics = MetricsCollector()
    metrics.max_num_slots = max_num_slots

    start = time.perf_counter()
    with torch.no_grad():
        batch_generate(
            model=model,
            sampler=GreedySampler(),
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            max_num_slots=max_num_slots,
            config=config,
            max_seq_len=max_seq_len,
            eos_token_id=eos,
            device=device,
            metrics=metrics,
        )
    total_ms = (time.perf_counter() - start) * 1000

    summary = metrics.summary()
    return {
        "total_ms": total_ms,
        "output_tokens": summary.get("total_output_tokens", 0),
        "tok_s": summary.get("output_tokens_per_s", 0),
    }


def bench_m4(
    model, tokenizer, prompts, max_new_tokens, config, num_blocks, block_size, eos, device
):
    """M4: paged block continuous batching。"""
    metrics = MetricsCollector()

    start = time.perf_counter()
    with torch.no_grad():
        batch_generate_paged(
            model=model,
            sampler=GreedySampler(),
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            num_blocks=num_blocks,
            block_size=block_size,
            config=config,
            eos_token_id=eos,
            device=device,
            metrics=metrics,
        )
    total_ms = (time.perf_counter() - start) * 1000

    summary = metrics.summary()
    return {
        "total_ms": total_ms,
        "output_tokens": summary.get("total_output_tokens", 0),
        "tok_s": summary.get("output_tokens_per_s", 0),
    }


def bench_m5(
    model, tokenizer, prompts, max_new_tokens, config, num_blocks, block_size, eos, device
):
    """M5: paged + prefix cache（同 prompt 命中 cache）。"""
    metrics = MetricsCollector()

    start = time.perf_counter()
    with torch.no_grad():
        batch_generate_paged(
            model=model,
            sampler=GreedySampler(),
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            num_blocks=num_blocks,
            block_size=block_size,
            config=config,
            eos_token_id=eos,
            device=device,
            metrics=metrics,
        )
    total_ms = (time.perf_counter() - start) * 1000

    summary = metrics.summary()
    return {
        "total_ms": total_ms,
        "output_tokens": summary.get("total_output_tokens", 0),
        "tok_s": summary.get("output_tokens_per_s", 0),
    }


def main():
    args = parse_args()
    device, dtype = resolve_device_dtype(args.device, args.dtype)

    print("Loading model...")
    model = load_causal_lm_from_hf(args.model_dir)
    config = model.config
    model.to(device=device, dtype=dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    eos = tokenizer.eos_token_id

    # ── 构造 prompts ──
    # M1-M4: 不同 prompt（无 prefix cache hit）
    torch.manual_seed(42)
    vocab_size = config.vocab_size
    different_prompts = [
        torch.randint(0, vocab_size, (1, args.prompt_len), dtype=torch.long, device=device)
        for _ in range(args.num_requests)
    ]

    # M5: 相同 prompt（触发 prefix cache hit）
    same_prompt = torch.randint(
        0, vocab_size, (1, args.prompt_len), dtype=torch.long, device=device
    )
    same_prompts = [same_prompt.clone() for _ in range(args.num_requests)]

    # ── Header ──
    print()
    print("=" * 68)
    print("M1-M5 Unified Benchmark — Qwen3-0.6B")
    print("=" * 68)
    print(f"  num_requests:    {args.num_requests}")
    print(f"  prompt_len:      {args.prompt_len}")
    print(f"  max_new_tokens:  {args.max_new_tokens}")
    print(f"  max_seq_len:     {args.max_seq_len}")
    print(f"  device/dtype:    {device}/{dtype}")
    print(f"  M5 scenario:    same prompt × {args.num_requests} (prefix cache hit)")
    print()

    # ── Warmup ──
    if args.warmup > 0:
        print("Warmup...")
        cache = KVCache.from_config(
            config, batch_size=1, max_seq_len=args.max_seq_len, dtype=dtype, device=device
        )
        engine = EngineCore(model=model, sampler=GreedySampler())
        with torch.no_grad():
            generate(engine, different_prompts[0].clone(), max_new_tokens=4, kv_cache=cache)
        print()

    # ── Run all milestones ──
    results = {}

    # M1: clear kv_cache bindings (warmup may have set them)
    for layer in model.model.layers:
        layer.self_attn.attn.kv_cache = None

    print("Running M1 (no cache, serial)...")
    results["M1"] = bench_m1(model, tokenizer, different_prompts, args.max_new_tokens)

    print("Running M2 (KV cache, serial)...")
    results["M2"] = bench_m2(
        model,
        tokenizer,
        different_prompts,
        args.max_new_tokens,
        config,
        dtype,
        device,
        args.max_seq_len,
    )

    print("Running M3 (batched slot, continuous batching)...")
    results["M3"] = bench_m3(
        model,
        tokenizer,
        different_prompts,
        args.max_new_tokens,
        config,
        args.max_seq_len,
        args.num_requests,
        eos,
        device,
    )

    print("Running M4 (paged block, continuous batching)...")
    results["M4"] = bench_m4(
        model,
        tokenizer,
        different_prompts,
        args.max_new_tokens,
        config,
        args.num_blocks,
        args.block_size,
        eos,
        device,
    )

    print("Running M5 (paged + prefix cache, same prompts)...")
    results["M5"] = bench_m5(
        model,
        tokenizer,
        same_prompts,
        args.max_new_tokens,
        config,
        args.num_blocks,
        args.block_size,
        eos,
        device,
    )

    # ── Results table ──
    print()
    print("=" * 68)
    print("Results")
    print("=" * 68)
    print(
        f"{'Milestone':<8} {'total_ms':>10} {'tok/s':>10} {'tokens':>8} {'vs M1':>8} {'vs M3':>8}"
    )
    print("-" * 68)

    m1_tps = results["M1"]["tok_s"]
    m3_tps = results["M3"]["tok_s"]

    for name in ["M1", "M2", "M3", "M4", "M5"]:
        r = results[name]
        vs_m1 = f"{r['tok_s'] / m1_tps:.2f}x" if m1_tps > 0 else "—"
        vs_m3 = f"{r['tok_s'] / m3_tps:.2f}x" if m3_tps > 0 else "—"
        print(
            f"{name:<8} {r['total_ms']:>10.1f} {r['tok_s']:>10.2f} {r['output_tokens']:>8} {vs_m1:>8} {vs_m3:>8}"
        )

    print()
    print("Notes:")
    print("  M1-M4: different prompts (no prefix cache hit)")
    print("  M5:    same prompt × N (prefix cache hit, but no skip-prefill yet)")
    print("  M5 vs M4 的差异仅在 block 分配开销，不在 forward 计算量")
    print()


if __name__ == "__main__":
    main()
