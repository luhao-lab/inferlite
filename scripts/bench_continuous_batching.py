"""M3 Continuous Batching benchmark。

对比 serial baseline（max_num_slots=1）和 M3 continuous batching，
拆解 prefill/decode/TTFT/ITL/throughput 指标，证明 M3 的收益来自 decode batching。

## 目的

M2 的 bench_kv_cache.py 只测单请求（batch_size=1），无法体现 M3 的 decode batching 收益。
本脚本用真实 Qwen3-0.6B 构造多请求场景，对比：
  - A. serial baseline：逐条 generate（max_num_slots=1，串行）
  - B. continuous batching：batch_generate（max_num_slots>1，并行 decode）

## 设计说明

**为什么用真实模型？**
- tiny config 的 batch 收益不明显（计算量太小，被 Python 开销主导）
- 真实模型才能反映 decode batching 在 attention/MLP 上的摊销效果

**测量方式**
- 用 pad_token 构造固定长度 prompt（控制变量）
- 两个 baseline 用同一个 model 副本（避免权重随机性差异）
- M3 batch 路径传 MetricsCollector 采集详细指标
- serial 路径手动计时（逐条 generate，总时间 / 总 token 数）

**局限性**
- 教学版不做 MPS/CUDA 同步（时间偏小）
- 合成 pad prompt 非真实分布
- 每组只跑一次，有随机波动

## 用法

    # 基础对比（4 请求，2 slot）
    uv run python scripts/bench_continuous_batching.py \\
        --model-dir ~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B \\
        --num-requests 4 --max-num-slots 2

    # MPS + bf16（Mac 推荐）
    uv run python scripts/bench_continuous_batching.py \\
        --model-dir <path> --device mps --dtype bf16 \\
        --num-requests 8 --max-num-slots 4

    # 扫描不同 slot 数
    uv run python scripts/bench_continuous_batching.py \\
        --model-dir <path> --num-requests 8 \\
        --max-num-slots-list 1 2 4 8

## 输出示例

    ============================================================
    M3 Continuous Batching Benchmark
    ============================================================
    requests: 8, max_new_tokens: 32, prompt_len: 32

    ── A. Serial baseline (max_num_slots=1) ──
      total_ms: 4123.5
      output_tokens_per_s: 62.1
      tpot_ms: 16.1

    ── B. Continuous batching (max_num_slots=4) ──
      prefill_ms_p50: 45.2
      decode_step_ms_p50: 12.3
      ttft_ms_p50: 120.5
      itl_ms_p50: 12.5
      output_tokens_per_s: 210.8
      avg_batch_size: 3.2
      slot_utilization: 0.85

    ── Comparison ──
      serial throughput:  62.1 tok/s
      batch throughput:   210.8 tok/s
      speedup:            3.40x

完整结果归档：bench/results/2026-xx-m3-continuous-batching-*.md
"""

import argparse
import time

import torch
from transformers import AutoTokenizer

from inferlite.cli import resolve_device_dtype
from inferlite.engine.batch_core import batch_generate
from inferlite.engine.core import EngineCore, generate
from inferlite.engine.metrics import MetricsCollector
from inferlite.model.kv_cache import KVCache
from inferlite.model.weights import load_causal_lm_from_hf
from inferlite.sampler import GreedySampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark M3 continuous batching vs serial baseline."
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
        "--num-requests", type=int, default=8, help="Number of requests (default: 8)."
    )
    parser.add_argument(
        "--max-num-slots", type=int, default=4, help="Max concurrent slots (default: 4)."
    )
    parser.add_argument(
        "--max-num-slots-list",
        type=int,
        nargs="+",
        help="Sweep multiple max_num_slots values (e.g., --max-num-slots-list 1 2 4 8).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="Tokens per request (default: 32)."
    )
    parser.add_argument("--prompt-len", type=int, default=32, help="Prompt length (default: 32).")
    parser.add_argument(
        "--max-seq-len", type=int, default=512, help="Max seq len for KV cache (default: 512)."
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (default: 1).")
    return parser.parse_args()


def make_prompts(
    num_requests: int, prompt_len: int, pad_token_id: int, device
) -> list[torch.Tensor]:
    """生成 num_requests 个固定长度的 pad prompt（控制变量），放在 device 上。"""
    return [
        torch.full((1, prompt_len), pad_token_id, dtype=torch.long, device=device)
        for _ in range(num_requests)
    ]


def bench_serial(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    config,
    max_seq_len,
    device,
    dtype,
):
    """A. serial baseline：逐条 generate。"""
    sampler = GreedySampler()
    engine = EngineCore(model=model, sampler=sampler)
    eos_token_id = tokenizer.eos_token_id

    total_start = time.perf_counter()
    total_output_tokens = 0
    for prompt in prompts:
        cache = KVCache.from_config(
            config,
            batch_size=1,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            out = generate(
                engine,
                prompt.clone(),
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                kv_cache=cache,
            )
        # out shape: [1, prompt_len + n]，n <= max_new_tokens
        total_output_tokens += out.shape[1] - prompt.shape[1]
    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "total_ms": total_ms,
        "output_tokens_per_s": total_output_tokens / (total_ms / 1000) if total_ms > 0 else 0.0,
        "tpot_ms": total_ms / total_output_tokens if total_output_tokens > 0 else 0.0,
        "total_output_tokens": total_output_tokens,
    }


def bench_batch(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    max_num_slots,
    config,
    max_seq_len,
    device,
    dtype,
):
    """B. M3 continuous batching。"""
    sampler = GreedySampler()
    metrics = MetricsCollector()
    metrics.max_num_slots = max_num_slots
    eos_token_id = tokenizer.eos_token_id

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

    return metrics.summary()


def print_header(args):
    print()
    print("=" * 60)
    print("M3 Continuous Batching Benchmark")
    print("=" * 60)
    print(f"requests: {args.num_requests}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"prompt_len: {args.prompt_len}")
    print(f"model: {args.model_dir}")
    print(f"device/dtype: {args.device}/{args.dtype}")
    print()


def print_serial_result(result, max_num_slots):
    print(f"{'─' * 40}")
    print(f"A. Serial baseline (max_num_slots={max_num_slots})")
    print(f"{'─' * 40}")
    print(f"  total_ms:            {result['total_ms']:.2f}")
    print(f"  output_tokens_per_s: {result['output_tokens_per_s']:.2f}")
    print(f"  tpot_ms:             {result['tpot_ms']:.2f}")
    print(f"  total_output_tokens: {result['total_output_tokens']}")
    print()


def print_batch_result(summary, max_num_slots):
    print(f"{'─' * 40}")
    print(f"B. Continuous batching (max_num_slots={max_num_slots})")
    print(f"{'─' * 40}")
    print(f"  prefill_ms_p50:      {summary['prefill_ms_p50']:.2f}")
    print(f"  decode_step_ms_p50:  {summary['decode_step_ms_p50']:.2f}")
    print(f"  ttft_ms_p50:         {summary['ttft_ms_p50']:.2f}")
    print(f"  itl_ms_p50:          {summary['itl_ms_p50']:.2f}")
    print(f"  output_tokens_per_s: {summary['output_tokens_per_s']:.2f}")
    print(f"  tpot_ms:             {summary['tpot_ms']:.2f}")
    print(f"  avg_batch_size:      {summary['avg_batch_size']:.2f}")
    print(f"  slot_utilization:    {summary['slot_utilization']:.2f}")
    print(f"  total_decode_ms:     {summary['total_decode_ms']:.2f}")
    print(f"  total_output_tokens: {summary['total_output_tokens']}")
    print()


def print_comparison(serial_tps, batch_tps):
    speedup = batch_tps / serial_tps if serial_tps > 0 else 0.0
    print(f"{'─' * 40}")
    print("Comparison")
    print(f"{'─' * 40}")
    print(f"  serial throughput:  {serial_tps:.2f} tok/s")
    print(f"  batch throughput:   {batch_tps:.2f} tok/s")
    print(f"  speedup:            {speedup:.2f}x")
    print()


def main():
    args = parse_args()
    device, dtype = resolve_device_dtype(args.device, args.dtype)

    # 加载模型 + tokenizer
    print("Loading model...")
    model, config = load_causal_lm_from_hf(args.model_dir), None
    # load_causal_lm_from_hf 返回 Qwen3ForCausalLM，config 从 model.config 取
    from inferlite.model.qwen3 import Qwen3ForCausalLM

    assert isinstance(model, Qwen3ForCausalLM)
    config = model.config
    model.to(device=device, dtype=dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 构造 prompts（放在 device 上，避免 embed_tokens 报 MPS device 错误）
    prompts = make_prompts(args.num_requests, args.prompt_len, pad_token_id, device)

    print_header(args)

    # Warmup（用最短序列跑一次）
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

    # ── A. serial baseline ──
    print("Running serial baseline...")
    serial_result = bench_serial(
        model,
        tokenizer,
        prompts,
        args.max_new_tokens,
        config,
        args.max_seq_len,
        device,
        dtype,
    )
    print_serial_result(serial_result, max_num_slots=1)

    # ── B. continuous batching ──
    slot_list = args.max_num_slots_list or [args.max_num_slots]
    for max_num_slots in slot_list:
        print(f"Running continuous batching (max_num_slots={max_num_slots})...")
        batch_summary = bench_batch(
            model,
            tokenizer,
            prompts,
            args.max_new_tokens,
            max_num_slots,
            config,
            args.max_seq_len,
            device,
            dtype,
        )
        print_batch_result(batch_summary, max_num_slots)
        print_comparison(serial_result["output_tokens_per_s"], batch_summary["output_tokens_per_s"])


if __name__ == "__main__":
    main()
