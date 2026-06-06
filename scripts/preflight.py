"""Pre-flight check for inferlite.

Run before starting M1 to verify:
  1. uv-managed venv has torch + transformers
  2. Qwen3-0.6B can be downloaded and run on the available device (MPS / CUDA / CPU)
  3. Greedy decoding produces a non-empty, coherent output

Usage:
  make preflight
  # or
  uv run python scripts/preflight.py
  uv run python scripts/preflight.py --prompt "Hello" --max-new-tokens 20
"""

from __future__ import annotations

import argparse
import sys
import time


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    print("[1/4] importing torch + transformers ...")
    try:
        import torch
        import transformers
    except ImportError as e:
        print(f"  FAIL: {e}")
        print("  hint: run 'make setup' first")
        return 1
    print(f"      torch        = {torch.__version__}")
    print(f"      transformers = {transformers.__version__}")
    print(f"      python       = {sys.version.split()[0]}")

    device = pick_device()
    print(f"[2/4] picking device ... {device}")
    if device == "cpu":
        print("  WARN: no GPU/MPS detected; will run on CPU (slow but OK)")

    print(f"[3/4] downloading + loading {args.model} ...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map=device,
    )
    model.eval()
    print(f"      loaded in {time.time() - t0:.1f}s; params = {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    print(f"[4/4] greedy decode prompt={args.prompt!r} max_new_tokens={args.max_new_tokens} ...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    tps = n_new / elapsed if elapsed > 0 else 0.0
    print(f"      done in {elapsed:.2f}s ({tps:.1f} tok/s)")
    print(f"      output: {text!r}")

    if n_new < 1 or not text.strip():
        print("\nFAIL: empty / no new tokens generated")
        return 2

    print("\nOK: pre-flight passed. ready to start M1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
