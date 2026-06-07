"""Pre-flight check for inferlite.

Run before starting M1 to verify:
  1. uv-managed venv has torch + transformers + modelscope
  2. Qwen3-0.6B can be downloaded (default via ModelScope, CN-friendly)
  3. Model loads on the available device (MPS / CUDA / CPU)
  4. Greedy decoding produces a non-empty output

Why ModelScope by default
-------------------------
- huggingface.co is unreachable in CN networks (Connection reset).
- HF mirror env vars (HF_ENDPOINT=https://hf-mirror.com) are validated against the
  official domain in `huggingface_hub>=1.x` and silently fail (FileMetadataError).
- ModelScope is an independent SDK (no httpx-based domain check), Qwen models are
  officially mirrored there with identical sha256.

Usage:
  make preflight
  # or
  uv run python scripts/preflight.py
  uv run python scripts/preflight.py --prompt "Hello" --max-new-tokens 20
  uv run python scripts/preflight.py --source hf            # pull via HuggingFace
  uv run python scripts/preflight.py --source local --model /path/to/dir
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


def ensure_local_model(model_id: str, source: str) -> str:
    """Resolve a model_id to a local directory, downloading if needed.

    source:
        "modelscope" - download from ModelScope (default, CN-friendly)
        "hf"         - download from HuggingFace Hub (needs network access)
        "local"      - treat model_id as a pre-existing local path
    """
    if source == "local":
        import os
        if not os.path.isdir(model_id):
            raise SystemExit(f"--source local but path is not a directory: {model_id}")
        return model_id

    if source == "modelscope":
        # Clean up stale lock files left over from killed downloads.
        # ModelScope's filelock blocks new processes for hours when an old
        # process is gone but its lockfile remains.
        import os
        from pathlib import Path
        lock_dir = Path.home() / ".cache" / "modelscope" / "hub" / ".lock"
        if lock_dir.exists():
            # Best-effort: ignore errors if the lock is genuinely held by another live process.
            for f in lock_dir.iterdir():
                try:
                    f.unlink()
                except OSError:
                    pass

        from modelscope import snapshot_download
        # Default cache: ~/.cache/modelscope/hub/<owner>/<name>
        path = snapshot_download(model_id)
        return path

    if source == "hf":
        from huggingface_hub import snapshot_download as hf_snapshot
        return hf_snapshot(repo_id=model_id)

    raise SystemExit(f"unknown --source: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="model id on the chosen source, or local path when --source=local")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--source",
        choices=["modelscope", "hf", "local"],
        default="modelscope",
        help="where to fetch weights from (default: modelscope, CN-friendly)",
    )
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

    print(f"[3/4] resolving {args.model} via source={args.source} ...")
    t0 = time.time()
    try:
        local_path = ensure_local_model(args.model, args.source)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        if args.source == "modelscope":
            print("  hint: check network; or fall back to --source hf / --source local")
        return 1
    print(f"      local_path   = {local_path}")
    print(f"      resolved in    {time.time() - t0:.1f}s")

    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # local_files_only=True forbids any further network call: once snapshot_download
    # succeeded, transformers must read the on-disk copy and never touch HF.
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    model.eval()
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      loaded in    {time.time() - t0:.1f}s; params = {n_params_m:.1f}M")

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
