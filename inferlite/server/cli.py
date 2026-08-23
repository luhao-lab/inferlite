"""inferlite serve 命令：启动 OpenAI-compatible HTTP 服务。

用法：
    inferlite serve --model-dir ~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B
    inferlite serve --model-dir /path/to/model --host 0.0.0.0 --port 8000

启动后提供：
    - GET  /health                   健康检查
    - POST /v1/chat/completions      OpenAI-compatible 聊天补全（流式/非流式）

与 inferlite-generate 的区别：
    - generate：单次 CLI 调用，跑完退出
    - serve：常驻 HTTP 服务，支持多请求并发 + continuous batching + SSE 流式输出

架构对齐 vLLM V1：
    vLLM：  vllm serve --model ... → AsyncLLM → EngineCore
    inferlite：inferlite serve --model-dir ... → AsyncEngine → batch_generate_loop
"""

import argparse
from pathlib import Path

import torch
import uvicorn
from transformers import AutoTokenizer

from inferlite.engine.async_engine import AsyncEngine
from inferlite.model.weights import load_causal_lm_from_hf
from inferlite.server.app import create_app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 inferlite serve 命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Start inferlite OpenAI-compatible serving.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Local HF/ModelScope model directory.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="HTTP server bind address (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP server port (default: 8000).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device. 'auto' selects mps > cuda > cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Model and KV cache dtype. 'auto' uses bf16 on mps/cuda, fp32 on cpu.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=128,
        help="Number of KV cache blocks for PagedAttention (default: 128).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Tokens per block (default: 16).",
    )
    return parser.parse_args(argv)


def resolve_device_dtype(device_arg: str, dtype_arg: str) -> tuple[str, torch.dtype]:
    """将 CLI 的 device/dtype 占位符解析为具体值（与 cli.py 共享逻辑）。"""
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device_arg

    if dtype_arg == "auto":
        dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_arg]

    return device, dtype


def main(argv: list[str] | None = None) -> None:
    """inferlite serve 主入口。

    流程：
      1. 解析参数
      2. 加载 tokenizer + model
      3. 创建 AsyncEngine 并启动后台线程
      4. 创建 FastAPI app
      5. 启动 uvicorn HTTP 服务
    """
    args = parse_args(argv)

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    device, dtype = resolve_device_dtype(args.device, args.dtype)

    print(f"Loading model from {model_dir}...")

    # ── 加载 tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )

    # ── 加载模型 ──
    model = load_causal_lm_from_hf(model_dir)
    model.to(device, dtype=dtype)
    model.eval()

    print(f"Model loaded on {device} ({dtype})")

    # ── 创建并启动 AsyncEngine ──
    engine = AsyncEngine(
        model=model,
        tokenizer=tokenizer,
        config=model.config,
        device=device,
        dtype=dtype,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
    )
    engine.start()
    print("Engine started (background thread)")

    # ── 创建 FastAPI app ──
    app = create_app(engine)

    # ── 启动 HTTP 服务 ──
    print(f"Serving on http://{args.host}:{args.port}")
    print("  POST /v1/chat/completions  (OpenAI-compatible)")
    print("  GET  /health               (health check)")

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        engine.stop()
        print("Engine stopped")


if __name__ == "__main__":
    main()
