"""M6 服务端 CLI：`inferlite serve` 启动 OpenAI 兼容的 HTTP 服务。

用法：
    inferlite serve --model-dir ~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B
    inferlite serve --model-dir <path> --host 0.0.0.0 --port 8000
    inferlite serve --model-dir <path> --device mps --dtype bf16

启动后可用 curl 测试：
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen3","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}'

流式：
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen3","messages":[{"role":"user","content":"Hi"}],"stream":true}'
"""

import argparse

import uvicorn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 inferlite serve 命令行参数。"""
    parser = argparse.ArgumentParser(description="Start inferlite OpenAI-compatible server.")
    parser.add_argument("--model-dir", required=True, help="Local HF/ModelScope model directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
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
        help="Model dtype. 'auto' uses bf16 on gpu, fp32 on cpu.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max sequence length for KV cache (default: 2048).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Server CLI 主入口。"""
    args = parse_args(argv)

    from inferlite.server.app import create_app

    print(f"Loading model from {args.model_dir}...")
    app = create_app(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
    )

    print(f"Starting server at http://{args.host}:{args.port}")
    print("  POST /v1/chat/completions — Chat completion (stream & non-stream)")
    print("  GET  /v1/models           — List models")
    print()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
