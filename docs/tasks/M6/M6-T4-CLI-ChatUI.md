# M6-T4 CLI serve + Chat UI

## 元信息
- **任务 ID**: T4
- **里程碑**: M6
- **状态**: ⬜ pending
- **前置**: T3
- **估时**: 3h

## 目标

**要解决什么问题**：
FastAPI app 已有，但没有启动入口。用户需要一条命令就能起服务，浏览器打开就能聊天。

**做完是什么效果**：
```bash
# 一条命令起服务
inferlite serve --model-dir ~/.cache/modelscope/.../Qwen3-0.6B

# 浏览器打开
open http://localhost:8000

# 或用 curl
curl http://localhost:8000/v1/chat/completions ...

# 或用 OpenAI Python SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
resp = client.chat.completions.create(model="qwen3", messages=[...])
```

**不做什么**（边界）：
- 不做多模型切换
- 不做生产级配置（只暴露 port / host / max-num-seqs）
- 不做 HTTPS

**在推理链路中的位置**：
```
inferlite serve --model-dir <path>
  ↓
cli.serve_main()
  ↓
加载 model + tokenizer
  ↓
创建 AsyncEngine
  ↓
create_app(engine) → FastAPI
  ↓
uvicorn.run(app, host, port)
```

## 产出文件
- `inferlite/cli.py` — 新增 `serve_main` 函数
- `pyproject.toml` — 新增 `inferlite-serve` script
- `inferlite/server/static/chat.html` — 极简 Chat UI

## 算法核心

### CLI serve 命令

```python
# cli.py — 新增

def serve_main():
    """inferlite serve --model-dir <path> [--port 8000] [--host 0.0.0.0]"""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Start inferlite serving")
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    # 1. 加载 model + tokenizer（复用 cli.py 已有逻辑）
    from inferlite.entrypoints import load_model_and_tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        args.model_dir, args.device, args.dtype
    )

    # 2. 创建 AsyncEngine
    from inferlite.engine.async_engine import AsyncEngine
    engine = AsyncEngine(
        model=model,
        tokenizer=tokenizer,
        config=config,
        max_num_seqs=args.max_num_seqs,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        device=model.device,
        dtype=next(model.parameters()).dtype,
    )

    # 3. 创建 FastAPI app
    from inferlite.server.app import create_app
    app = create_app(engine)

    # 4. 启动 uvicorn
    print(f"🚀 inferlite serving on http://{args.host}:{args.port}")
    print(f"   Chat UI: http://localhost:{args.port}")
    print(f"   API:     http://localhost:{args.port}/v1/chat/completions")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
```

`pyproject.toml` 新增：
```toml
[project.scripts]
inferlite-generate = "inferlite.cli:main"
inferlite-serve = "inferlite.cli:serve_main"
```

### Chat UI

单文件 HTML（`server/static/chat.html`），核心功能：

```html
<!DOCTYPE html>
<html>
<head>
    <title>inferlite Chat</title>
    <style>
        /* 深色主题，教学 demo 风格 */
        /* 消息列表 + 输入框 + 发送按钮 */
        /* thinking 内容折叠 */
    </style>
</head>
<body>
    <div id="app">
        <div id="messages"></div>
        <div id="input-area">
            <textarea id="input" placeholder="Type a message..."></textarea>
            <button id="send">Send</button>
        </div>
        <div id="params">
            <label>Temperature: <input id="temp" type="number" value="0.7"></label>
            <label>Max Tokens: <input id="max-tokens" type="number" value="256"></label>
        </div>
    </div>
    <script>
        // SSE 流式接收
        async function send() {
            const messages = [...];
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    model: 'inferlite',
                    messages: messages,
                    stream: true,
                    temperature: parseFloat(document.getElementById('temp').value),
                    max_tokens: parseInt(document.getElementById('max-tokens').value),
                }),
            });

            const reader = response.body.getReader();
            // 逐行读取 SSE，解析 data: {json}
            // 区分 delta.content 和 delta.reasoning_content
            // thinking 内容折叠显示
        }
    </script>
</body>
</html>
```

UI 功能：
- 消息列表（user / assistant 分色显示）
- 底部输入框 + 发送按钮（Enter 发送）
- SSE 流式显示（逐 token 追加）
- thinking 内容折叠（点击展开）
- 参数面板（temperature / max_tokens）
- 深色主题

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|--------|-------------|------|
| 1 | `inferlite serve --help` | 显示参数说明 | exact |
| 2 | `inferlite serve` 启动 | 服务监听 port | exact |
| 3 | Chat UI 加载 | GET / 返回 HTML | exact |
| 4 | Chat UI 发送消息 | SSE 流式返回 | exact |
| 5 | 参数传递 | temperature 生效 | exact |

## DoD
- [ ] 测试 5/5 全绿
- [ ] `inferlite serve --help` 可用
- [ ] `inferlite serve` 可启动服务
- [ ] Chat UI 可在浏览器中使用
- [ ] commit `feat(cli): add inferlite serve + Chat UI (T4 done)`
- [ ] PROGRESS.md 更新

## 坑（按概率排序）
1. `load_model_and_tokenizer` 可能不存在，需要抽取或复用 cli.py 的加载逻辑
2. uvicorn 的 `log_level` 与 inferlite 日志冲突
3. Chat UI 的 SSE 解析（`ReadableStream` + `TextDecoder` 逐行分割）
4. Static file serving 路径（`server/static/` 相对于包安装位置）
5. Chat UI 的 thinking 折叠（`` 标签可能跨 chunk）
