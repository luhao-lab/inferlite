# M6-T3 OpenAI Schemas + FastAPI App

## 元信息
- **任务 ID**: T3
- **里程碑**: M6
- **状态**: ⬜ pending
- **前置**: T2
- **估时**: 4h

## 目标

**要解决什么问题**：
AsyncEngine 提供了 submit/stream/get_output 能力，但没有 HTTP 端点。外部客户端（curl / OpenAI SDK / 浏览器）无法调用。

**做完是什么效果**：
```bash
# 非流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hi"}]}'

# 流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

**不做什么**（边界）：
- 不做 /v1/models、/v1/completions、/v1/embeddings
- 不做 API Key 鉴权
- 不做 rate limit
- 不做 completions API（只做 chat completions）

**在推理链路中的位置**：
```
HTTP POST /v1/chat/completions
  ↓
FastAPI endpoint
  ↓
chat template → prompt_ids
  ↓
SamplingParams 构造
  ↓
AsyncEngine.submit() → AsyncEngine.stream() / get_output()
  ↓
SSE chunk / JSON response
```

## 产出文件
- `inferlite/server/schemas.py` — Pydantic 请求/响应模型
- `inferlite/server/app.py` — FastAPI app + endpoint
- `inferlite/server/__init__.py` — 导出 create_app
- `tests/unit/test_openai_api.py` — 格式兼容测试
- `tests/e2e/test_sse_stream.py` — SSE 流式测试

## 算法核心

### schemas.py

```python
from pydantic import BaseModel

class Message(BaseModel):
    role: str          # "system" | "user" | "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "inferlite"
    messages: list[Message]
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stream: bool = False
    seed: int | None = None
    stop: list[str] | None = None  # 暂不实现，预留字段

class DeltaContent(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None  # Qwen3 <think> 标签内容

class Choice(BaseModel):
    index: int = 0
    delta: DeltaContent | None = None     # stream 模式
    message: DeltaContent | None = None   # 非 stream 模式
    finish_reason: str | None = None      # "stop" | "length" | None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[Choice]
```

### app.py

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from inferlite.engine.async_engine import AsyncEngine

def create_app(engine: AsyncEngine) -> FastAPI:
    app = FastAPI(title="inferlite")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # 1. messages → prompt_ids（tokenizer.apply_chat_template）
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = engine.tokenizer.encode(prompt, return_tensors="pt")

        # 2. 构造 SamplingParams
        params = SamplingParams(
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
        )

        # 3. submit
        eos_id = engine.tokenizer.eos_token_id
        req_id = await engine.submit(prompt_ids, params, request.max_tokens, eos_id)

        # 4. stream 或 非 stream
        completion_id = f"chatcmpl-{req_id[:8]}"
        created = int(time.time())

        if request.stream:
            return StreamingResponse(
                sse_generator(engine, req_id, completion_id, created),
                media_type="text/event-stream",
            )
        else:
            tokens = await engine.get_output(req_id)
            text = engine.tokenizer.decode(tokens, skip_special_tokens=True)
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[Choice(
                    message=DeltaContent(role="assistant", content=text),
                    finish_reason="stop",
                )],
            )

    return app

async def sse_generator(engine, request_id, completion_id, created):
    """生成 SSE 格式的 token 流。"""
    # 首个 chunk：role
    first_chunk = ChatCompletionChunk(
        id=completion_id, created=created, model="inferlite",
        choices=[Choice(delta=DeltaContent(role="assistant"))],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # thinking 状态跟踪
    in_thinking = False
    buffer = ""

    async for token_id in engine.stream(request_id):
        token_text = engine.tokenizer.decode([token_id])

        # 检测 <think> /  标签
        if "" in token_text:
            in_thinking = True
        if "" in token_text:
            in_thinking = False

        if in_thinking:
            delta = DeltaContent(reasoning_content=token_text)
        else:
            delta = DeltaContent(content=token_text)

        chunk = ChatCompletionChunk(
            id=completion_id, created=created, model="inferlite",
            choices=[Choice(delta=delta)],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # 最终 chunk：finish_reason
    final_chunk = ChatCompletionChunk(
        id=completion_id, created=created, model="inferlite",
        choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
```

### reasoning_content 映射

Qwen3 生成 `<think>...</think>` 标签：
- `<think>` 内的 token → `delta.reasoning_content`
- `` 后的 token → `delta.content`

非流式时：
- `message.reasoning_content` = 完整 thinking 内容
- `message.content` = 完整回答内容（去掉 thinking 部分）

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|--------|-------------|------|
| 1 | POST 非流式格式 | 符合 OpenAI ChatCompletionResponse | exact |
| 2 | POST 流式格式 | SSE chunk 有 data: 前缀 + [DONE] | exact |
| 3 | Chat template | messages → prompt_ids 正确 | exact |
| 4 | reasoning_content（流式） | thinking token 在 delta.reasoning_content | exact |
| 5 | reasoning_content（非流式） | message.reasoning_content 有内容 | exact |
| 6 | 参数传递 | temperature/top_p/seed 正确传到 SamplingProcessor | exact |
| 7 | 错误请求 | 无效 JSON → 422 | exact |
| 8 | 空 messages | 合理错误/默认行为 | no crash |

## DoD
- [ ] 测试 8/8 全绿
- [ ] commit `feat(server): add OpenAI-compatible API + SSE streaming (T3 done)`
- [ ] `server/__init__.py` 导出 create_app
- [ ] PROGRESS.md 更新

## 坑（按概率排序）
1. `apply_chat_template` 的 `tokenize` 参数（False 返回字符串，True 返回 id 列表）
2. SSE 格式必须是 `data: {json}\n\n`（两个换行），少一个客户端会卡住
3. `StreamingResponse` 的 `media_type` 必须是 `text/event-stream`
4. thinking 标签可能跨 token 分裂（`` 在两个 token 中）
5. Pydantic v2 的 `model_dump_json()` vs v1 的 `.json()`
