"""FastAPI 应用：OpenAI-compatible /v1/chat/completions 端点。

提供两个核心端点：
  - GET  /health：健康检查
  - POST /v1/chat/completions：聊天补全（支持流式/非流式）

流式输出使用 SSE（Server-Sent Events）：
  - 每个 chunk 格式：data: {"id":"...","choices":[{"delta":{"content":"..."}}]}\n\n
  - 结束信号：data: [DONE]\n\n
  - 对齐 OpenAI 的 stream chunk 格式

思维链（reasoning_content）：
  - Qwen3 输出中的 <think>...</think> 标签内容会被提取
  - 映射到 response 的 reasoning_content 字段
  - 非 <think> 部分映射到 content 字段

架构对齐：
  - vLLM V1：API Server → AsyncLLM → EngineCore
  - inferlite M6：FastAPI app → AsyncEngine → batch_generate_loop
"""

import asyncio
import queue
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inferlite.engine.async_engine import AsyncEngine, _Sentinel
from inferlite.sampler.sampling import SamplingParams
from inferlite.server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    DeltaMessage,
    ResponseChoice,
    ResponseMessage,
    StreamChoice,
    UsageInfo,
)

# ── FastAPI app ──

app = FastAPI(title="inferlite", version="0.1.0")

# engine 在 startup 时注入，通过 app.state.engine 访问
# （见 create_app() 工厂函数）


def create_app(engine: AsyncEngine) -> FastAPI:
    """创建 FastAPI 应用实例，注入引擎依赖。

    Args:
        engine: 已启动的 AsyncEngine 实例

    Returns:
        配置好路由和依赖的 FastAPI 应用
    """
    application = FastAPI(title="inferlite", version="0.1.0")
    application.state.engine = engine

    # 422 校验失败时打印请求体，方便排查客户端发了什么不合法的内容
    from fastapi.exceptions import RequestValidationError
    from starlette.responses import JSONResponse as StarletteJSONResponse

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        body = await request.body()
        print(f"[422] Validation error: {exc.errors()}")
        print(f"[422] Request body: {body.decode('utf-8', errors='replace')[:500]}")
        return StarletteJSONResponse(status_code=422, content={"detail": exc.errors()})

    @application.get("/health")
    async def health():
        """健康检查端点。"""
        return {"status": "ok"}

    @application.get("/v1/models")
    async def list_models():
        """列出可用模型（OpenAI-compatible）。

        Open WebUI 等客户端通过此端点获取模型列表。
        返回格式对齐 OpenAI /v1/models 响应。
        """
        # 从 engine.config 获取模型信息
        model_id = getattr(application.state.engine.config, "model_name_or_path", "inferlite")
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "permission": [],
                    "root": model_id,
                    "parent": None,
                }
            ],
        }

    @application.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
        """OpenAI-compatible 聊天补全端点。

        根据 request.stream 决定返回模式：
          - stream=false：等待全部生成完毕，返回完整 ChatCompletionResponse
          - stream=true：SSE 流式推送 ChatCompletionStreamChunk
        """
        eng: AsyncEngine = raw_request.app.state.engine

        # ── 1. 构造 prompt：messages → chat template → token ids ──
        messages = []
        for m in request.messages:
            # content 可能是 list（OpenAI 多模态格式 [{"type":"text","text":"..."}]）
            # 这里简化为只取文本
            content = m.content if isinstance(m.content, str) else str(m.content)
            messages.append({"role": m.role, "content": content})
        prompt_text = eng.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = eng.tokenizer.encode(prompt_text, return_tensors="pt").to(eng.device)

        # ── 2. 构造采样参数 ──
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
        )

        # 计算 prompt token 数（在 tokenizer encode 时已经知道）
        prompt_len = prompt_ids.shape[-1]

        # ── 3. 提交请求到引擎 ──
        request_id, out_q = eng.submit(
            prompt_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            eos_token_id=eng.tokenizer.eos_token_id,
            sampling_params=sampling_params,
        )

        # ── 4. 流式 or 非流式 ──
        if request.stream:
            return StreamingResponse(
                _stream_response(request_id, out_q, eng, request.model, prompt_len),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # nginx 不缓冲 SSE
                },
            )
        else:
            return await _complete_response(request_id, out_q, eng, request.model, prompt_len)

    return application


# ── 思维链提取 ──

# Qwen3 thinking 标签：<think>...</think>
_THINK_START = "<think>"
_THINK_END = "</think>"


def _split_thinking(text: str) -> tuple[str, str | None]:
    """从生成文本中提取思维链内容。

    Qwen3 在 thinking 模式下输出格式：
      <think>推理过程...</think>回答内容

    Args:
        text: 完整的生成文本

    Returns:
        (content, reasoning_content) 元组。
        reasoning_content 为 None 表示没有思维链。
    """
    if _THINK_START not in text:
        return text, None

    # 找到最后一个完整的 <think>...</think> 块
    # （可能有多个 think 块，取最后一段作为 reasoning）
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = list(think_pattern.finditer(text))

    if not matches:
        # 有 </think> 但没有闭合的 </think>，全部内容视为 reasoning
        if _THINK_START in text:
            start_idx = text.index(_THINK_START) + len(_THINK_START)
            return "", text[start_idx:]
        return text, None

    # 提取最后一个 think 块作为 reasoning
    last_match = matches[-1]
    reasoning = last_match.group(1).strip()
    # 去掉所有 think 块后的剩余文本作为 content
    content = text[: last_match.start()] + text[last_match.end() :]
    content = content.strip()

    return content if content else "", reasoning


# ── 流式思维链解析器 ──


class _StreamingThinkParser:
    """流式 <think>/</think> 标签解析器（状态机）。

    逐 token 接收 decode 后的文本片段，根据当前状态将文本分流到
    reasoning_content（<think> 标签内）或 content（标签外）。

    状态转移：
      idle → 看到 <think> → in_think
      in_think → 看到 </think> → after_think
      after_think → 终态，全部输出为 content

    标签边界处理：
      - 标签可能被 tokenizer 拆成多个 token（如 "<thi" + "nk>"）
      - 通过 buffer 缓存尾部，避免误判
      - 每次 feed 时保留尾部 len(tag)-1 个字符，等下一个 token 确认

    使用方式：
        parser = _StreamingThinkParser()
        for text in token_stream:
            for field, delta in parser.feed(text):
                yield field, delta   # ("reasoning_content", "...") 或 ("content", "...")
        for field, delta in parser.flush():   # 流结束时 flush 剩余
            yield field, delta
    """

    def __init__(self) -> None:
        self._state = "idle"  # idle / in_think / after_think
        self._buffer = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        """接收新的文本片段，返回 (field, delta) 列表。

        field 取值：
          - "reasoning_content"：<think> 标签内的文本
          - "content"：标签外的文本

        返回列表可能为空（文本被 buffer 缓存等待标签闭合确认）。
        """
        self._buffer += text
        results: list[tuple[str, str]] = []

        while self._buffer:
            if self._state == "idle":
                idx = self._buffer.find(_THINK_START)
                if idx != -1:
                    # 标签前的文本作为 content（如果有的话）
                    if idx > 0:
                        results.append(("content", self._buffer[:idx]))
                    # 跳过 <think> 标签本身（不推送给客户端）
                    self._buffer = self._buffer[idx + len(_THINK_START) :]
                    self._state = "in_think"
                else:
                    # 没找到 <think>，保留尾部可能是不完整的标签前缀
                    # 例如 buffer 以 "<thi" 结尾，下一个 token 可能是 "nk>"
                    hold = len(_THINK_START) - 1  # 保留 6 个字符
                    safe = len(self._buffer) - hold
                    if safe > 0:
                        results.append(("content", self._buffer[:safe]))
                        self._buffer = self._buffer[safe:]
                    break  # 等更多文本

            elif self._state == "in_think":
                idx = self._buffer.find(_THINK_END)
                if idx != -1:
                    # </think> 前的文本作为 reasoning
                    if idx > 0:
                        results.append(("reasoning_content", self._buffer[:idx]))
                    # 跳过 </think> 标签本身
                    self._buffer = self._buffer[idx + len(_THINK_END) :]
                    self._state = "after_think"
                else:
                    # 没找到 </think>，保留尾部可能是不完整的标签前缀
                    hold = len(_THINK_END) - 1  # 保留 7 个字符
                    safe = len(self._buffer) - hold
                    if safe > 0:
                        results.append(("reasoning_content", self._buffer[:safe]))
                        self._buffer = self._buffer[safe:]
                    break

            elif self._state == "after_think":
                # <think> 已经结束，全部作为 content
                results.append(("content", self._buffer))
                self._buffer = ""
                break

        return results

    def flush(self) -> list[tuple[str, str]]:
        """流结束时 flush 剩余 buffer。

        - 未闭合的 <think> 内容全部作为 reasoning_content
        - idle 状态下剩余的文本作为 content
        """
        if not self._buffer:
            return []
        results: list[tuple[str, str]] = []
        if self._state == "in_think":
            results.append(("reasoning_content", self._buffer))
        elif self._state == "idle":
            results.append(("content", self._buffer))
        # after_think: buffer 应该已空（上面 while 已处理完）
        self._buffer = ""
        return results


# ── 非流式响应 ──


async def _complete_response(
    request_id: str,
    out_q: "queue.Queue",
    engine: AsyncEngine,
    model_name: str,
    prompt_len: int,
) -> JSONResponse:
    """等待全部 token 生成完毕，返回完整的 ChatCompletionResponse。

    工作原理：
      - out_q 是 thread-safe 的 queue.Queue（由后台引擎线程写入）
      - 在 asyncio 中不能直接 await queue.get()，所以用 get(timeout=0.1)
        + asyncio.sleep(0.001) 轮询，避免阻塞 event loop
      - 收到 _Sentinel 时停止，否则持续读取 token_id 并累积

    返回的 JSON 结构对齐 OpenAI /v1/chat/completions 非流式响应格式。
    """
    token_ids: list[int] = []
    finish_reason = "stop"

    while True:
        try:
            item = out_q.get(timeout=0.1)
        except queue.Empty:
            await asyncio.sleep(0.001)
            continue

        if isinstance(item, _Sentinel):
            finish_reason = item.finish_reason
            break
        token_ids.append(item)

    # decode 完整文本
    full_text = engine.tokenizer.decode(token_ids, skip_special_tokens=True)
    content, reasoning = _split_thinking(full_text)

    # 计算 usage
    completion_tokens = len(token_ids)
    usage = UsageInfo(
        prompt_tokens=prompt_len,
        completion_tokens=completion_tokens,
        total_tokens=prompt_len + completion_tokens,
    )

    response = ChatCompletionResponse(
        id=f"chatcmpl-{request_id[:8]}",
        model=model_name,
        choices=[
            ResponseChoice(
                index=0,
                message=ResponseMessage(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )

    # 清理 output queue
    engine.cleanup_request(request_id)

    return JSONResponse(content=response.model_dump())


# ── 流式响应（SSE）──


async def _stream_response(
    request_id: str,
    out_q: "queue.Queue",
    engine: AsyncEngine,
    model_name: str,
    prompt_len: int,
):
    """SSE 流式生成器：逐 token 推送 ChatCompletionStreamChunk。

    SSE（Server-Sent Events）协议格式：
      - 每个事件：data: {json}\n\n
      - 结束信号：data: [DONE]\n\n
      - 客户端（如 OpenAI SDK）通过 EventSource 或逐行读取解析

    chunk 序列：
      1. 首 chunk：delta={role: "assistant", content: ""}
         → 告知客户端这是 assistant 角色的响应
      2. 中间 chunks：delta={content: "xxx"}
         → 每产生一个 token 就 decode 并推送
      3. 末 chunk：delta={}, finish_reason="stop"/"length"
         → 告知客户端生成结束的原因
      4. [DONE]：SSE 流终止信号

    token 解码策略：
      - 每收到 1 个 token_id 就立即 decode（低延迟优先）
      - 使用 skip_special_tokens=True 过滤 BOS/EOS/PAD 等特殊 token
      - 多字节字符（如中文）的 UTF-8 编码可能被 tokenizer 拆成多个 token，
        单个 token decode 可能产生空字符串（被 if text: 过滤掉）
    """
    chunk_id = f"chatcmpl-{request_id[:8]}"
    created = int(time.time())

    # 首 chunk：发送 role
    first_chunk = ChatCompletionStreamChunk(
        id=chunk_id,
        created=created,
        model=model_name,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # ── 逐 token 流式推送 ──
    # 累积 token ids，定期 decode 避免多字节字符截断
    # （当前策略：每 1 个 token 立即 decode，优先低延迟）
    pending_token_ids: list[int] = []
    finish_reason = "stop"
    # 流式思维链解析器：检测 <think>/</think> 标签，分流到 reasoning_content / content
    think_parser = _StreamingThinkParser()

    while True:
        try:
            item = out_q.get(timeout=0.05)
        except queue.Empty:
            await asyncio.sleep(0.001)
            continue

        if isinstance(item, _Sentinel):
            finish_reason = item.finish_reason
            break

        pending_token_ids.append(item)

        # 每 1 个 token decode 一次（简单策略，保证低延迟）
        if pending_token_ids:
            text = engine.tokenizer.decode(pending_token_ids, skip_special_tokens=True)
            pending_token_ids.clear()

            if text:
                # 通过状态机分流到 reasoning_content 或 content
                for field, delta in think_parser.feed(text):
                    if field == "reasoning_content":
                        delta_msg = DeltaMessage(reasoning_content=delta)
                    else:
                        delta_msg = DeltaMessage(content=delta)
                    chunk = ChatCompletionStreamChunk(
                        id=chunk_id,
                        created=created,
                        model=model_name,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=delta_msg,
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

    # ── flush 思维链解析器剩余 buffer ──
    # 处理未闭合标签或尾部缓存的文本
    for field, delta in think_parser.flush():
        if field == "reasoning_content":
            delta_msg = DeltaMessage(reasoning_content=delta)
        else:
            delta_msg = DeltaMessage(content=delta)
        chunk = ChatCompletionStreamChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                StreamChoice(
                    index=0,
                    delta=delta_msg,
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # ── 结束 chunk：发送 finish_reason ──
    end_chunk = ChatCompletionStreamChunk(
        id=chunk_id,
        created=created,
        model=model_name,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=finish_reason,
            )
        ],
    )
    yield f"data: {end_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

    # 清理 output queue
    engine.cleanup_request(request_id)
