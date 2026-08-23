"""M6 OpenAI Chat Completions API 兼容的请求/响应 Pydantic 模型。

对齐 OpenAI API spec (https://platform.openai.com/docs/api-reference/chat)：
- ChatCompletionRequest: POST /v1/chat/completions 的请求体
- ChatCompletionResponse: 非流式响应
- ChatCompletionChunk: 流式 SSE 的每个 chunk

只实现核心字段，忽略 OpenAI API 中的高级功能（function calling、logprobs 等）。
"""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

# ── 请求 ──


class ChatMessage(BaseModel):
    """单条聊天消息。"""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions 请求体。

    对齐 OpenAI ChatCompletionCreateParams，只保留核心字段。
    """

    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    stop: list[str] | None = None
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    seed: int | None = None


# ── 非流式响应 ──


class ChatCompletionChoice(BaseModel):
    """非流式响应中的单个 choice。"""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", None] = None


class UsageInfo(BaseModel):
    """Token 用量统计。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """非流式 /v1/chat/completions 响应。"""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


# ── 流式 SSE 响应 ──


class DeltaMessage(BaseModel):
    """流式 chunk 中的增量消息。"""

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """流式 chunk 中的单个 choice。"""

    index: int
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", None] = None


class ChatCompletionChunk(BaseModel):
    """流式 SSE 的单个 chunk（data: {json}）。"""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
