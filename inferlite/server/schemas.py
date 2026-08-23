"""OpenAI Chat Completions API 兼容的 Pydantic 数据模型。

对齐 OpenAI /v1/chat/completions 的请求/响应格式，
使 inferlite 服务能被 OpenAI Python SDK、ChatBox、NextChat 等客户端直接调用。

请求示例（客户端发送）：
    {
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 256,
        "stream": true
    }

非流式响应示例（stream=false）：
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi there!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }

流式响应示例（stream=true，SSE）：
    data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":""}}]}
    data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"}}]}
    data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}
    data: [DONE]

支持的请求字段：
  - model: 模型名称（校验用）
  - messages: 对话消息列表（role + content）
  - stream: 是否启用 SSE 流式输出
  - temperature / top_p / top_k: 采样参数
  - max_tokens: 最大生成 token 数
  - stop: 停止词列表（M6 暂不实现，预留字段）
  - seed: 随机种子，用于可复现采样
  - repetition_penalty: 重复惩罚系数（inferlite 扩展，OpenAI 不提供）

支持的响应字段：
  - id: 请求唯一标识
  - object: "chat.completion" 或 "chat.completion.chunk"
  - created: Unix 时间戳
  - model: 模型名称
  - choices: 生成结果列表（每条包含 message/delta + finish_reason）
  - usage: token 用量统计（仅非流式响应）
"""

import time

from pydantic import BaseModel, Field

# ── 请求模型 ──


class ChatMessage(BaseModel):
    """单条对话消息。"""

    role: str = Field(description="消息角色：system / user / assistant")
    content: str | list = Field(description="消息内容（str 或 OpenAI 多模态 list 格式）")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible 请求体。

    字段对齐 OpenAI spec，不需要的字段用 model_config 忽略。

    字段映射到引擎层：
      temperature / top_k / top_p / repetition_penalty / seed
        → SamplingParams → SamplingProcessor（per-request 实例）
      max_tokens → RequestState.max_new_tokens
      messages → tokenizer.apply_chat_template() → prompt token ids

    Pydantic 校验：
      - temperature: [0, 2]，0 表示 greedy
      - top_p: [0, 1]
      - max_tokens: ≥ 1
      - repetition_penalty: ≥ 1.0
    """

    model: str = Field(default="qwen3", description="模型名称")
    messages: list[ChatMessage] = Field(description="对话消息列表")
    stream: bool = Field(default=False, description="是否启用 SSE 流式输出")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="采样温度，0=greedy")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="nucleus sampling 阈值")
    top_k: int = Field(default=-1, description="top-k 过滤，-1=不过滤")
    max_tokens: int = Field(default=256, ge=1, description="最大生成 token 数")
    stop: list[str] | None = Field(default=None, description="停止词列表（M6 预留）")
    seed: int | None = Field(default=None, description="随机种子，用于可复现采样")
    repetition_penalty: float = Field(default=1.0, ge=1.0, description="重复惩罚系数")


# ── 非流式响应模型 ──


class UsageInfo(BaseModel):
    """token 用量统计。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseMessage(BaseModel):
    """非流式响应中的完整消息。"""

    role: str = "assistant"
    content: str
    reasoning_content: str | None = Field(
        default=None, description="思维链内容（Qwen3 的 <think> 标签内容）"
    )


class ResponseChoice(BaseModel):
    """非流式响应中的单条生成结果。"""

    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = Field(default=None, description="停止原因：stop / length / null")


class ChatCompletionResponse(BaseModel):
    """非流式完整响应，对齐 OpenAI ChatCompletion 格式。

    与流式 ChatCompletionStreamChunk 的区别：
      - object 字段为 "chat.completion"（chunk 为 "chat.completion.chunk"）
      - choices 包含完整的 ResponseMessage（role + content + reasoning_content）
      - 包含 usage 统计（流式响应不含 usage）
    """

    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "qwen3"
    choices: list[ResponseChoice]
    usage: UsageInfo


# ── 流式响应模型（SSE chunk）──


class DeltaMessage(BaseModel):
    """流式 chunk 中的增量消息。"""

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None


class StreamChoice(BaseModel):
    """流式 chunk 中的单条结果。"""

    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionStreamChunk(BaseModel):
    """流式 SSE chunk，对齐 OpenAI ChatCompletionStream 格式。

    SSE 协议要求每个事件的格式：
        data: {json}\n\n        ← 每个 chunk 都是一行 "data: " + JSON + 两个换行
        data: [DONE]\n\n        ← 结束信号，客户端据此关闭连接

    一个完整的流式响应的 chunk 序列：
        1. 首 chunk：delta={role: "assistant", content: ""}，告知客户端角色
        2. 中间 chunks：delta={content: "xxx"}，逐 token 推送生成内容
        3. 末 chunk：delta={}, finish_reason="stop"/"length"，告知结束原因
        4. [DONE] 信号
    """

    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "qwen3"
    choices: list[StreamChoice]
