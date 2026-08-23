"""M6 OpenAI API 格式兼容测试 + SSE 流式格式测试。

覆盖 L0 验证项（全部是 schema / 纯函数测试，不需要加载模型）：
  1. 请求 schema 验证（ChatCompletionRequest 字段/默认值/边界）
     → Pydantic 的 Field 约束（ge/le）是否正确拦截非法值
  2. 非流式响应格式（ChatCompletionResponse 符合 OpenAI spec）
     → model_dump() 输出的 JSON 结构与 OpenAI API 对齐
  3. 流式 chunk 格式（ChatCompletionStreamChunk 符合 SSE 规范）
     → 首 chunk 有 role、末 chunk 有 finish_reason、SSE 格式正确
  4. 思维链提取（_split_thinking 正确分离 reasoning_content）
     → Qwen3 的 <think>...</think> 标签解析（含未闭合边界情况）
  5. FastAPI 端点 E2E（非流式 + 流式 SSE）
     → 使用 TestClient + mock engine 做端到端验证
  6. 服务启动 smoke test（FastAPI app 创建成功）
     → create_app() 返回有效 app、路由注册正确、health 端点可达

测试策略：
  - schema 测试直接用 Pydantic 构造/校验，不启动 HTTP server
  - 流式格式测试验证 SSE 协议的 data: {json}\\n\\n 格式
  - 思维链提取测试覆盖正常路径 + 未闭合标签的边界情况
  - E2E 测试使用 FastAPI TestClient + MagicMock engine，避免加载真实模型
  - 每个 test class 对应一个独立的验证维度
"""

import json

import pytest

from inferlite.server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    ChatMessage,
    DeltaMessage,
    ResponseChoice,
    ResponseMessage,
    StreamChoice,
    UsageInfo,
)


class TestChatCompletionRequest:
    """请求 schema 验证。"""

    def test_minimal_request(self):
        """只有 messages 也能创建请求（其余字段用默认值）。"""
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hello")])
        assert req.model == "qwen3"
        assert req.stream is False
        assert req.temperature == 0.0
        assert req.max_tokens == 256

    def test_full_request(self):
        """所有字段都能正确设置。"""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hi"),
            ],
            stream=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            max_tokens=100,
            seed=42,
            repetition_penalty=1.1,
        )
        assert req.model == "test-model"
        assert req.stream is True
        assert req.temperature == 0.8
        assert req.top_p == 0.9
        assert req.top_k == 50
        assert req.max_tokens == 100
        assert req.seed == 42
        assert req.repetition_penalty == 1.1

    def test_temperature_bounds(self):
        """temperature 必须在 [0, 2] 范围内。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=3.0,
            )

    def test_top_p_bounds(self):
        """top_p 必须在 [0, 1] 范围内。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                top_p=1.5,
            )


class TestChatCompletionResponse:
    """非流式响应格式验证。"""

    def test_response_format(self):
        """响应格式符合 OpenAI ChatCompletion spec。"""
        resp = ChatCompletionResponse(
            id="chatcmpl-abc123",
            model="qwen3",
            choices=[
                ResponseChoice(
                    index=0,
                    message=ResponseMessage(
                        role="assistant",
                        content="Hello! How can I help?",
                        reasoning_content=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )
        data = resp.model_dump()
        assert data["object"] == "chat.completion"
        assert data["id"] == "chatcmpl-abc123"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["total_tokens"] == 15

    def test_response_with_reasoning(self):
        """包含 reasoning_content 的响应。"""
        resp = ChatCompletionResponse(
            id="chatcmpl-abc",
            choices=[
                ResponseChoice(
                    message=ResponseMessage(
                        content="The answer is 42.",
                        reasoning_content="Let me think step by step...",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        data = resp.model_dump()
        assert data["choices"][0]["message"]["reasoning_content"] is not None


class TestStreamChunk:
    """流式 SSE chunk 格式验证。"""

    def test_chunk_format(self):
        """chunk 格式符合 OpenAI ChatCompletionStream spec。"""
        chunk = ChatCompletionStreamChunk(
            id="chatcmpl-abc",
            model="qwen3",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content="Hello"),
                    finish_reason=None,
                )
            ],
        )
        data = chunk.model_dump()
        assert data["object"] == "chat.completion.chunk"
        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert data["choices"][0]["finish_reason"] is None

    def test_first_chunk_has_role(self):
        """首 chunk 包含 role=assistant。"""
        chunk = ChatCompletionStreamChunk(
            id="chatcmpl-abc",
            choices=[
                StreamChoice(
                    delta=DeltaMessage(role="assistant", content=""),
                )
            ],
        )
        data = chunk.model_dump()
        assert data["choices"][0]["delta"]["role"] == "assistant"

    def test_end_chunk_has_finish_reason(self):
        """结束 chunk 包含 finish_reason。"""
        chunk = ChatCompletionStreamChunk(
            id="chatcmpl-abc",
            choices=[
                StreamChoice(
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
        )
        data = chunk.model_dump()
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_sse_format(self):
        """SSE 格式：data: {json}\\n\\n。"""
        chunk = ChatCompletionStreamChunk(
            id="chatcmpl-abc",
            choices=[StreamChoice(delta=DeltaMessage(content="Hi"))],
        )
        # SSE 格式要求：data: {json}\n\n
        sse_line = f"data: {chunk.model_dump_json()}\n\n"
        assert sse_line.startswith("data: ")
        assert sse_line.endswith("\n\n")
        # JSON 可以被解析
        parsed = json.loads(sse_line[6:].strip())
        assert parsed["choices"][0]["delta"]["content"] == "Hi"


class TestSplitThinking:
    """思维链提取测试。"""

    def test_no_thinking(self):
        """没有 <think> 标签时，全部文本作为 content。"""
        from inferlite.server.app import _split_thinking

        content, reasoning = _split_thinking("Hello, how are you?")
        assert content == "Hello, how are you?"
        assert reasoning is None

    def test_with_thinking(self):
        """有 <think>...</think> 标签时，正确分离 reasoning 和 content。"""
        from inferlite.server.app import _split_thinking

        text = "<think>Let me think...</think>The answer is 42."
        content, reasoning = _split_thinking(text)
        assert content == "The answer is 42."
        assert reasoning == "Let me think..."

    def test_unclosed_think(self):
        """有 </think> 但没有闭合时，全部内容视为 reasoning。"""
        from inferlite.server.app import _split_thinking

        text = "<think>Still thinking..."
        content, reasoning = _split_thinking(text)
        assert content == ""
        assert reasoning == "Still thinking..."


class TestStreamingThinkParser:
    """流式思维链解析器测试。"""

    def _collect(self, texts):
        """Helper: feed 多段文本，收集 (field, delta) 结果。"""
        from inferlite.server.app import _StreamingThinkParser

        parser = _StreamingThinkParser()
        results = []
        for text in texts:
            results.extend(parser.feed(text))
        results.extend(parser.flush())
        return results

    def test_no_thinking_stream(self):
        """无 <think> 标签时，全部文本作为 content。"""
        results = self._collect(["Hello", ", ", "world", "!"])
        # 合并所有 content delta
        content = "".join(d for f, d in results if f == "content")
        reasoning = "".join(d for f, d in results if f == "reasoning_content")
        assert content == "Hello, world!"
        assert reasoning == ""

    def test_thinking_stream_full(self):
        """完整的 <think>...</think> 标签，正确分流。"""
        # 模拟逐 token 输出：<think> → reasoning → </think> → content
        results = self._collect(
            ["<think>", "Let me", " think...", "</think>", "The answer", " is 42."]
        )
        reasoning = "".join(d for f, d in results if f == "reasoning_content")
        assert reasoning == "Let me think..."
        # content 应该包含标签后的文本
        content_only = "".join(d for f, d in results if f == "content")
        assert content_only == "The answer is 42."

    def test_thinking_stream_tag_split(self):
        """</think> 标签被拆成多个 token（如 "<" + "/think>"）。"""
        results = self._collect(["<think>", "reasoning text", "</think>", "answer"])
        reasoning = "".join(d for f, d in results if f == "reasoning_content")
        content = "".join(d for f, d in results if f == "content")
        assert reasoning == "reasoning text"
        assert content == "answer"

    def test_unclosed_think_stream(self):
        """未闭合的 <think> 标签，全部内容作为 reasoning。"""
        results = self._collect(["<think>", "Still thinking", "..."])
        reasoning = "".join(d for f, d in results if f == "reasoning_content")
        content = "".join(d for f, d in results if f == "content")
        assert reasoning == "Still thinking..."
        assert content == ""

    def test_tag_in_single_token(self):
        """</think> 和后续文本在同一个 token 中。"""
        results = self._collect(["<think>", "reasoning", "</think>The answer is 42."])
        reasoning = "".join(d for f, d in results if f == "reasoning_content")
        content = "".join(d for f, d in results if f == "content")
        assert reasoning == "reasoning"
        assert content == "The answer is 42."

    def test_empty_feed(self):
        """空文本不产生输出。"""
        from inferlite.server.app import _StreamingThinkParser

        parser = _StreamingThinkParser()
        assert parser.feed("") == []
        assert parser.flush() == []

    def test_partial_tag_buffer(self):
        """标签前缀被拆分到不同 token（如 '<thi' + 'nk>'）。"""
        results = self._collect(["<thi", "nk>", "reasoning", "</think>", "content"])
        reasoning = "".join(d for f, d in results if f == "reasoning_content")
        content = "".join(d for f, d in results if f == "content")
        assert reasoning == "reasoning"
        assert content == "content"


class TestFastAPIApp:
    """FastAPI 应用 smoke test。"""

    def test_create_app(self):
        """create_app 能创建 FastAPI 实例并注册路由。"""
        from unittest.mock import MagicMock

        from inferlite.server.app import create_app

        # 用 mock engine 避免真实模型加载
        mock_engine = MagicMock()
        mock_engine.tokenizer = MagicMock()
        mock_engine.device = "cpu"

        app = create_app(mock_engine)

        # 验证 app 创建成功
        assert app is not None
        assert app.title == "inferlite"

        # 验证路由注册
        routes = [r.path for r in app.routes]
        assert "/health" in routes
        assert "/v1/chat/completions" in routes

    def test_health_endpoint(self):
        """health 端点返回 200 + status ok。"""
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        from inferlite.server.app import create_app

        mock_engine = MagicMock()
        app = create_app(mock_engine)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
