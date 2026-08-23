"""M6 Server API 测试。

使用 FastAPI TestClient（同步 HTTP 测试），覆盖：
- /v1/models 端点
- /v1/chat/completions 非流式
- /v1/chat/completions 流式 SSE
- 请求参数校验
- seed 可复现

注意：这些测试需要真实模型，标记为 @pytest.mark.local_model。
"""

import json

import pytest

from inferlite.server.app import _manager, app


@pytest.fixture(scope="module")
def loaded_app():
    """加载模型并返回 TestClient。"""
    from pathlib import Path

    from fastapi.testclient import TestClient

    model_dir = str(Path("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B").expanduser())
    if _manager.model is None:
        _manager.load(model_dir, device="cpu", dtype="fp32", max_seq_len=128)
    return TestClient(app)


@pytest.mark.local_model
class TestModelsEndpoint:
    """GET /v1/models 测试。"""

    def test_list_models(self, loaded_app):
        resp = loaded_app.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["object"] == "model"
        assert "id" in body["data"][0]


@pytest.mark.local_model
class TestChatCompletions:
    """POST /v1/chat/completions 非流式测试。"""

    def test_basic_completion(self, loaded_app):
        resp = loaded_app.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert len(body["choices"][0]["message"]["content"]) > 0
        assert body["choices"][0]["finish_reason"] in ("stop", "length")

    def test_usage_info(self, loaded_app):
        resp = loaded_app.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4,
            },
        )
        body = resp.json()
        assert body["usage"]["prompt_tokens"] > 0
        assert body["usage"]["completion_tokens"] > 0
        assert body["usage"]["total_tokens"] == (
            body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
        )

    def test_seed_reproducible(self, loaded_app):
        """相同 seed 产生相同输出。"""
        request = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": "Count: 1, 2,"}],
            "max_tokens": 4,
            "temperature": 0.7,
            "seed": 42,
        }
        resp1 = loaded_app.post("/v1/chat/completions", json=request)
        resp2 = loaded_app.post("/v1/chat/completions", json=request)
        text1 = resp1.json()["choices"][0]["message"]["content"]
        text2 = resp2.json()["choices"][0]["message"]["content"]
        assert text1 == text2

    def test_max_tokens_length_finish(self, loaded_app):
        """达到 max_tokens 时 finish_reason = length。"""
        resp = loaded_app.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Tell me a long story"}],
                "max_tokens": 2,
                "temperature": 0.0,
            },
        )
        body = resp.json()
        assert body["choices"][0]["finish_reason"] == "length"


@pytest.mark.local_model
class TestStreamResponse:
    """流式 SSE 响应测试。"""

    def test_stream_format(self, loaded_app):
        """SSE 流式响应的格式正确性。"""
        resp = loaded_app.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4,
                "stream": True,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        lines = resp.text.strip().split("\n")
        # 至少有: role chunk + token chunks + [DONE]
        assert len(lines) >= 3

        # 首 chunk 应该包含 role
        first_data = lines[0]
        assert first_data.startswith("data: ")
        first_json = json.loads(first_data[6:])
        assert first_json["object"] == "chat.completion.chunk"
        assert first_json["choices"][0]["delta"].get("role") == "assistant"

        # 最后应该是 [DONE]
        assert lines[-1].strip() == "data: [DONE]"

    def test_stream_tokens(self, loaded_app):
        """流式响应包含实际 token 内容。"""
        resp = loaded_app.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Count to 3"}],
                "max_tokens": 8,
                "stream": True,
                "temperature": 0.0,
            },
        )
        lines = resp.text.strip().split("\n")
        # 收集所有 content delta
        content_parts = []
        for line in lines:
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"]
                if delta.get("content"):
                    content_parts.append(delta["content"])
        assert len(content_parts) > 0
        full_text = "".join(content_parts)
        assert len(full_text) > 0


class TestSchemas:
    """Schema 验证测试（不需要模型）。"""

    def test_request_defaults(self):
        from inferlite.server.schemas import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test",
            messages=[],
        )
        assert req.max_tokens == 128
        assert req.temperature == 0.0
        assert req.stream is False

    def test_request_validation(self):
        from pydantic import ValidationError

        from inferlite.server.schemas import ChatCompletionRequest

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[],
                temperature=5.0,  # > 2.0 不允许
            )

    def test_chunk_format(self):
        from inferlite.server.schemas import (
            ChatCompletionChunk,
            ChatCompletionChunkChoice,
            DeltaMessage,
        )

        chunk = ChatCompletionChunk(
            id="test-123",
            created=1234567890,
            model="qwen3",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(content="hello"),
                )
            ],
        )
        d = chunk.model_dump()
        assert d["object"] == "chat.completion.chunk"
        assert d["choices"][0]["delta"]["content"] == "hello"
