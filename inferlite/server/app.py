"""M6 FastAPI 服务：OpenAI 兼容的 Chat Completions API。

提供两个核心端点：
- POST /v1/chat/completions: 非流式 + 流式 (SSE) 聊天补全
- GET  /v1/models: 模型列表（兼容 OpenAI 客户端）

架构：
- ModelManager: 单例，持有 model / tokenizer / engine / device / dtype
- generate_stream: 逐 token 生成的同步函数（在 asyncio.to_thread 中运行）
- asyncio.Lock: 序列化推理请求（MPS 不支持并发 forward）

SSE 格式对齐 OpenAI：
    data: {"id":"...","object":"chat.completion.chunk","choices":[...]}

    data: [DONE]
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from inferlite.cache.kv_cache import KVCache
from inferlite.cli import resolve_device_dtype
from inferlite.engine import EngineCore
from inferlite.engine.context import set_forward_context
from inferlite.model.weights import load_causal_lm_from_hf
from inferlite.sampler.sampling import SamplingParams, SamplingProcessor
from inferlite.server.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)

# ── Model Manager ──


class ModelManager:
    """全局模型管理器（单例）。

    在 app lifespan 中初始化，所有请求共享同一份模型权重。
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.engine = None
        self.device = "cpu"
        self.dtype = torch.float32
        self.model_name = ""
        self.max_seq_len = 2048
        self.lock = asyncio.Lock()  # 序列化推理请求

    def load(
        self, model_dir: str, device: str = "auto", dtype: str = "auto", max_seq_len: int = 2048
    ) -> None:
        """加载模型、tokenizer、构造 engine。"""
        model_dir = str(Path(model_dir).expanduser().resolve())
        self.device, self.dtype = resolve_device_dtype(device, dtype)
        self.max_seq_len = max_seq_len
        self.model_name = model_dir.rsplit("/", 1)[-1]

        self.model = load_causal_lm_from_hf(model_dir)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        )

        from inferlite.sampler import GreedySampler

        self.engine = EngineCore(self.model, GreedySampler())


_manager = ModelManager()


# ── Streaming Generate ──


def generate_stream(
    engine: EngineCore,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    kv_cache: KVCache,
    sampling_processor: SamplingProcessor,
):
    """M2 路径的流式 generate：yield 每个新生成的 token。

    与 engine.generate() 的区别：
    - generate() 返回完整 output_ids
    - generate_stream() 逐个 yield (token_id_tensor, is_finished)

    走 M2 prefill/decode 两阶段路径，每步 yield 一个 token。
    """
    from inferlite.cache.adapter import SingleCacheAdapter

    adapter = SingleCacheAdapter(kv_cache)
    adapter.bind_kv_cache(engine.model)
    kv_cache.reset()

    # ── Prefill ──
    T_p = input_ids.shape[1]
    position_ids = torch.arange(T_p, device=input_ids.device).unsqueeze(0)
    metadata = adapter.make_prefill_metadata(input_ids, position_ids)
    with set_forward_context(metadata):
        logits = engine.model(input_ids, positions=position_ids)
    adapter.cur_len = T_p
    kv_cache.cur_len = T_p

    # 采样首个 token
    next_token = sampling_processor(logits[:, -1, :], input_ids)
    is_eos = eos_token_id is not None and next_token.item() == eos_token_id
    yield next_token, (is_eos or max_new_tokens <= 1)

    if is_eos or max_new_tokens <= 1:
        return

    input_ids = torch.cat([input_ids, next_token], dim=1)
    num_generated = 1

    # ── Decode loop ──
    for _ in range(max_new_tokens - 1):
        pos = torch.tensor([[kv_cache.cur_len]], device=input_ids.device)
        metadata = adapter.make_decode_metadata(next_token, pos)
        with set_forward_context(metadata):
            logits = engine.model(next_token, positions=pos, logits_to_keep=1)

        next_token = sampling_processor(logits[:, -1, :], input_ids)
        num_generated += 1
        is_eos = eos_token_id is not None and next_token.item() == eos_token_id
        is_done = is_eos or num_generated >= max_new_tokens

        yield next_token, is_done

        if is_done:
            return

        input_ids = torch.cat([input_ids, next_token], dim=1)


# ── SSE Formatting ──


def _make_chunk(
    completion_id: str,
    model: str,
    created: int,
    delta: DeltaMessage,
    finish_reason: str | None = None,
) -> str:
    """构造一个 SSE chunk JSON 字符串。"""
    chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


# ── FastAPI App ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan：启动时加载模型，关闭时清理。"""
    # 模型在 create_app() 中已加载，这里只做启动日志
    print(f"✅ Model loaded: {_manager.model_name} on {_manager.device}/{_manager.dtype}")
    yield


app = FastAPI(title="inferlite", version="0.1.0", lifespan=lifespan)


@app.get("/v1/models")
async def list_models():
    """GET /v1/models — 返回可用模型列表。"""
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": _manager.model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """POST /v1/chat/completions — OpenAI 兼容的聊天补全。

    支持 stream=true（SSE 流式）和 stream=false（非流式）。
    """
    if _manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ── 构造 prompt ──
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt_text = _manager.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = _manager.tokenizer.encode(prompt_text, return_tensors="pt").to(_manager.device)
    prompt_tokens = input_ids.shape[1]

    if prompt_tokens >= _manager.max_seq_len:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long ({prompt_tokens} >= max_seq_len {_manager.max_seq_len})",
        )

    # ── 采样参数 ──
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )
    processor = SamplingProcessor(sampling_params)

    # ── Seed ──
    if request.seed is not None:
        torch.manual_seed(request.seed)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    eos_id = _manager.tokenizer.eos_token_id

    # ── 流式响应 ──
    if request.stream:
        return StreamingResponse(
            _stream_response(
                completion_id, created, request, input_ids, processor, eos_id, prompt_tokens
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── 非流式响应 ──
    async with _manager.lock:
        result = await asyncio.to_thread(
            _run_generate, input_ids, request.max_tokens, eos_id, processor
        )

    output_ids, completion_tokens = result
    output_text = _manager.tokenizer.decode(output_ids, skip_special_tokens=True)

    finish_reason = "stop" if completion_tokens < request.max_tokens else "length"

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output_text),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _run_generate(input_ids, max_new_tokens, eos_token_id, processor):
    """同步推理函数（在 asyncio.to_thread 中运行）。"""
    kv_cache = KVCache.from_config(
        _manager.model.config,
        batch_size=1,
        max_seq_len=_manager.max_seq_len,
        dtype=_manager.dtype,
        device=_manager.device,
    )
    token_ids = []
    for token, is_done in generate_stream(
        _manager.engine, input_ids, max_new_tokens, eos_token_id, kv_cache, processor
    ):
        token_ids.append(token.item())
        if is_done:
            break
    return token_ids, len(token_ids)


async def _stream_response(
    completion_id, created, request, input_ids, processor, eos_id, prompt_tokens
):
    """异步生成器：yield SSE chunks。"""
    # 首 chunk：发送 role
    yield _make_chunk(completion_id, request.model, created, DeltaMessage(role="assistant"))

    # 在锁内执行推理（序列化请求）
    async with _manager.lock:
        kv_cache = KVCache.from_config(
            _manager.model.config,
            batch_size=1,
            max_seq_len=_manager.max_seq_len,
            dtype=_manager.dtype,
            device=_manager.device,
        )

        completion_tokens = 0
        for token, is_done in await asyncio.to_thread(
            _collect_tokens, input_ids, request.max_tokens, eos_id, processor, kv_cache
        ):
            completion_tokens += 1
            token_text = _manager.tokenizer.decode([token.item()], skip_special_tokens=True)
            finish = (
                "stop" if is_done and token.item() == eos_id else ("length" if is_done else None)
            )
            yield _make_chunk(
                completion_id,
                request.model,
                created,
                DeltaMessage(content=token_text),
                finish_reason=finish,
            )

    # 终止标记
    yield "data: [DONE]\n\n"


def _collect_tokens(input_ids, max_new_tokens, eos_token_id, processor, kv_cache):
    """同步收集所有生成 token（在 asyncio.to_thread 中运行）。"""
    tokens = []
    for token, is_done in generate_stream(
        _manager.engine, input_ids, max_new_tokens, eos_token_id, kv_cache, processor
    ):
        tokens.append((token, is_done))
    return tokens


# ── App Factory ──


def create_app(
    model_dir: str, device: str = "auto", dtype: str = "auto", max_seq_len: int = 2048
) -> FastAPI:
    """创建并配置 FastAPI app，加载模型。

    Args:
        model_dir: 本地模型目录路径。
        device: 推理设备（auto/cpu/mps/cuda）。
        dtype: 推理精度（auto/bf16/fp16/fp32）。
        max_seq_len: KV Cache 最大序列长度。

    Returns:
        配置好的 FastAPI app 实例。
    """
    _manager.load(model_dir, device, dtype, max_seq_len)
    return app
