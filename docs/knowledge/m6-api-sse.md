# inferlite M6：API + SSE 服务化

> **状态**：✅ 已完成（2026-08-23）

## 一句话本质

把内存中的推理引擎包装成 HTTP 服务，让任何 OpenAI 兼容客户端（curl / ChatBox / Open WebUI）都能通过标准 API 使用 inferlite。

## 1. 目标与范围

**目标**：`inferlite serve` 一条命令启动服务；兼容 OpenAI Chat Completions 基本格式；支持非流式和 SSE 流式两种响应。

**范围**：
- ✅ POST `/v1/chat/completions`（非流式 + 流式）
- ✅ GET `/v1/models`
- ✅ SamplingParams（temperature / top_k / top_p / repetition_penalty）
- ✅ Seed 可复现
- ❌ Function calling / tool use（不在教学范围）
- ❌ Logprobs / n-choices（不在教学范围）
- ❌ Embeddings API（不在 LLM 推理引擎范围）

## 2. 架构

```
Client (curl / ChatBox / Open WebUI)
  │
  │  HTTP POST /v1/chat/completions
  ▼
FastAPI App (inferlite/server/app.py)
  │
  │  ChatCompletionRequest → messages → apply_chat_template → input_ids
  │
  ▼
ModelManager (singleton)
  │
  │  asyncio.Lock → 序列化推理（MPS 不支持并发 forward）
  │  asyncio.to_thread → CPU-bound 推理不阻塞 event loop
  │
  ▼
generate_stream() (M2 路径)
  │
  │  KVCache.from_config → SingleCacheAdapter → prefill + decode loop
  │  SamplingProcessor → temperature / top_k / top_p / repetition_penalty
  │
  ▼
yield (token, is_done) → SSE chunk → StreamingResponse
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 推理路径 | M2 (single cache) | 最简单、无 batch 维开销；API 场景每请求独立 |
| 并发模型 | asyncio.Lock 序列化 | MPS 不支持并发 forward；CUDA 可后续升级为 batch |
| 阻塞处理 | asyncio.to_thread | CPU-bound 推理不阻塞 event loop |
| 采样器 | SamplingProcessor | 替代 GreedySampler，支持 temperature/top_k/top_p/penalty |
| SSE 格式 | StreamingResponse | FastAPI 原生支持，不需要额外依赖 |

### 数据流

```
request.json → ChatCompletionRequest
  → messages → tokenizer.apply_chat_template → prompt_text
  → tokenizer.encode → input_ids [1, T]
  → KVCache.from_config → kv_cache
  → generate_stream(engine, input_ids, max_tokens, eos, kv_cache, processor)
      → prefill: model(input_ids) → logits → sample → first token
      → decode loop: model(token) → logits → sample → yield token
  → tokenizer.decode → output_text
  → ChatCompletionResponse / SSE chunks
```

## 3. 核心代码

### 3.1 SamplingParams + SamplingProcessor

```python
@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0   # 0 = greedy
    top_k: int = 0             # 0 = no filter
    top_p: float = 1.0         # 1.0 = no filter
    repetition_penalty: float = 1.0  # 1.0 = no penalty
```

处理流水线：`repetition_penalty → temperature → top_k → top_p → multinomial`

- `repetition_penalty`（CTRL 论文）：对已出现 token，正 logit 除以 penalty，负 logit 乘以 penalty
- `top_k`：只保留 logits 最高的 k 个 token，其余设为 -inf
- `top_p`（nucleus）：排序后累积概率超过 p 的 token 设为 -inf

### 3.2 generate_stream

与 `engine.generate()` 的区别：
- `generate()` 返回完整 `output_ids [B, T+n]`
- `generate_stream()` 逐个 `yield (token_id, is_done)`，支持流式输出

走 M2 路径：`SingleCacheAdapter.bind_kv_cache → prefill → decode loop`。

### 3.3 SSE 格式

对齐 OpenAI streaming spec：

```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}

data: [DONE]
```

## 4. 文件结构

```
inferlite/server/
  __init__.py          # 空
  schemas.py           # Pydantic models (Request / Response / Chunk)
  app.py               # FastAPI app + generate_stream + ModelManager
  cli.py               # inferlite serve CLI
inferlite/sampler/
  sampling.py          # SamplingParams + SamplingProcessor
pyproject.toml         # 新增 inferlite-serve console script
```

## 5. 与 vLLM 的对照

| 特性 | inferlite M6 | vLLM |
|------|-------------|------|
| API 格式 | OpenAI 兼容 | OpenAI 兼容 |
| 并发模型 | asyncio.Lock 串行 | 真正的 continuous batching |
| 推理路径 | M2 (单请求) | PagedAttention + continuous batching |
| 采样 | temperature/top_k/top_p/penalty | 同 + frequency_penalty/presence_penalty/min_tokens |
| 流式 | SSE StreamingResponse | 同 |
| 框架 | FastAPI + uvicorn | FastAPI + uvicorn |

**核心差异**：vLLM 的服务层（`AsyncLLMEngine`）背后是真正的 continuous batching scheduler，多个请求共享一次 forward。inferlite M6 用 asyncio.Lock 序列化请求，每个请求独立走 M2 路径。这是教学简洁性的选择——真正的 batch serving 需要 M3/M4 路径的异步版本，复杂度显著增加。

## 6. 使用

```bash
# 启动服务
inferlite serve --model-dir ~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B \
    --device mps --dtype bf16 --port 8000

# 非流式请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'

# 流式请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# OpenAI Python 客户端兼容
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=16,
)
```

## 7. 测试

- `tests/unit/test_sampling.py`：20 tests（SamplingParams + temperature + top_k + top_p + repetition_penalty）
- `tests/unit/test_server.py`：10 tests（models endpoint + non-streaming + streaming SSE + schema validation + seed 可复现）
- 端到端 smoke test：curl 验证三个端点

## 8. 局限性

1. **请求串行**：asyncio.Lock 保证同一时刻只有一个请求在推理。高并发场景下延迟线性增长。
2. **无请求排队**：没有 waiting queue，新请求直接等待锁。
3. **无健康检查**：缺少 `/health` 端点（生产环境需要）。
4. **KV Cache 每次重建**：每个请求都 `KVCache.from_config` 新建，没有池化复用。
5. **无 timeout**：长请求不会超时中断。

这些是教学简洁性的选择，生产框架（vLLM/TGI/SGLang）都有对应的工程方案。
