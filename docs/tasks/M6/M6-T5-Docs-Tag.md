# M6-T5 测试 + Docs + Tag

## 元信息
- **任务 ID**: T5
- **里程碑**: M6
- **状态**: ⬜ pending
- **前置**: T4
- **估时**: 4h

## 目标

**要解决什么问题**：
M6 核心功能完成后，需要收口测试、文档和里程碑标记。

**做完是什么效果**：
- M1-M5 全量 314 tests + M6 新增 tests 全部通过
- `docs/knowledge/m6-api-sse.md` 完成
- annotated tag `m6/api-sse` 创建
- 文章草稿完成

**不做什么**（边界）：
- 不做性能 benchmark（M6 不追求性能）
- 不做 GPU benchmark

## 产出文件

### 测试收口

| 测试文件 | 数量 | 覆盖 |
|---------|------|------|
| `tests/unit/test_sampling.py` | ~12 | SamplingParams + SamplingProcessor |
| `tests/unit/test_async_engine.py` | ~8 | AsyncEngine submit/stream/shutdown |
| `tests/unit/test_openai_api.py` | ~8 | schemas + endpoint 格式兼容 |
| `tests/e2e/test_sse_stream.py` | ~6 | SSE 流式 + reasoning_content |
| `tests/e2e/test_serve_smoke.py` | ~4 | 服务启动 + curl 端到端 |

### E2E smoke test

```python
# tests/e2e/test_serve_smoke.py

@pytest.mark.local_model
async def test_serve_smoke():
    """启动服务 → curl 非流式 → 验证格式 → curl 流式 → 验证 SSE。"""
    # 启动 AsyncEngine + FastAPI（用 httpx.AsyncClient）
    # POST /v1/chat/completions（非流式）
    # 验证 response 格式
    # POST /v1/chat/completions（流式）
    # 验证 SSE chunk 格式
    # 验证 data: [DONE]
```

### 文档

| 文件 | 内容 |
|------|------|
| `docs/knowledge/m6-api-sse.md` | M6 知识总结 |
| `docs/plan/M6.md` | 状态更新为 ✅ |
| `docs/plan/PROGRESS.md` | M6 ✅ + tag |
| `README.md` | M6 ✅ + Quick Start 更新 |
| `docs/plan/PLAN.md` | M6 状态更新 |

### 知识文档大纲

```markdown
# M6 知识总结：从 Python 函数到 ChatGPT 式流式服务

## 1. vLLM V1 架构 vs inferlite M6
  - 多进程 vs 单进程
  - AsyncLLM + EngineCore vs AsyncEngine
  - IPC vs asyncio.Queue

## 2. 异步引擎设计
  - 常驻循环 vs 一次性执行
  - asyncio.to_thread 桥接同步 forward
  - Queue 驱动流式输出

## 3. SSE 协议
  - data: {json}\n\n 格式
  - [DONE] 结束信号
  - 与 WebSocket 的取舍

## 4. OpenAI API 兼容
  - Chat Completions 最小格式
  - reasoning_content 扩展字段
  - 为什么不是完整 spec

## 5. Sampling 策略
  - temperature / top-k / top-p 的数学原理
  - repetition_penalty 实现
  - seed 可复现性

## 6. 与 vLLM V1 的简化对照表
## 7. 已知局限性
```

### Tag

```bash
git tag -a m6/api-sse -m "M6: API + SSE 服务化

- SamplingParams + SamplingProcessor (temperature/top-k/top-p/penalty)
- AsyncEngine 常驻异步引擎 (vLLM V1 AsyncLLM 简化版)
- OpenAI Chat Completions 兼容 (非流式 + SSE 流式)
- reasoning_content 映射 (Qwen3 thinking 标签)
- inferlite serve CLI 命令
- 极简 Chat UI (单 HTML 文件)
- N tests 全绿"
```

## 全量回归

| 范围 | 预期 |
|------|------|
| M1-M5 已有测试 | 314/314 通过 |
| M6 新增测试 | ~38/38 通过 |
| 总计 | ~352/352 通过 |

## DoD
- [ ] 全量测试 ~352/352 全绿
- [ ] `docs/knowledge/m6-api-sse.md` 完成
- [ ] M6.md 状态更新为 ✅
- [ ] PROGRESS.md 更新 M6 ✅
- [ ] README.md 更新 M6 ✅
- [ ] PLAN.md M6 状态更新
- [ ] annotated tag `m6/api-sse` 创建
- [ ] 文章草稿：《从 Python 函数到 ChatGPT 式流式服务》

## 坑（按概率排序）
1. 全量回归可能出现新的跨模块问题
2. knowledge 文档写太长（控制在 500 行以内）
3. tag message 遗漏关键信息
