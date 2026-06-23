# M1-T12 Real Qwen3-0.6B Smoke Test

## 元信息
- **任务 ID**: T12
- **里程碑**: M1
- **状态**: ✅ done
- **前置**: T12-pre `logits_to_keep` 优化、T11 CLI/e2e
- **估时**: 2h

## 目标

加载本地真实 Qwen3-0.6B 权重，用 inferlite 进行贪心生成，验证前 N 个输出 token 与
`transformers.generate(do_sample=False)` 完全一致。

这是 M1 的最终验收：架构正确 + 权重加载正确 + 生成逻辑正确，三项合一。

## 产出文件

- `tests/integration/test_real_qwen3_smoke.py`
  — `@pytest.mark.local_model`，不加 `-m local_model` 时 CI 自动跳过，不下载模型

## 算法核心

```
本地模型路径（两个缓存位置任选一个可用的）：
  ~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B
  ~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B

对齐目标：
  inferlite.generate(prompt, max_new_tokens=N, do_sample=False)
  == transformers.pipeline/generate(prompt, max_new_tokens=N, do_sample=False)

  前 N=10 个 token ID 完全相同（int64 精确匹配）
```

生成对齐用的 prompt 采用 chat template 格式（T11 经验：裸 prompt 质量差）：

```python
messages = [{"role": "user", "content": "Hello! What is 1+1?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|--------|--------------|------|
| 1 | 前 10 个 token ID 完全一致 | `transformers.generate(do_sample=False)` | 精确匹配 |
| 2 | 输出文本可读（非乱码） | 目测 | — |

## 具体改动

**`tests/integration/test_real_qwen3_smoke.py`**（新文件）：

```python
import pytest

MODEL_PATH = "~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

@pytest.mark.local_model
def test_qwen3_generate_matches_transformers():
    """
    加载真实 Qwen3-0.6B，用 inferlite.generate 和 transformers.generate
    做贪心解码，验证前 10 个 token 完全一致。
    """
    ...
```

**`pyproject.toml`**（补充 marker 声明，已有的话跳过）：

```toml
[tool.pytest.ini_options]
markers = [
    "local_model: tests that require a locally downloaded model, skipped in CI",
]
```

## DoD

- [ ] `tests/integration/test_real_qwen3_smoke.py` 实现，本地 `pytest -m local_model` 全绿
- [ ] `pytest` (不加 `-m`) 跳过该测试（CI 安全）
- [ ] 前 10 个 token ID 与 transformers 精确匹配
- [ ] `docs/kb/knowledge.md` 补知识点（真实权重加载路径、chat template 影响）
- [ ] `docs/tasks/README.md` + `docs/plan/M1.md` 状态更新
- [ ] `README.md` 进度同步
- [ ] commit `feat(tests): add real Qwen3-0.6B smoke test (T12 done)`

## 坑（按概率排序）

1. **两个缓存路径**：`Qwen3-0.6B` 和 `Qwen3-0___6B` 都存在，用 `pathlib.Path.exists()` 选一个可用的。
2. **thinking 模式**：Qwen3 默认开 thinking（`<think>...</think>`），greedy decode 下两侧 thinking token
   会影响 token 对齐。用 `enable_thinking=False` 或在 chat template 里加 `/no_think`。
3. **`trust_remote_code`**：加载 tokenizer/model 需要 `trust_remote_code=True`（transformers 侧）。
4. **`use_cache=False`**：inferlite 没有 KV-cache，transformers 侧也要关 cache 保持公平。
5. **tie_word_embeddings**：Qwen3-0.6B `config.json` 里 `tie_word_embeddings: false`，
   权重加载时 `lm_head.weight` 和 `embed_tokens.weight` 独立存在。
6. **`generation_config.json`**：Qwen3-0.6B 有自己的 `generation_config.json`，
   transformers 默认读该配置，可能改变 do_sample / temperature 行为，
   需显式传 `do_sample=False, temperature=1.0` 覆盖。
