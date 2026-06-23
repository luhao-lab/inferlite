# M1-T11 CLI + L2 e2e

## 元信息
- **任务 ID**: T11
- **里程碑**: M1·P2（出字闭环）
- **状态**: 🟡 in-progress
- **前置**: T10 `EngineCore.step()`
- **估时**: 2h

## 背景
T10 已经完成最小单步推理：

```text
input_ids -> EngineCore.step -> next_token [B, 1]
```

T11 要把单步 step 扩展成最小 generate loop，并提供一个可以从命令行触发的 L2 e2e 闭环。

最小闭环：

```text
prompt text
  -> tokenizer.encode
  -> input_ids
  -> load_causal_lm_from_hf
  -> EngineCore + GreedySampler
  -> 多步 step + cat
  -> tokenizer.decode
  -> generated text
```

## 目标
实现最小文本生成路径：

```bash
uv run inferlite-generate --model-dir <local_model_dir> --prompt "你好" --max-new-tokens 8
```

或等价 CLI/API。

T11 完成后，应能用 fake/tiny 模型或本地小模型路径完成一次 encode -> generate -> decode 的 e2e 流程。

## 非目标
- 不做 KV cache。
- 不做 streaming。
- 不做 top-k/top-p/temperature。
- 不做服务化 API。
- 不要求真实 Qwen3-0.6B 长文本性能。
- 不做 last-token lm_head 优化；该优化已规划为 T12-pre。

## 产出文件
- `inferlite/engine/generate.py` 或 `inferlite/engine/core.py`
  - 最小 `generate(...)` loop
- `inferlite/cli.py` 或等价 CLI 入口
- `tests/unit/test_generate.py`
- `tests/e2e/test_cli_generate.py` 或 `tests/unit/test_cli.py`（按项目现有测试结构取舍）
- 如需 CLI scripts：`pyproject.toml` 增加 console script

## API 草案

### 1. generate loop

```python
def generate(
    engine: EngineCore,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        next_token = engine.step(input_ids)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

### 2. tokenizer helper

T11 可以直接使用 transformers tokenizer：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = generate(engine, input_ids, max_new_tokens=max_new_tokens)
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### 3. CLI 草案

```python
def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = load_causal_lm_from_hf(args.model_dir)
    sampler = GreedySampler()
    engine = EngineCore(model, sampler)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    output_ids = generate(engine, input_ids, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | `generate` 会 append 指定数量 token | fake engine | exact |
| 2 | `generate` 每轮用更新后的 input_ids | fake engine calls | exact |
| 3 | CLI 参数解析 | `--model-dir --prompt --max-new-tokens` | exact |
| 4 | tokenizer encode/decode helper | fake/stub tokenizer | exact |
| 5 | L2 e2e smoke | fake model/tokenizer 或 tiny local fixture | no crash + expected text/id |

## 建议单测

### 1. generate append

```python
class FakeEngine:
    def step(self, input_ids):
        return torch.tensor([[input_ids.shape[1]]])

input_ids = torch.tensor([[10, 20]])
output = generate(FakeEngine(), input_ids, max_new_tokens=3)
assert output.tolist() == [[10, 20, 2, 3, 4]]
```

### 2. CLI 不强依赖真实 0.6B

CLI 的单测可以通过 monkeypatch：

```text
AutoTokenizer.from_pretrained
load_causal_lm_from_hf
generate
```

避免单测下载或加载真实大模型。

## T12-pre 关系

T11 可以先用当前 T10 的完整 logits 路径：

```text
model(input_ids) -> logits [B, T, V] -> logits[:, -1, :]
```

T12-pre 再优化：

```text
model(input_ids, logits_to_keep=1) -> logits [B, 1, V]
```

不要在 T11 混入该优化，避免 CLI/e2e 和性能优化耦合。

## DoD
- [ ] `generate` loop 实现
- [ ] CLI 或等价 e2e 入口实现
- [ ] 单测覆盖 append 和每轮 input_ids 更新
- [ ] CLI 测试不依赖真实大模型下载
- [ ] T0-T11 回归全绿
- [ ] `docs/kb/knowledge.md` 补 generate loop / CLI e2e 知识点（如缺）
- [ ] `docs/tasks/README.md` 与 `docs/plan/M1.md` 状态更新
- [ ] `README.md` 当前进度同步（任务完成时）
- [ ] commit `feat(engine): add greedy generate loop`

## 坑
1. **不要让单测依赖真实模型下载**：真实 Qwen3-0.6B smoke 放 T12。
2. **output_ids 是完整序列**：decode 时一般 decode prompt + new tokens；后续可再支持只输出新增部分。
3. **没有 EOS 停止**：T11 固定跑 `max_new_tokens`，EOS 停止后续再加。
4. **没有 KV cache**：每步都会 full forward，正确但慢。
5. **真实 tokenizer 可能需要 trust_remote_code**：CLI 先保留参数或固定 True。
