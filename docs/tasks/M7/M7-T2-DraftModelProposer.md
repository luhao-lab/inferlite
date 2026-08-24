# M7-T2 DraftModelProposer

> **任务 ID**: T2
> **里程碑**: M7 推测解码
> **状态**: ✅ done
> **前置**: T1
> **估时**: 1d

## 目标

**要解决什么问题**：
NgramProposer 只能利用文本重复 pattern，自然语言场景命中率低。DraftModelProposer 用一个小模型（或同模型 greedy）做 K 次 autoregressive forward，生成更高质量的 draft tokens + 概率分布，供后续 RejectionSampler 使用。

**做完是什么效果**：
```python
# T2：传入同一个 Qwen3-0.6B 作为 draft_model（代码上区分 draft_model 和 target_model）
proposer = DraftModelProposer(draft_model=model, num_draft_tokens=5)
draft_tokens = proposer.propose(context)
# draft_tokens = [t_100, t_101, t_102, t_103, t_104]
# proposer.last_draft_probs = [p_100, p_101, p_102, p_103, p_104]  # 供 T4 使用
```

**不做什么**（边界）：
- 不加载第二个模型（T2 用同一个 Qwen3-0.6B 实例作为 draft_model，greedy 采样）
- 不做 KV cache 管理（draft model 的 cache 生命周期由 T6 engine loop 管理）
- 不做 batch draft（先单请求）
- 不做 verify/accept（那是 T4 RejectionSampler 的事）

**为什么代码上要区分 draft_model 和 target_model**：
- T2 虽然传入同一个模型实例，但接口设计上是两个独立的引用
- 好处：① 职责分离（proposer 只管 draft，不碰 target）② M10 MoE 自推测时同一个模型不同配置 ③ 后续可换真正的小模型
- 类似 vLLM 的设计：draft_model 和 target_model 是两个引用，可以是同一个也可以是不同的

**在推理链路中的位置**：
```
engine loop decode step:
  ┌─ DraftModelProposer.propose(context) → draft_tokens + draft_probs
  │
  ├─ model([last] + draft_tokens) → logits  ← verify（T6 集成）
  │
  └─ rejection_sample(logits, draft_tokens, draft_probs) → accepted  ← T4
```

## 产出文件

- `inferlite/spec/draft_model_proposer.py::DraftModelProposer`
- `tests/unit/test_draft_model_proposer.py`

## 算法核心

**核心思想**：用 draft_model 做 K 次 autoregressive forward，每次生成 1 个 token + 概率分布，累积 K 个 draft tokens。

T2 定位：**pipeline 验证**。draft_model 和 target_model 是同一个模型，验证整个 spec decode 流程跑通。真正的加速效果等 M10。

**采样策略（关键设计决策）**：

draft model 用 **sampling**（temperature）而非 greedy，原因：

1. **vLLM 做法**：draft model 的采样参数（temperature、top_p、top_k）和用户的 sampling 参数一致
2. **rejection sampling 要求**：`p_draft` 必须是真实的生成分布，greedy 让 `p_draft` 变成 one-hot（选中 token 概率 = 1），rejection sampling 退化为 `accept_prob = min(1, p_target / 1) = p_target`，完全失去 spec decode 意义
3. **T2 同模型**：同一个模型 + 同样的 sampling 参数 → `p_draft ≈ p_target` → 接受率仍然很高

**temperature 的作用**：
- temperature = 0（greedy）：`p_draft` 是 one-hot，rejection sampling 失效
- temperature < 1：`p_draft` 更集中，draft 质量高但接受率可能低
- temperature = 1：`p_draft` 是原始分布，和 target 一致
- temperature > 1：`p_draft` 更分散，draft 质量低但接受率可能高

**推荐**：T2 用 temperature = 1.0（和 target model 一致），验证 pipeline 正确性。

```python
class DraftModelProposer:
    """基于模型的推测解码 drafter。

    用 draft_model 做 K 次 forward，每次生成 1 个 token + 概率分布，
    累积 K 个 draft tokens 和对应的 draft_probs。

    设计说明：
    - draft_model 和 target_model 在代码上是两个独立引用
    - T2 传入同一个模型实例（pipeline 验证）
    - M10 可传入同模型不同配置（如 top-2 vs top-8 experts）

    Args:
        draft_model: 用于 draft 的模型（实现 LLMModel Protocol）
        num_draft_tokens: 每次 draft 几个 token（默认 5）
        temperature: 采样温度（默认 1.0，和 target model 一致）
    """

    def __init__(self, draft_model: LLMModel, num_draft_tokens: int = 5, temperature: float = 1.0):
        self.draft_model = draft_model
        self.num_draft_tokens = num_draft_tokens
        self.temperature = temperature
        self.last_draft_probs: list[torch.Tensor] = []

    def propose(self, context: list[int]) -> list[int]:
        """用 draft_model 做 K 次 forward，生成 draft tokens。

        算法：
        1. 初始化：last_token = context[-1]
        2. 循环 K 次：
           a. logits = draft_model([last_token]) → [1, vocab_size]
           b. logits = logits / temperature  # 温度缩放
           c. probs = softmax(logits) → [1, vocab_size]
           d. next_token = multinomial(probs) → sampling
           e. 保存 probs 到 last_draft_probs
           f. draft_tokens.append(next_token)
           g. last_token = next_token
        3. 返回 draft_tokens
        """
        draft_tokens = []
        draft_probs = []

        last_token = context[-1]

        for _ in range(self.num_draft_tokens):
            logits = self.draft_model(input_ids=[[last_token]])
            logits = logits[0, 0] / self.temperature  # 温度缩放

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()  # sampling

            draft_tokens.append(next_token)
            draft_probs.append(probs)
            last_token = next_token

        self.last_draft_probs = draft_probs
        return draft_tokens
```

**为什么用 sampling 而不是 greedy**：
- greedy（temperature=0）让 `p_draft` 变成 one-hot，rejection sampling 退化为 `accept_prob = p_target`，失去加速意义
- sampling 让 `p_draft` 是真实的生成分布，rejection sampling 正常工作
- vLLM 和原版论文都用 sampling，不用 greedy

**为什么 draft_probs 要保存**：
- T4 RejectionSampler 需要 draft_probs 和 target_probs 做 rejection sampling
- 公式：`accept_prob = min(1, p_target / p_draft)`
- proposer 把每个位置的完整 probs 保存下来供 T4 使用

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|--------|--------------|------|
| 1 | K=0（num_draft_tokens=0） | 返回空列表 | 精确 |
| 2 | K=1（num_draft_tokens=1） | 返回 1 个 token + 1 个 prob | 精确 |
| 3 | K=5（num_draft_tokens=5） | 返回 5 个 token + 5 个 prob | 精确 |
| 4 | greedy 行为（argmax 选择） | 每次选概率最大的 token | 精确 |
| 5 | draft_probs shape | 每个 prob 是 [vocab_size] tensor | 精确 |
| 6 | draft_probs 数值（softmax 后和为 1） | sum(probs) == 1.0 | 1e-5 |
| 7 | last_draft_probs 更新 | propose 后 last_draft_probs 长度为 K | 精确 |
| 8 | Mock model 测试（可控 logits） | 给定 logits，验证 draft 行为 | 精确 |

## DoD

- [x] `DraftModelProposer` 实现完成
- [x] 9 个单测全绿（比计划多 1 个：parameter validation）
- [x] 代码有详细注释（算法步骤 + multinomial sampling + draft_probs 保存）
- [x] commit `feat(spec): add DraftModelProposer for model-based speculative decoding (T2 done)`

## 完成总结

**实际实现**：
- `inferlite/spec/draft_model_proposer.py::DraftModelProposer`：基于模型的推测解码 drafter
- `tests/unit/test_draft_model_proposer.py`：9 个测试用例（K=0/1/5、空 context、probs shape/sum、更新、温度影响、参数验证）

**关键设计决策**：
1. **multinomial sampling 而非 greedy**：保持 p_draft 是真实分布，rejection sampling 才有效
2. **temperature 参数**：控制 draft 分布的集中度，默认 1.0（与 target model 一致）
3. **保存完整 probs**：每个位置的完整 vocab 分布，供 rejection sampling 拒绝时重新采样
4. **draft_model 参数**：与 target_model 区分，支持未来 MoE 自推测（同一模型不同配置）

**与任务卡的差异**：
- 原计划测试 greedy 行为（argmax），实际实现 multinomial sampling，测试改为验证温度参数影响
- 新增 parameter validation 测试（num_draft_tokens >= 0, temperature > 0）

**验证结果**：
- T2 单测：9/9 全绿
- 全量回归：341/341 全绿（332 原有 + 9 新增）

## 坑（按概率排序）

1. **model 接口不匹配**：model 可能不支持 `input_ids=[[last_token]]` 这种 2D list，需要确认 LLMModel Protocol 的接口
2. **logits shape 不对**：model 返回的 logits 可能是 `[batch, seq, vocab]` 或 `[batch, vocab]`，需要处理
3. **draft_probs 内存占用**：K=5 时保存 5 个 `[vocab_size]` tensor，vocab_size=151936 时约 3MB，可以接受
4. **greedy vs sampling**：greedy 是确定性采样，但 T6 集成时可能需要 sampling（temperature > 0），需要扩展接口
5. **KV cache 管理**：draft model 的 KV cache 需要单独管理，但 T2 不处理（T6 负责）
