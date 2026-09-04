# M7-T5 RejectionSampler

> **任务 ID**: T5
> **里程碑**: M7 推测解码
> **状态**: ✅ done
> **前置**: T1-T4（所有 Proposer）
> **估时**: 1d

## 目标

**要解决什么问题**：
T1-T4 的 Proposer 能快速猜 K 个 token，但没有验证机制。需要用 rejection sampling 判断哪些 draft tokens 可以接受，保证输出分布与无推测时完全一致（lossless）。

**做完是什么效果**：
```python
# T5：验证 draft tokens，返回 accepted + bonus
sampler = RejectionSampler(temperature=1.0)
result = sampler.sample(
    draft_tokens=[101, 203, 456],
    draft_probs=[p0, p1, p2],         # 每个 [vocab_size]
    target_logits=logits,              # [4, vocab_size]（K+1 个位置）
)
# result.accepted_tokens = [101, 203]   ← 接受了前 2 个
# result.bonus_token = 555              ← 从修正/目标分布采样
# result.all_tokens = [101, 203, 555]   ← 总共输出 3 个
```

**不做什么**（边界）：
- 不做 KV cache rollback（那是 T6）
- 不做 engine loop 集成（那是 T7）
- 不做 batch rejection sampling（多请求场景）
- 不做 Triton kernel 版本

**在推理链路中的位置**：
```
engine loop decode step:
  ┌─ Proposer.propose(context) → draft_tokens + draft_probs    ← T1-T4
  │
  ├─ model([last] + draft_tokens) → target_logits [K+1, V]     ← verify
  │
  └─ RejectionSampler.sample(draft_tokens, draft_probs, target_logits)
       → accepted_tokens + bonus_token                          ← T5（你在这里）
```

## 产出文件

- `inferlite/spec/rejection_sampler.py::RejectionSampler`
- `tests/unit/test_rejection_sampler.py`

## 算法核心

**核心思想**：用 `accept_prob = min(1, p_target / p_draft)` 决定是否接受每个 draft token，拒绝时从修正分布采样 bonus，保证最终输出分布 = target 分布。

**两种模式**：

| 模式 | 条件 | 接受条件 | bonus 来源 |
|------|------|----------|------------|
| Classic | temperature > 0 | `random < min(1, p_t/p_d)` | `normalize(relu(p_t - p_d))` |
| Greedy | temperature = 0 | `draft_token == argmax(p_t)` | `argmax(p_t)` |

**Classic 模式算法**：

```python
def sample(draft_tokens, draft_probs, target_logits, temperature):
    target_probs = softmax(target_logits / temperature)  # [K+1, V]

    accepted = []
    for i, token in enumerate(draft_tokens):
        p_d = draft_probs[i][token]
        p_t = target_probs[i][token]

        # 防除零
        if p_d < 1e-10:
            accept_prob = 0.0
        else:
            accept_prob = min(1.0, p_t / p_d)

        if random() < accept_prob:
            accepted.append(token)
        else:
            # 拒绝：从修正分布采样 bonus
            adjusted = relu(target_probs[i] - draft_probs[i])
            if adjusted.sum() < 1e-10:
                adjusted = target_probs[i]  # fallback
            bonus = multinomial(adjusted, 1)
            return accepted + [bonus]

    # 全接受：从 target 最后一个位置采样 bonus
    bonus = multinomial(target_probs[K], 1)
    return accepted + [bonus]
```

**Greedy 模式算法**：

```python
def sample_greedy(draft_tokens, target_logits):
    target_argmax = argmax(target_logits, dim=-1)  # [K+1]

    accepted = []
    for i, token in enumerate(draft_tokens):
        if token == target_argmax[i]:
            accepted.append(token)
        else:
            return accepted + [target_argmax[i]]

    return accepted + [target_argmax[K]]
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|--------|--------------|------|
| 1 | 全接受 + bonus | 返回 K+1 个 token | 精确 |
| 2 | 第一个就拒绝 | 返回 1 个 bonus token | 精确 |
| 3 | 中间拒绝 | 返回 accepted + 1 个 bonus | 精确 |
| 4 | K=0（无 draft） | 返回 1 个 bonus（从 target 采样） | 精确 |
| 5 | Greedy 全接受 | argmax 全匹配 → K+1 个 token | 精确 |
| 6 | Greedy 部分拒绝 | 第一个不匹配处截断 + bonus | 精确 |
| 7 | draft_probs shape | 输入 probs 和为 1 | 1e-5 |
| 8 | target_logits shape | [K+1, vocab_size] | 精确 |
| 9 | 修正分布 fallback | p_target == p_draft 时不崩溃 | 精确 |

## DoD

- [x] `RejectionSampler` 实现完成（classic + greedy 两种模式）
- [x] 10 个单测全绿
- [x] 代码有详细注释（accept_prob + 修正分布 + lossless 保证）
- [ ] commit `feat(spec): add RejectionSampler for speculative decoding verification (T5 done)`

## 完成总结

**实际实现**：
- `inferlite/spec/rejection_sampler.py::RejectionSampler`：classic + greedy 两种模式
- `inferlite/spec/rejection_sampler.py::SampleResult`：结果容器（accepted + bonus + 统计属性）
- `tests/unit/test_rejection_sampler.py`：10 个测试用例

**关键设计决策**：
1. **SampleResult 类**：包含 `num_accepted`、`num_rejected`、`num_draft_tokens`，T6 rollback 需要知道拒绝几个
2. **Greedy 不需要 draft_probs**：验证断言移到模式分支之后，greedy 模式不检查 draft_probs
3. **修正分布 fallback**：`relu(p_t - p_d)` 全 0 时 fallback 到 `p_target`，避免 multinomial 崩溃
4. **p_draft 防除零**：`p_d < eps` 时 accept_prob = 0，直接拒绝

**修复的 bug**：
- `num_draft_tokens` 漏传：5 处 `SampleResult` 构造都补上了
- `_sample_classic` / `_sample_greedy` 内部 `K` 未定义：加了 `K = len(draft_tokens)`
- greedy 模式传空 `draft_probs=[]` 被验证拦住：断言移到 classic 分支内

**验证结果**：
- T5 单测：10/10 全绿
- 全量回归：400/400 全绿

**教学要点**：
- **Lossless 保证**：接受 + 修正分布的数学证明，输出分布严格等于 target
- **Bonus token**：保证最坏情况也能产出 1 个 token，永远不亏
- **K+1 个 logits**：前 K 个验证 draft，第 K+1 个给全接受时采 bonus（免费的）

## 坑（按概率排序）

1. **p_draft 为 0 导致除零**：`p_draft[i][token]` 可能为 0（draft 从没采过这个 token），需要 `max(p_d, eps)` 保护
2. **修正分布全 0**：`relu(p_target - p_draft)` 可能全 0（当 p_target ≤ p_draft 对所有 token 成立），fallback 到 target_probs
3. **target_logits 是 K+1 不是 K**：最后一个位置给"全接受"情况的 bonus 用
4. **temperature 同时影响 draft 和 target**：greedy 模式 temperature=0，classic 模式 temperature>0
5. **draft_probs 和 target_probs 的 vocab_size 必须一致**：不同类型 Proposer 的 vocab_size 可能不同，需要在构造时检查
