# M7-T3 EagleProposer

> **任务 ID**: T3
> **里程碑**: M7 推测解码
> **状态**: 🔄 in_progress
> **前置**: T2（DraftModelProposer）
> **估时**: 1.5d

## 目标

**要解决什么问题**：
DraftModelProposer（T2）每次 draft 需要 K 次完整 model forward，即使小模型也有完整 decoder layers（28 层 × K 次）。EagleProposer 用一个小 MLP head（2 层 Linear）预测下一个 hidden state，比任何完整 model forward 都快 100 倍以上。

**做完是什么效果**：
```python
# T3：加载训好的 eagle_head，用 target model 的 hidden state 做 draft
proposer = EagleProposer(
    target_model=model,
    eagle_head=head,
    num_draft_tokens=5
)
draft_tokens = proposer.propose(context)
# draft_tokens = [t_{100}, t_{101}, t_{102}, t_{103}, t_{104}]
# proposer.last_draft_probs = [p_{100}, p_{101}, p_{102}, p_{103}, p_{104}]
```

**不做什么**（边界）：
- 不做完整 EAGLE（含 decoder layer + KV cache 的 head）——那是完整版，教学不需要
- 不做 tree-structured draft（EAGLE-2）
- 不做 verify/accept（那是 T5 RejectionSampler）
- 不做 KV cache rollback（那是 T6）

**在推理链路中的位置**：
```
engine loop decode step:
  ┌─ EagleProposer.propose(context) → draft_tokens + draft_probs
  │     h_t = target_model.model(context)[-1]
  │     h_{t+1} = eagle_head(h_t)
  │     h_{t+2} = eagle_head(h_{t+1})  ← 递归
  │     ...
  │
  ├─ model([last] + draft_tokens) → logits  ← verify（T7 集成）
  │
  └─ rejection_sample(logits, draft_tokens, draft_probs) → accepted  ← T5
```

## 产出文件

- `inferlite/spec/eagle_head.py::EagleHead`
- `scripts/train_eagle_head.py`（训练脚本）
- `inferlite/spec/eagle_proposer.py::EagleProposer`
- `tests/unit/test_eagle_proposer.py`

## 算法核心

**核心思想**：transformer 最后一层 hidden states 在相邻位置变化平滑（residual stream 特性），用一个 2 层 MLP 就能近似 h_t → h_{t+1} 的映射。

**EagleHead 结构**：

```python
class EagleHead(nn.Module):
    """h_t → h_{t+1} 的特征预测器。

    2 层 MLP：Linear → SiLU → Linear
    和 Qwen3 的 SwiGLUMLP 结构一致但更轻量（去掉门控机制）。
    """
    def __init__(self, hidden_size: int = 1024):
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h_t):
        h = F.silu(self.fc1(h_t))
        h = self.fc2(h)
        return h
```

**训练脚本**：

```python
# scripts/train_eagle_head.py

def main():
    model = load_qwen3()
    model.eval()

    # 收集训练数据：(h_t, h_{t+1}) pairs
    for prompt in load_prompts():
        hidden_states = model.model(prompt)  # [seq_len, hidden_size]
        for i in range(len(hidden_states) - 1):
            pairs.append((hidden_states[i], hidden_states[i + 1]))

    # 训练 EagleHead
    head = EagleHead(hidden_size=1024)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)

    for step in range(500):
        batch = sample_batch(pairs, batch_size=32)
        h_t, h_true = batch[:, 0], batch[:, 1]

        h_pred = head(h_t)

        # MSE loss：hidden state 数值接近
        v_loss = F.mse_loss(h_pred, h_true)

        # KL loss：token 概率分布接近（EAGLE 论文推荐 p_w=0.1）
        logits_pred = model.lm_head(h_pred)
        logits_true = model.lm_head(h_true)
        p_pred = F.log_softmax(logits_pred, dim=-1)
        p_true = F.softmax(logits_true, dim=-1)
        p_loss = F.kl_div(p_pred, p_true, reduction="batchmean")

        # 混合 loss（EAGLE 推荐配置）
        loss = 1.0 * v_loss + 0.1 * p_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(head.state_dict(), "models/eagle_head.pt")
```

**Proposer 实现**：

```python
class EagleProposer:
    """基于 EAGLE head 的 feature-level drafter。

    用训好的小 MLP 预测 hidden state 序列，比完整 model forward 快 100 倍。

    设计说明：
    - 递归 K 次：h_{t+1} = eagle_head(h_t)
    - 每步成本：2 个矩阵乘法 + 1 个 SiLU（vs 36 层 decoder layer）
    - 误差累积：递归 K 次会累积误差，K 越大 draft 质量越低

    Args:
        target_model: target 模型（用于提取 hidden states + lm_head）
        eagle_head: 训好的 EagleHead
        num_draft_tokens: 每次 draft 几个 token（默认 5）
        temperature: 采样温度（默认 1.0）
    """

    def __init__(self, target_model, eagle_head, num_draft_tokens: int = 5, temperature: float = 1.0):
        self.target_model = target_model
        self.eagle_head = eagle_head
        self.num_draft_tokens = num_draft_tokens
        self.temperature = temperature
        self.last_draft_probs: list[torch.Tensor] = []

    def propose(self, context: list[int]) -> list[int]:
        """用 eagle_head 递归预测，生成 draft tokens。

        算法：
        1. 从 target model 提取最后一个 hidden state h_t
        2. 递归 K 次：
           a. h_next = eagle_head(h_t)
           b. logits = target_model.lm_head(h_next)
           c. probs = softmax(logits / temperature)
           d. token = multinomial(probs)
           e. h_t = h_next  # 递归
        3. 返回 draft_tokens + last_draft_probs
        """
        # 提取最后一个 hidden state
        input_ids = torch.tensor([context], dtype=torch.long)
        hidden_states = self.target_model.model(input_ids)  # [1, seq, hidden_size]
        h_t = hidden_states[0, -1]  # [hidden_size]

        draft_tokens = []
        draft_probs = []

        for _ in range(self.num_draft_tokens):
            h_next = self.eagle_head(h_t)
            logits = self.target_model.lm_head(h_next)  # [vocab_size]
            logits = logits / self.temperature
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()

            draft_tokens.append(token)
            draft_probs.append(probs)
            h_t = h_next  # 递归

        self.last_draft_probs = draft_probs
        return draft_tokens
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|--------|--------------|------|
| 1 | K=0（num_draft_tokens=0） | 返回空列表 | 精确 |
| 2 | K=1（num_draft_tokens=1） | 返回 1 个 token + 1 个 prob | 精确 |
| 3 | K=5（num_draft_tokens=5） | 返回 5 个 token + 5 个 prob | 精确 |
| 4 | draft_probs shape | 每个 prob 是 [vocab_size] tensor | 精确 |
| 5 | draft_probs 数值（softmax 后和为 1） | sum(probs) == 1.0 | 1e-5 |
| 6 | last_draft_probs 更新 | propose 后 last_draft_probs 长度为 K | 精确 |
| 7 | 递归依赖（h_{t+1} 依赖 h_t） | 第二次输出受第一次影响 | 精确 |

## DoD

- [ ] `EagleHead` 实现完成
- [ ] `scripts/train_eagle_head.py` 训练脚本完成
- [ ] `EagleProposer` 实现完成
- [ ] 7 个单测全绿
- [ ] 代码有详细注释（feature-level drafting + MSE+KL loss + 递归）
- [ ] commit `feat(spec): add EagleProposer for feature-level speculative decoding (T3 done)`

## 坑（按概率排序）

1. **hidden states 提取**：`model.model(input_ids)` 返回 `[B, T, hidden_size]`，需要取最后一个位置 `hidden_states[0, -1]`
2. **lm_head 直接调用**：`model.lm_head(h_next)` 输入是 `[hidden_size]`，输出是 `[vocab_size]`，不需要额外 reshape
3. **训练数据量**：几个 prompt 可能不够，建议用 50-100 个 prompt，每个 50-100 tokens，总共 2500-10000 pairs
4. **训练时间**：500 steps × 32 batch = 16000 samples，在 MPS 上几分钟就能跑完
5. **递归误差累积**：K 越大 draft 质量越低，建议 K=3-5，不要太大
6. **KL loss 计算**：`F.kl_div` 要求输入是 log-prob，target 是 prob，注意参数顺序
