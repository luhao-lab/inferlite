# M7-T4 MTPProposer

> **任务 ID**: T4
> **里程碑**: M7 推测解码
> **状态**: ✅ done
> **前置**: T3（EagleProposer）
> **估时**: 1d

## 目标

**要解决什么问题**：
EagleProposer（T3）用单个 head 递归 K 次预测 +1, +2, ..., +K，每步误差累积，draft 质量随 K 增大下降明显。MTPProposer 用多个独立 head，每个直接预测 +N 位置的 hidden state，避免递归误差累积。

**做完是什么效果**：
```python
# T4：加载多个 head，每个 head 独立预测不同位置
proposer = MTPProposer(
    target_model=model,
    heads=[head_1, head_2, head_3],  # 每个 head 预测 +1, +2, +3
    num_draft_tokens=3
)
draft_tokens = proposer.propose(context)
# draft_tokens = [t_{100}, t_{101}, t_{102}]
# proposer.last_draft_probs = [p_{100}, p_{101}, p_{102}]
```

**不做什么**（边界）：
- 不做原生 MTP（模型结构修改，那是预训练阶段的事）
- 不做 EAGLE 递归（那是 T3）
- 不做 verify/accept（那是 T5 RejectionSampler）
- 不做 KV cache rollback（那是 T6）

**在推理链路中的位置**：
```
engine loop decode step:
  ┌─ MTPProposer.propose(context) → draft_tokens + draft_probs
  │     head_1(h_t) → h_{t+1} → token_{t+1}
  │     head_2(h_t) → h_{t+2} → token_{t+2}
  │     head_3(h_t) → h_{t+3} → token_{t+3}
  │
  ├─ model([last] + draft_tokens) → logits  ← verify（T7 集成）
  │
  └─ rejection_sample(logits, draft_tokens, draft_probs) → accepted  ← T5
```

## 产出文件

- `inferlite/spec/mtp_proposer.py::MTPProposer`
- `scripts/train_mtp_heads.py`（训练脚本，训练多个 head）
- `tests/unit/test_mtp_proposer.py`

## 算法核心

**核心思想**：训练 K 个独立的 MTP head，每个 head 直接预测 +N 位置的 hidden state，不依赖前一个 head 的输出，避免递归误差累积。

**T3（EagleProposer）vs T4（MTPProposer）**：

```
EagleProposer（T3，单 head 递归）：
  head_1(h_t)       → h_{t+1} → token_{t+1}
  head_1(h_{t+1})   → h_{t+2} → token_{t+2}  ← 依赖上一步输出，误差累积
  head_1(h_{t+2})   → h_{t+3} → token_{t+3}

MTPProposer（T4，多 head 独立）：
  head_1(h_t) → h_{t+1} → token_{t+1}
  head_2(h_t) → h_{t+2} → token_{t+2}  ← 直接从 h_t 预测，无累积误差
  head_3(h_t) → h_{t+3} → token_{t+3}
```

**Draft head 结构（和 T3 相同）**：

```python
class MTPHead(nn.Module):
    """预测 +N 位置的 hidden state（和 EagleHead 结构相同，训练数据不同）"""
    def __init__(self, hidden_size):
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h_t):
        h_next = F.silu(self.fc1(h_t))
        h_next = self.fc2(h_next)
        return h_next  # [hidden_size]
```

**训练脚本（多个 head）**：

```python
# scripts/train_mtp_heads.py

def main():
    model = load_qwen3()
    model.eval()

    # 收集训练数据：(h_t, h_{t+N}) pairs
    num_heads = 3  # 预测 +1, +2, +3
    pairs_by_offset = {i: [] for i in range(1, num_heads + 1)}

    for prompt in load_prompts():
        hidden_states = model.get_hidden_states(prompt)  # [seq_len, hidden_size]
        for offset in range(1, num_heads + 1):
            for i in range(len(hidden_states) - offset):
                pairs_by_offset[offset].append(
                    (hidden_states[i], hidden_states[i + offset])
                )

    # 分别训练每个 head
    heads = {}
    for offset, pairs in pairs_by_offset.items():
        head = MTPHead(hidden_size=896)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)

        for step in range(500):
            batch = sample_batch(pairs, batch_size=32)
            h_t, h_tn = batch[:, 0], batch[:, 1]

            h_pred = head(h_t)
            loss = F.mse_loss(h_pred, h_tn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        heads[offset] = head
        torch.save(head.state_dict(), f"models/mtp_head_plus{offset}.pt")
```

**Proposer 实现**：

```python
class MTPProposer:
    """基于多 head 的 MTP drafter。

    用 K 个独立的 MTP head，每个直接从 h_t 预测 +N 位置的 hidden state，
    避免递归误差累积。

    设计说明：
    - 每个 head 独立预测，无递归依赖
    - draft 质量随 K 增大下降更慢（对比 EagleProposer）
    - 本质是"伪 MTP"：外挂多个 head，不是原生 MTP 模型

    Args:
        target_model: target 模型（用于提取 hidden states）
        heads: 多个 MTPHead，按顺序预测 +1, +2, +3, ...
        num_draft_tokens: 每次 draft 几个 token（= heads 数量）
    """

    def __init__(self, target_model, heads: list[MTPHead], num_draft_tokens: int = 3):
        self.target_model = target_model
        self.heads = heads
        self.num_draft_tokens = min(num_draft_tokens, len(heads))
        self.last_draft_probs: list[torch.Tensor] = []

    def propose(self, context: list[int]) -> list[int]:
        """用多个 head 独立预测，生成 draft tokens。

        算法：
        1. 从 target model 提取最后一个 hidden state h_t
        2. 对每个 head_i：
           a. h_{t+i} = head_i(h_t)  # 直接预测 +i 位置
           b. logits = target_model.lm_head(h_{t+i})
           c. probs = softmax(logits / temperature)
           d. token = multinomial(probs)
        3. 返回 [token_{t+1}, token_{t+2}, ..., token_{t+K}]
        """
        h_t = self.target_model.get_last_hidden_state(context)

        draft_tokens = []
        draft_probs = []

        for i, head in enumerate(self.heads[:self.num_draft_tokens]):
            h_next = head(h_t)
            logits = self.target_model.lm_head(h_next)
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()

            draft_tokens.append(token)
            draft_probs.append(probs)

        self.last_draft_probs = draft_probs
        return draft_tokens
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|--------|--------------|------|
| 1 | 多 head 独立预测（不依赖前一个输出） | head_2 的输出不受 head_1 影响 | 精确 |
| 2 | draft tokens 数量 = heads 数量 | num_draft_tokens = len(heads) | 精确 |
| 3 | draft_probs shape | 每个 prob 是 [vocab_size] tensor | 精确 |
| 4 | Mock heads 测试（可控输出） | 给定 head 输出，验证 token 选择 | 精确 |
| 5 | num_draft_tokens > len(heads) 时截断 | draft 数量不超过 heads 数量 | 精确 |
| 6 | last_draft_probs 更新 | propose 后 last_draft_probs 长度为 K | 精确 |

## DoD

- [x] `MTPProposer` 实现完成
- [x] `scripts/train_mtp_heads.py` 训练脚本完成
- [x] 9 个单测全绿（8 个 MTPProposer + 1 个 Head 独立性测试）
- [x] 代码有详细注释（多 head vs 单 head 递归的区别）
- [ ] commit `feat(spec): add MTPProposer for multi-head speculative decoding (T4 done)`

## 完成总结

**实际实现**：
- `inferlite/spec/mtp_proposer.py::MTPProposer`：K 个独立 head 并行预测，无递归误差累积
- `scripts/train_mtp_heads.py`：收集 (h_t, h_{t+N}) pairs + MSE+KL 混合 loss 训练 K 个 head
- `tests/unit/test_mtp_proposer.py`：9 个测试用例（8 个 MTPProposer + 1 个 Head 独立性）
- 复用 `EagleHead` 结构（和 T3 完全相同），不新建 `MTPHead`

**关键设计决策**：
1. **复用 EagleHead**：MTP head 和 Eagle head 结构完全一致（`Linear → SiLU → Linear`），只是训练数据不同（offset=1 vs offset=N），不需要重复定义
2. **MSE + KL 混合 loss**：和 T3 一致，`1.0 * MSE + 0.1 * KL`（EAGLE 论文推荐）
3. **num_draft_tokens 截断**：`min(num_draft_tokens, len(heads))`，防止请求的 draft 数量超过 head 数量
4. **device 自动推断**：`next(model.parameters()).device`，和 EagleProposer 保持一致
5. **num_heads 可配置**：训练脚本的 head 数量通过变量控制，方便调整

**验证结果**：
- T4 单测：9/9 全绿
- 全量回归：390/390 全绿

**教学要点**：
- **独立 vs 递归**：MTP 的每个 head 都从同一个 h_t 出发，不像 EAGLE 递归依赖前一步输出
- **伪 MTP vs 原生 MTP**：我们是外挂 MLP head（post-trained），DeepSeek-V3 是预训练阶段内置 MTP module（含 decoder layer + shared lm_head）
- **K 的代价**：每增加一个 head 就多 ~2M 参数 + 一次训练，但换来的是无误差累积的 draft 质量

**与任务卡的差异**：
- 测试用例从 6 个扩展到 9 个（增加 Head 独立性测试、K=0 边界、截断测试）
- 训练脚本改用 MSE + KL 混合 loss（原任务卡只有 MSE）
- 不新建 MTPHead 类，直接复用 EagleHead

## 坑（按概率排序）

1. **训练时间翻倍**：K 个 head 需要训练 K 次，K=3 时训练时间 3x（但每个 head 可以并行训练）
2. **文件管理**：K 个 head 产生 K 个 .pt 文件，需要统一目录管理（`models/mtp_heads/`）
3. **lm_head 复用**：MTPProposer 需要访问 target_model.lm_head 来映射 hidden state → token，需要确认 model 接口支持
4. **和 EAGLE 对比**：T7 benchmark 需要对比 EAGLE（单 head 递归）vs MTP（多 head 独立），证明多 head 的接受率更高
5. **num_draft_tokens 固定**：MTP 的 K 由 heads 数量决定，不像 n-gram/DraftModel 可以动态调整
