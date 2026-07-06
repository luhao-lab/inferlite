# M2-T4 Generate Loop 拆 Prefill/Decode

## 元信息
- **任务 ID**: M2-T4
- **里程碑**: M2（KV Cache）
- **状态**: ✅ done
- **前置**: M2-T3（Model 层 cache 透传）
- **估时**: 2h

## 目标

**要解决什么问题**：
T1-T3 把数据结构和读写接口都备好了，但 `generate()` 还是 M1 版本：每步把所有 token 重新跑一遍 full forward，没有用到 KV Cache。
本卡把 generate loop 拆成 Prefill（一次性处理整个 prompt）+ Decode（每步只跑一个新 token）两阶段，让 KV Cache 真正发挥作用。

**做完是什么效果**：
```python
# 有 cache 和无 cache 的输出完全一致（torch.equal），但有 cache 时 Decode 步更快
out_no_cache = generate(engine, input_ids, max_new_tokens=10)
out_with_cache = generate(engine, input_ids, max_new_tokens=10, kv_cache=cache)
assert torch.equal(out_no_cache, out_with_cache)  # 语义等价
```

**不做什么**（边界）：
只改 generate loop 和 protocol 接口，不改模型内部（T2/T3 已完成）。
性能 benchmark 不在本卡（只要正确性通过即可）。

**在推理链路中的位置**：
```
CLI / 用户调用
    └── generate(engine, input_ids, kv_cache=cache)  ← 本卡
          │
          ├── [Prefill]  engine.model(full_prompt, position_ids, kv_cache)
          │              cache.cur_len = T_prompt    ← 本卡：显式推进 cur_len
          │
          └── [Decode loop]
                engine.model(last_token, position_ids=[[cur_len]], kv_cache)
                cache.cur_len += 1                   ← 本卡：每步推进
                sampler → next_token → 拼到 output
```

## 产出文件
- `inferlite/engine/protocol.py`（修改）— 新增 `position_ids`、`kv_cache` 可选参数
- `inferlite/engine/core.py`（修改）— `generate()` 拆 prefill/decode
- `tests/unit/test_generate_kv.py`（新建）

## 参考代码
- 设计文档 §3.4、§4 ADR-02、ADR-04、ADR-06：`inferlite/docs/m2-kv-cache-design.md`
- 现有 M1 实现：`inferlite/engine/core.py`

## 算法核心

### generate() 两阶段骨架

```python
def generate(engine, input_ids, max_new_tokens, eos_token_id=None, kv_cache=None):
    if kv_cache is None:
        # M1 路径：原逻辑不变（向后兼容）
        for _ in range(max_new_tokens):
            next_token = engine.step(input_ids)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return input_ids

    # M2 路径：prefill + decode loop
    kv_cache.reset()

    # --- Prefill ---
    T_p = input_ids.shape[1]
    position_ids = torch.arange(T_p, device=input_ids.device).unsqueeze(0)  # [1, T_p]
    logits = engine.model(input_ids, position_ids=position_ids, kv_cache=kv_cache)
    kv_cache.cur_len = T_p   # ← 显式更新

    # 采样 prefill 最后一步的 token
    next_token = engine.sampler(logits[:, -1, :])
    input_ids = torch.cat([input_ids, next_token], dim=1)

    # --- Decode Loop ---
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
        pos = torch.tensor([[kv_cache.cur_len]], device=input_ids.device)  # 绝对位置
        logits = engine.model(next_token, position_ids=pos, kv_cache=kv_cache)
        kv_cache.cur_len += 1   # ← 显式更新
        next_token = engine.sampler(logits[:, -1, :])
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
```

**关键点**：
- `position_ids` decode 步用绝对位置 `[[kv_cache.cur_len]]`，不是 `[[0]]`（见 ADR-04）
- `kv_cache.cur_len` 在 generate loop 里显式更新，不在 model 内部更新（见 ADR-02）

### protocol.py 接口扩展

```python
class LLMModel(Protocol):
    def __call__(
        self,
        input_ids: torch.Tensor,
        logits_to_keep: int = 0,
        position_ids: torch.Tensor | None = None,   # 新增
        kv_cache: KVCache | None = None,             # 新增
    ) -> torch.Tensor: ...
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | `generate(kv_cache=KVCache)` 输出 == `generate(kv_cache=None)` | M1 输出 | `torch.equal` |
| 2 | 有 cache 时生成长度正确（max_new_tokens） | 手工断言 | exact |
| 3 | EOS 提前停止（有 cache 路径） | EOS token id | exact |
| 4 | `kv_cache=None` 路径所有 M1 generate 单测继续通过 | 现有测试 | 全绿 |
| 5 | decode 步 `position_ids` 是绝对位置（非从 0 开始） | `kv_cache.cur_len` | exact |

## DoD
- [x] `tests/unit/test_generate_kv.py` 全绿
- [x] `uv run pytest tests/ -q` 所有 M1 单测（95 个）继续通过
- [x] 有 cache == 无 cache（`torch.equal`）在 Qwen3-0.6B 规格小模型上验证
- [x] commit `feat(engine): add prefill/decode split with KV cache (M2-T4)` (`0e13a42`)
- [x] 任务状态改 ✅

## 坑（按概率排序）
1. **prefill 的 `cur_len` 更新时机**：prefill `model()` 调用后立刻 `kv_cache.cur_len = T_p`，不是在 loop 里累加。
2. **EOS 检查位置**：EOS 检查放在 decode loop 开头（检查上一步生成的 token），避免最后一个 EOS token 被漏掉或多生成一步。
3. **`position_ids` 写 `[[0]]` 是沉默 bug**：每步 RoPE 都认为是位置 0，语义错误但不报错，输出质量下降。
4. **`logits_to_keep` 在 decode 步可以为 1**：decode 步只需要最后 1 个 token 的 logits，M1 已有此优化，M2 继承。
5. **prefill 时 `logits_to_keep=1` 是否安全**：prefill 的 `logits[:, -1, :]` 就是 prompt 最后一个 token 的预测，用于采样第一个 generated token，逻辑正确。
