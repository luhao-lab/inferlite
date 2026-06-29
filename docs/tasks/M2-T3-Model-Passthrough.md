# M2-T3 Model 层 cache 透传

## 元信息
- **任务 ID**: M2-T3
- **里程碑**: M2（KV Cache）
- **状态**: ✅ done
- **前置**: M2-T2（Attention KV Cache 接口）
- **估时**: 1h

## 目标

**要解决什么问题**：
T2 改好了 `GQAAttention`，但 `Qwen3Model.forward` 和 `DecoderLayer.forward` 的签名还是 M1 版本，没有 `kv_cache` 参数，新参数传不进去。
本卡做"管道改造"——让参数能从最外层的 `generate()` 一路透传到最里层的 Attention。

**做完是什么效果**：
```python
# kv_cache=None 时：行为与 M1 完全一致，95 个单测全绿（零回归）
model.forward(input_ids, position_ids=pos)

# kv_cache 传入时：K/V 被写入 cache，generate loop 可以用
model.forward(input_ids, position_ids=pos, kv_cache=cache)
```

**不做什么**（边界）：
只做参数透传和 `position_embeddings` 统一计算，不改 Attention 内部逻辑（T2 已完成）。
不改 generate loop（T4 负责）。

**在推理链路中的位置**：
```
generate()                                ← T4
    └── model.forward(kv_cache=cache)     ← 本卡：接收并透传 kv_cache
          ├── position_embeddings = rotary_emb(...)   ← 本卡：统一算一次（M1 是各层各算）
          └── for i, layer in enumerate(self.layers):
                layer.forward(... layer_kv_cache=cache.layers[i] ...)  ← 本卡：透传
                    └── GQAAttention.forward(... layer_kv_cache=... )  ← T2 已完成
```

## 产出文件
- `inferlite/model/qwen3.py`（修改）
- `inferlite/model/layers.py`（修改）— `DecoderLayer` 透传新参数

## 参考代码
- 设计文档 §4 ADR-07：`inferlite/docs/m2-kv-cache-design.md`
- transformers `Qwen3Model.forward`（position_embeddings 统一计算位置）：https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py

## 算法核心

### Qwen3Model.forward 改动

```python
def forward(self, input_ids, position_ids, kv_cache=None):
    hidden = self.embed_tokens(input_ids)

    # 统一计算一次 position_embeddings，传入所有层
    # M1 是各层 Attention 各自调用 self.rotary_emb，重复 28 次
    position_embeddings = self.rotary_emb(hidden, position_ids)

    for i, layer in enumerate(self.layers):
        hidden = layer(
            hidden,
            position_embeddings=position_embeddings,
            layer_kv_cache=kv_cache.layers[i] if kv_cache is not None else None,
            cache_position=kv_cache.cur_len if kv_cache is not None else None,
        )
    return self.norm(hidden)
```

**关键**：`rotary_emb` 从 `GQAAttention` 移到 `Qwen3Model`（或保留在 `GQAAttention` 但 `Qwen3Model` 拿到第一层的引用统一调用一次）。

### DecoderLayer.forward 透传

```python
def forward(self, hidden_states, position_embeddings,
            layer_kv_cache=None, cache_position=None):
    # 透传给 self.self_attn
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        hidden_states,
        position_embeddings=position_embeddings,
        layer_kv_cache=layer_kv_cache,
        cache_position=cache_position,
    )
    hidden_states = residual + hidden_states
    ...
```

## L0 测试清单

本任务以**回归测试为主**，无需新建测试文件，验证 M1 单测全部继续通过即可。

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | M1 全部 95 个单测 | 现有测试结果 | 全绿 |
| 2 | `kv_cache=None` 时 `Qwen3Model.forward` 输出与 M1 完全一致 | M1 forward | exact |

## DoD
- [x] `uv run pytest tests/ -q` 全部 95 个 M1 单测继续通过（不允许任何回归）
- [x] `kv_cache=None` 路径行为与 M1 完全不变
- [x] commit `refactor(model): unify position_embeddings in Qwen3Model, add kv_cache passthrough (M2-T3)` (`1dd4c8d`)
- [x] 任务状态改 ✅

## 坑（按概率排序）
1. **`rotary_emb` 放在哪里**：`GQAAttention.__init__` 里的 `self.rotary_emb` 要删（已移到 Model 层），否则多余参数会占显存并影响权重加载（如果有 checkpoint）。
2. **`kv_cache=None` 兼容路径要显式处理**：`layer(... layer_kv_cache=None, cache_position=None)` 时 Attention 走无 cache 路径，行为与 M1 完全一致。
3. **`position_embeddings` 的 shape 验证**：cos/sin 的 shape 是 `[B, T, head_dim]`，统一算一次传入各层即可，不需要每层各算。
4. **`Qwen3ForCausalLM.forward` 也要同步更新接口**：它调用了 `Qwen3Model.forward`，签名改动要向上传递。
