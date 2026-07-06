# M2-T2 Attention KV Cache 接口

## 元信息
- **任务 ID**: M2-T2
- **里程碑**: M2（KV Cache）
- **状态**: ✅ done
- **前置**: M2-T1（KVCache 数据结构）
- **估时**: 2h

## 目标

**要解决什么问题**：
T1 造好了盒子，但 M1 的 `GQAAttention.forward` 不知道这个盒子的存在，每次调用都只用当前输入的 K/V。
本卡让 Attention 学会"往盒子里写，从盒子里读"，这是 KV Cache 真正发挥作用的关键一步。

**做完是什么效果**：
Prefill（T=5）后再做一步 Decode（T=1），Decode 的 Attention 输出和"把 6 个 token 全量输入做 full attention"的结果数值一致（fp32 误差 < 1e-5）：
```python
# prefill 5 个 token
out_prefill = attn(x[:, :5, :], ..., layer_kv_cache=cache.layers[0], cache_position=0)
cache.cur_len = 5
# decode 1 个 token，结果应与 full attention 中第 6 步一致
out_decode = attn(x[:, 5:6, :], ..., layer_kv_cache=cache.layers[0], cache_position=5)
```

**不做什么**（边界）：
只改 `attention.py`，不动 `qwen3.py` / `layers.py`（T3 负责在 Model 层透传参数）。
不负责 cur_len 的推进（T4 负责）。

**在推理链路中的位置**：
```
generate()
    └── model.forward(kv_cache=cache)             ← T3 透传参数
          └── 每层 DecoderLayer → GQAAttention    ← 本卡
                ├── Prefill: 写入 cache.layers[i].k/v[:, :, 0:T_p, :]
                ├── Decode:  写入 cache.layers[i].k/v[:, :, cur_len:cur_len+1, :]
                └── 读取:    k/v[:, :, :cur_len+T, :]（完整有效历史）
```

## 产出文件
- `inferlite/model/attention.py`（修改）— 新接口 + cache 读写 + causal mask 改动
- `tests/unit/test_attention_kv.py`（新建）

## 参考代码
- 设计文档 §4 ADR-03、ADR-05：`inferlite/docs/m2-kv-cache-design.md`
- 现有 M1 实现：`inferlite/model/attention.py`
- transformers `Qwen3Attention.forward`（position_embeddings 传入方式）：https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py

## 算法核心

### 新接口签名

```python
def forward(
    self,
    hidden_states: torch.Tensor,                              # [B, T, H]
    position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (cos, sin)，由 Qwen3Model 统一计算
    layer_kv_cache: LayerKVCache | None = None,
    cache_position: int | None = None,   # = kv_cache.cur_len，由 generate loop 透传
) -> torch.Tensor:
```

**关键变化**：
- 删除 `position_ids` 参数（由 `position_embeddings` 替代）
- 删除内部 `self.rotary_emb(...)` 调用（移到 `Qwen3Model.forward`）

### Cache 读写逻辑

```python
if layer_kv_cache is not None:
    # 写入当前 token(s)：切片赋值，原地写入，不分配新内存
    layer_kv_cache.k[:, :, cache_position:cache_position + T, :] = k
    layer_kv_cache.v[:, :, cache_position:cache_position + T, :] = v
    # 读取完整有效历史
    k = layer_kv_cache.k[:, :, :cache_position + T, :]
    v = layer_kv_cache.v[:, :, :cache_position + T, :]
```

### Causal Mask 改动

```python
# 旧：无条件构造（M1）
# 新：T > 1 时才需要（decode 步 T=1，无需 causal mask）
if T > 1:
    T_k = k.shape[-2]  # 带 cache 时 T_k > T
    causal_mask = torch.triu(torch.ones(T, T_k, dtype=torch.bool, device=...), diagonal=1)
    ...
```

注意：有 cache 时 Q 是 `[B, H_q, T, D]`，K 是 `[B, H_q, T_k, D]`，`T_k >= T`，mask shape 应为 `[T, T_k]` 而非 `[T, T]`。

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | prefill 后 `layer_kv_cache.k[:,:,:T_p,:]` 已写入 | 直接比较写入前后 | exact |
| 2 | decode 步 cache 追加：写在 `cache_position` 位置 | 手工验证 shape/值 | exact |
| 3 | prefill 有 causal mask（`T_p > 1`） | `T > 1` 判断分支 | — |
| 4 | decode 步无 causal mask（`T = 1`） | `T > 1` 判断分支 | — |
| 5 | 有 cache 的 attention 输出 == 无 cache（M1 full attention） | M1 `forward` | fp32 `1e-5` |
| 6 | `kv_cache=None` 时行为与 M1 完全一致（兼容路径） | M1 `forward` | exact |

## DoD
- [x] `tests/unit/test_attention_kv.py` 全绿
- [x] `kv_cache=None` 时所有 M1 attention 单测继续通过：`uv run pytest tests/unit/test_attention.py -q`
- [x] 有 cache 的输出与无 cache fp32 误差 < 1e-5
- [x] commit `feat(model): add KV cache support to GQAAttention (M2-T2)` (`8179b74`)
- [x] 任务状态改 ✅

## 坑（按概率排序）
1. **causal mask shape 变了**：有 cache 时 K 维度是 `cache_position + T`，mask 应是 `[T, T_k]` 不是 `[T, T]`；prefill 无 cache 时仍是 `[T, T]`。
2. **repeat_kv 时机**：cache 读写发生在 `repeat_kv` 之前（cache 存原始 `n_kv` 维度），读出来之后再 repeat。
3. **`position_embeddings` 的 cos/sin 已按 position_ids 算好**：不需要再传 `position_ids`，直接用传进来的 `(cos, sin)`。
4. **删 `self.rotary_emb` 属性**：移到 `Qwen3Model` 后 `GQAAttention.__init__` 里的 `self.rotary_emb` 也要删，但要保证 `RotaryEmbedding` 仍在 `Qwen3Model` 里存在。
5. **`cache_position` 用 int 还是 tensor**：用 `int` 即可，切片操作支持 Python int，不需要 `torch.tensor`。
