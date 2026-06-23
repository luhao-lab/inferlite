# M2-T2 Attention KV Cache 接口

## 元信息
- **任务 ID**: M2-T2
- **里程碑**: M2（KV Cache）
- **状态**: ⬜ pending
- **前置**: M2-T1（KVCache 数据结构）
- **估时**: 2h

## 目标

修改 `model/attention.py` 的 `GQAAttention.forward`，支持 prefill/decode 两阶段：新增 `position_embeddings`、`layer_kv_cache`、`cache_position` 三个参数，实现 cache 读写逻辑，并将 causal mask 改为 `T > 1` 判断。

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
- [ ] `tests/unit/test_attention_kv.py` 全绿
- [ ] `kv_cache=None` 时所有 M1 attention 单测继续通过：`uv run pytest tests/unit/test_attention.py -q`
- [ ] 有 cache 的输出与无 cache fp32 误差 < 1e-5
- [ ] commit `feat(model): add KV cache support to GQAAttention (M2-T2)`
- [ ] `docs/2-tasks/README.md` 状态改 ✅

## 坑（按概率排序）
1. **causal mask shape 变了**：有 cache 时 K 维度是 `cache_position + T`，mask 应是 `[T, T_k]` 不是 `[T, T]`；prefill 无 cache 时仍是 `[T, T]`。
2. **repeat_kv 时机**：cache 读写发生在 `repeat_kv` 之前（cache 存原始 `n_kv` 维度），读出来之后再 repeat。
3. **`position_embeddings` 的 cos/sin 已按 position_ids 算好**：不需要再传 `position_ids`，直接用传进来的 `(cos, sin)`。
4. **删 `self.rotary_emb` 属性**：移到 `Qwen3Model` 后 `GQAAttention.__init__` 里的 `self.rotary_emb` 也要删，但要保证 `RotaryEmbedding` 仍在 `Qwen3Model` 里存在。
5. **`cache_position` 用 int 还是 tensor**：用 `int` 即可，切片操作支持 Python int，不需要 `torch.tensor`。
