# M2-T5 CLI device/dtype/max-seq-len 支持

## 元信息
- **任务 ID**: M2-T5
- **里程碑**: M2（KV Cache）
- **状态**: ⬜ pending
- **前置**: M2-T4（Generate Loop）
- **估时**: 1h

## 目标

修改 `cli.py`，新增 `--device`、`--dtype`、`--max-seq-len` 三个参数，让 inferlite 能在 MPS（Apple Silicon）和 CUDA 上以 bf16 运行 Qwen3-0.6B，并支持用户自定义 KV Cache 大小。

## 产出文件
- `inferlite/cli.py`（修改）

> **注意**：cli.py 改动为纯业务代码，AI 不写实现，只写测试。测试见 `tests/unit/test_cli.py`（已有，可补充新用例）或新建 `tests/unit/test_cli_m2.py`。

## 参考代码
- 设计文档 §3.3、M2 设计文档 device/dtype 小节：`inferlite/docs/m2-kv-cache-design.md`
- 现有 `inferlite/cli.py`

## 算法核心

### 三个新参数

```
--device   {auto,cpu,mps,cuda}   默认 auto（自动检测）
--dtype    {auto,bf16,fp16,fp32} 默认 auto（mps/cuda → bf16，cpu → fp32）
--max-seq-len  int               默认 1024，预分配 KV Cache 的最大序列长度
```

### device/dtype 自动检测逻辑

```python
def resolve_device_dtype(device_arg: str, dtype_arg: str):
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device_arg

    if dtype_arg == "auto":
        dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_arg]

    return device, dtype
```

### 模型加载后 to(device, dtype)

```python
model = Qwen3ForCausalLM(config)
load_weights(model, model_path)
model.to(device=device, dtype=dtype)
model.eval()
```

### KVCache 创建

```python
kv_cache = KVCache.from_config(config, batch_size=1, max_seq_len=args.max_seq_len,
                                dtype=dtype, device=device)
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | `auto` device 检测返回合理值（cpu/mps/cuda 之一） | 运行环境 | — |
| 2 | `auto` dtype 在 cpu 时为 fp32 | `torch.float32` | exact |
| 3 | `auto` dtype 在 mps/cuda 时为 bf16 | `torch.bfloat16` | exact |
| 4 | `--max-seq-len 512` 时 KVCache shape 的 `max_seq_len` 维为 512 | 512 | exact |
| 5 | `--dtype bf16` 显式指定 bf16 | `torch.bfloat16` | exact |

## DoD
- [ ] `uv run python -m inferlite.cli --model <path> --prompt "hello" --device auto --dtype auto` 在当前机器上跑通
- [ ] MPS 可用时自动选 mps + bf16
- [ ] `uv run pytest tests/unit/test_cli.py -q` 全绿（含新增 M2 用例）
- [ ] commit `feat(cli): add --device, --dtype, --max-seq-len for M2 (M2-T5)`
- [ ] `docs/2-tasks/README.md` 状态改 ✅

## 坑（按概率排序）
1. **`mps` 上 bf16 兼容性**：Qwen3 的 `RMSNorm` 在 M1 里已有 fp32 upcast 再 cast 回原 dtype 的逻辑，bf16 应该没问题；但 `index_copy_` 在 MPS 上有限制，M2 用切片写入可以规避。
2. **model.to() 顺序**：先 `load_weights`（fp32 加载）再 `to(dtype=bf16)` 转换，不要反过来。
3. **KVCache device 要与 model 一致**：`KVCache.from_config(..., device=device)` 和 `model.to(device=device)` 必须用同一个 device 字符串。
4. **`--max-seq-len` 越界保护**：如果 prompt_len + max_new_tokens > max_seq_len，generate 时会触发 IndexError，可以在 cli.py 里提前检查并给出友好报错。
