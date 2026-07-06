# M2-T5 CLI device/dtype/max-seq-len 支持

## 元信息
- **任务 ID**: M2-T5
- **里程碑**: M2（KV Cache）
- **状态**: ✅ done
- **前置**: M2-T4（Generate Loop）
- **估时**: 1h

## 目标

**要解决什么问题**：
T4 的 generate loop 里 device/dtype 都硬编码为 `cpu` / `float32`，用户无法用 MPS（Apple Silicon）或 CUDA 加速，也无法控制 KV Cache 大小。
本卡让 CLI 支持 `--device`、`--dtype`、`--max-seq-len` 三个参数，让推理可以在 MPS 上以 bf16 跑，吞吐更高、显存占用更低。

**做完是什么效果**：
```bash
# Mac M 系列芯片自动用 MPS + bf16
uv run python -m inferlite.cli --model /path/to/qwen3 --prompt "你好" --device auto --dtype auto

# 显式控制
uv run python -m inferlite.cli --model /path/to/qwen3 --prompt "你好" \
  --device mps --dtype bf16 --max-seq-len 2048
```

**不做什么**（边界）：
只改 CLI 入口（`cli.py`）和辅助函数 `resolve_device_dtype`，不改模型和 generate loop 内部。
性能测试 / benchmark 不在本卡。

**在推理链路中的位置**：
```
用户命令行
    └── cli.py --device auto --dtype auto --max-seq-len 1024   ← 本卡
          ├── resolve_device_dtype() → device="mps", dtype=bf16
          ├── model.to(device, dtype)
          ├── KVCache.from_config(..., dtype=bf16, device="mps")
          └── generate()                                        ← T4 已完成
```

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
- [x] `uv run python -m inferlite.cli --model <path> --prompt "hello" --device auto --dtype auto` 在当前机器上跑通
- [x] MPS 可用时自动选 mps + bf16
- [x] `uv run pytest tests/unit/test_cli.py -q` 全绿（10 个用例）
- [x] commit `feat(T5): add --device/--dtype/--max-seq-len, KVCache wiring, and tests` (`a0a2004`)
- [x] 任务状态改 ✅

## 坑（按概率排序）
1. **`mps` 上 bf16 兼容性**：Qwen3 的 `RMSNorm` 在 M1 里已有 fp32 upcast 再 cast 回原 dtype 的逻辑，bf16 应该没问题；但 `index_copy_` 在 MPS 上有限制，M2 用切片写入可以规避。
2. **model.to() 顺序**：先 `load_weights`（fp32 加载）再 `to(dtype=bf16)` 转换，不要反过来。
3. **KVCache device 要与 model 一致**：`KVCache.from_config(..., device=device)` 和 `model.to(device=device)` 必须用同一个 device 字符串。
4. **`--max-seq-len` 越界保护**：如果 prompt_len + max_new_tokens > max_seq_len，generate 时会触发 IndexError，可以在 cli.py 里提前检查并给出友好报错。
