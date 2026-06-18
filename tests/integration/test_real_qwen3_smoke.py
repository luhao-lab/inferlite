"""Integration smoke test: real Qwen3-0.6B greedy generation.

T12 目标：加载本地真实 Qwen3-0.6B 权重，用 inferlite greedy generate 生成前 10 个 token，
验证 token ids 与 transformers.generate(do_sample=False) 精确一致。

运行方式（需要本地模型，CI 自动跳过）：
  uv run pytest tests/integration/test_real_qwen3_smoke.py -m local_model -v -s

CI 不加 -m local_model，本文件中的用例会自动被 skip。

设计说明：
- chat template + /no_think：
    Qwen3 默认开启 thinking 模式，裸 prompt 下输出重复、不稳定。
    通过 chat template 构造格式化 prompt，并在末尾加 /no_think 关闭 thinking，
    两侧生成在 token id 层面完全一致（T11 经验教训）。
- do_sample=False / temperature=1.0：
    transformers 会读 generation_config.json，可能默认开 do_sample。
    必须显式覆盖，否则 transformers 侧带随机性，无法精确比对。
- use_cache=False (transformers)：
    inferlite 没有 KV cache，关掉 transformers KV cache 保持计算等价。
- 前 10 token 精确匹配：
    int64 精确匹配，不用 atol/rtol。
"""

from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3ForCausalLM

from inferlite.engine import EngineCore, generate
from inferlite.model.weights import load_causal_lm_from_hf
from inferlite.sampler import GreedySampler

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# ModelScope 可能产生两种不同的缓存目录名：点号版和下划线版。
_CANDIDATE_MODEL_DIRS = [
    Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-0.6B",
    Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
]

# 对齐用的生成 token 数量（不含 prompt）
_NEW_TOKENS = 10

# Qwen3 thinking 模式的关键字（出现在 chat template extra 里）
_NO_THINK_SUFFIX = "/no_think"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_model_dir() -> Path:
    """在候选路径中找到第一个包含 model.safetensors 的目录。

    Returns:
        Path to the local model directory.

    Raises:
        pytest.skip: 如果没有找到本地模型，跳过该测试，不报错。
    """
    for candidate in _CANDIDATE_MODEL_DIRS:
        if (candidate / "model.safetensors").exists():
            return candidate
    # 没找到模型就 skip，而不是 fail：本地没下载模型是正常的。
    pytest.skip(
        "Local Qwen3-0.6B model not found. "
        f"Searched: {[str(p) for p in _CANDIDATE_MODEL_DIRS]}. "
        "Run `modelscope download --model Qwen/Qwen3-0.6B` to download."
    )


def _build_prompt(tokenizer: AutoTokenizer) -> str:
    """用 chat template 构造格式化 prompt，并关闭 thinking 模式。

    为什么用 chat template？
    - Qwen3 在 <|im_start|>...<|im_end|> 格式下训练，裸 prompt 会导致输出不稳定、重复。
    - thinking 模式（<think>...</think>）开启时会多产生大量 token，
      两侧对齐必须保持 thinking 行为一致。

    为什么加 /no_think？
    - Qwen3 chat template 支持在用户消息末尾加 /no_think 关闭 thinking。
    - 关闭后输出更稳定，且不需要处理 thinking token 的对齐问题。
    """
    messages = [{"role": "user", "content": f"What is 1+1? {_NO_THINK_SUFFIX}"}]
    # add_generation_prompt=True 会在末尾加上 <|im_start|>assistant\n，
    # 这样模型知道下一步要输出 assistant 的回答。
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------


@pytest.mark.local_model
def test_qwen3_generate_matches_transformers() -> None:
    """inferlite greedy generate 的前 N 个 token 必须与 transformers 精确一致。

    验证维度：
    1. output_ids shape 正确（[1, T + _NEW_TOKENS]）
    2. 前 _NEW_TOKENS 个 new token id 与 transformers 精确匹配（int64 精确，atol=0）
    3. decode 后输出非空（可读性目测检查）
    """
    model_dir = _find_model_dir()

    # ------------------------------------------------------------------
    # 1. 加载 tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    # ------------------------------------------------------------------
    # 2. 构造格式化 prompt 并 encode
    # ------------------------------------------------------------------
    prompt_text = _build_prompt(tokenizer)
    # return_tensors="pt" 返回 [1, T] 的 int64 tensor。
    input_ids: torch.Tensor = tokenizer.encode(prompt_text, return_tensors="pt")
    prompt_len = input_ids.shape[1]

    # ------------------------------------------------------------------
    # 3. inferlite 生成
    # ------------------------------------------------------------------
    # load_causal_lm_from_hf 加载完整 Qwen3ForCausalLM（backbone + lm_head）。
    inferlite_model = load_causal_lm_from_hf(str(model_dir))
    inferlite_model.eval()

    sampler = GreedySampler()
    engine = EngineCore(inferlite_model, sampler)

    with torch.no_grad():
        inferlite_output_ids = generate(engine, input_ids, max_new_tokens=_NEW_TOKENS)

    # generate 返回 [1, T + _NEW_TOKENS]，new tokens 从 prompt_len 位置开始。
    inferlite_new_tokens = inferlite_output_ids[0, prompt_len:]
    assert inferlite_new_tokens.shape == (
        _NEW_TOKENS,
    ), f"inferlite generated {inferlite_new_tokens.shape[0]} tokens, expected {_NEW_TOKENS}"

    # ------------------------------------------------------------------
    # 4. transformers 生成（作为 ground truth）
    # ------------------------------------------------------------------
    # 用完整 CausalLM，不加 device_map 避免 MPS/CUDA 与 CPU 精度差异。
    hf_model = HFQwen3ForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,  # 与 inferlite 默认 dtype 一致
        trust_remote_code=True,
        use_cache=False,  # inferlite 没有 KV cache，关掉 HF KV cache 保持等价
    )
    hf_model.eval()

    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=_NEW_TOKENS,
            do_sample=False,  # 贪心解码，确定性输出
            temperature=1.0,  # 覆盖 generation_config.json 里可能有的 temperature
            top_p=1.0,  # 覆盖 generation_config.json 里可能有的 top_p
            use_cache=False,  # 与模型加载时的 use_cache=False 保持一致
        )

    hf_new_tokens = hf_output[0, prompt_len:]
    assert hf_new_tokens.shape == (
        _NEW_TOKENS,
    ), f"transformers generated {hf_new_tokens.shape[0]} tokens, expected {_NEW_TOKENS}"

    # ------------------------------------------------------------------
    # 5. 对齐验证：前 N 个 token id 精确匹配
    # ------------------------------------------------------------------
    # 打印方便调试，不影响断言。
    inferlite_text = tokenizer.decode(inferlite_new_tokens, skip_special_tokens=False)
    hf_text = tokenizer.decode(hf_new_tokens, skip_special_tokens=False)
    print(f"\n[inferlite] new tokens: {inferlite_new_tokens.tolist()}")
    print(f"[inferlite] decoded:    {inferlite_text!r}")
    print(f"[HF]        new tokens: {hf_new_tokens.tolist()}")
    print(f"[HF]        decoded:    {hf_text!r}")

    # 核心断言：int64 精确匹配，不允许任何偏差。
    assert torch.equal(inferlite_new_tokens, hf_new_tokens), (
        f"Token mismatch!\n"
        f"  inferlite: {inferlite_new_tokens.tolist()}\n"
        f"  HF:        {hf_new_tokens.tolist()}\n"
        f"  inferlite decoded: {inferlite_text!r}\n"
        f"  HF decoded:        {hf_text!r}"
    )

    # 可读性验证：输出不能是空字符串。
    full_inferlite_text = tokenizer.decode(inferlite_output_ids[0], skip_special_tokens=True)
    assert full_inferlite_text.strip(), "inferlite output is empty after decoding"
