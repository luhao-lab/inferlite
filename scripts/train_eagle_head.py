"""训练 EagleHead：用 Qwen3-0.6B 的 hidden states 训练 feature-level drafter。

Loss = 1.0 * MSE(h_pred, h_true) + 0.1 * KL(p_pred, p_true)

MSE：让 hidden state 数值接近
KL：让 token 概率分布也接近（EAGLE 论文推荐 p_w=0.1）

用法：
  uv pip install -e ".[training]"
  python scripts/train_eagle_head.py
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from inferlite.model.weights import load_causal_lm_from_hf
from inferlite.spec.eagle_head import EagleHead


def get_device() -> torch.device:
    """自动选择设备：MPS (Apple Silicon) > CUDA > CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_prompts(num_prompts: int = 100, min_length: int = 50) -> list[str]:
    """从 WikiText-2 加载训练数据。

    Args:
        num_prompts: 要加载的 prompt 数量
        min_length: 最小字符长度（过滤空行和标题）

    Returns:
        list of text strings
    """
    print("正在加载 WikiText-2 数据集...")
    try:
        dataset = load_dataset("wikimedia/wikitext", "wikitext-2-raw-v1", split="train")

        # 过滤空行和标题，取前 num_prompts 条
        prompts = [text.strip() for text in dataset["text"] if len(text.strip()) > min_length][
            :num_prompts
        ]
    except Exception as e:
        print(f"WikiText-2 加载失败（{e}），使用内置示例文本")
        prompts = _builtin_prompts(num_prompts)

    print(f"加载了 {len(prompts)} 条 prompt")
    if prompts:
        print(f"示例: {prompts[0][:100]}...")
    return prompts


def _builtin_prompts(num_prompts: int) -> list[str]:
    """内置示例文本，WikiText-2 加载失败时使用。"""
    base = [
        "The Transformer architecture introduced by Vaswani et al in 2017 has become "
        "the foundation of modern natural language processing. It uses self-attention "
        "mechanisms to capture relationships between tokens in a sequence efficiently.",
        "In speculative decoding a small draft model proposes multiple tokens quickly "
        "and a large target model verifies them in parallel. This approach can significantly "
        "reduce inference latency while maintaining the same output distribution quality.",
        "Machine learning models learn patterns from data through optimization algorithms. "
        "The training process involves computing gradients of a loss function with respect "
        "to model parameters and updating them iteratively using gradient descent methods.",
        "Large language models are trained on massive text corpora to predict the next "
        "token in a sequence. The model learns grammar facts reasoning abilities and "
        "even some coding skills from the training data through this simple objective.",
        "Neural networks with many layers can learn hierarchical representations of data. "
        "Each layer extracts increasingly abstract features from the input allowing the "
        "network to capture complex patterns that simpler models cannot represent well.",
        "Attention mechanisms allow models to focus on relevant parts of the input when "
        "making predictions. The query key value framework provides a flexible way to "
        "compute weighted combinations of information from different token positions.",
        "Gradient descent is the most common optimization algorithm used in deep learning. "
        "It iteratively adjusts model parameters in the direction that reduces the loss "
        "function using the chain rule of calculus to compute gradients efficiently.",
        "Tokenization is the process of converting text into smaller units called tokens. "
        "Modern tokenizers like BPE and SentencePiece learn subword units from data "
        "balancing vocabulary size with the ability to represent rare words accurately.",
        "Knowledge distillation transfers knowledge from a large teacher model to a "
        "smaller student model. The student learns to mimic the teacher output distribution "
        "achieving similar performance with fewer parameters and faster inference speed.",
        "Reinforcement learning from human feedback aligns language models with human "
        "preferences. A reward model trained on human comparisons guides the language model "
        "to generate responses that are helpful harmless and honest for all users.",
    ]
    prompts = base * ((num_prompts // len(base)) + 1)
    return prompts[:num_prompts]


def collect_hidden_states(
    model,
    prompts: list[str],
    tokenizer,
    device: torch.device,
    max_length: int = 128,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """收集 (h_t, h_{t+1}) pairs。

    Args:
        model: Qwen3ForCausalLM（frozen）
        prompts: 文本列表
        tokenizer: 分词器
        device: 设备
        max_length: 每个 prompt 最大 token 长度（避免 OOM）

    Returns:
        list of (h_t, h_{t+1}) tuples, each shape [hidden_size]
    """
    print("正在收集 hidden states...")
    all_pairs = []

    for i, prompt in enumerate(prompts):
        # tokenize
        tokens = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).to(device)

        # forward，拿 hidden states
        with torch.no_grad():
            hidden = model.model(tokens)  # [1, seq_len, hidden_size]

        # 构造 (h_t, h_{t+1}) pairs
        for j in range(hidden.shape[1] - 1):
            h_t = hidden[0, j]  # [hidden_size]
            h_next = hidden[0, j + 1]  # [hidden_size]
            all_pairs.append((h_t, h_next))

        if (i + 1) % 20 == 0:
            print(f"  进度: {i + 1}/{len(prompts)} prompts, {len(all_pairs)} pairs")

    print(f"总共收集了 {len(all_pairs)} pairs")
    return all_pairs


def train_eagle_head(
    model,
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_steps: int = 500,
    batch_size: int = 32,
    lr: float = 1e-4,
    v_w: float = 1.0,
    p_w: float = 0.1,
) -> EagleHead:
    """训练 EagleHead。

    Args:
        model: target model（用于 lm_head 计算 KL loss）
        pairs: (h_t, h_{t+1}) 训练数据
        device: 设备
        num_steps: 训练步数
        batch_size: batch 大小
        lr: 学习率
        v_w: MSE loss 权重（EAGLE 推荐 1.0）
        p_w: KL loss 权重（EAGLE 推荐 0.1）

    Returns:
        训练好的 EagleHead
    """
    print(f"\n开始训练 EagleHead（{num_steps} steps, batch_size={batch_size}）...")

    hidden_size = pairs[0][0].shape[0]
    head = EagleHead(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    for step in range(num_steps):
        # 随机采样 batch
        indices = torch.randint(0, len(pairs), (batch_size,))
        h_t = torch.stack([pairs[i][0] for i in indices])  # [batch, hidden_size]
        h_true = torch.stack([pairs[i][1] for i in indices])  # [batch, hidden_size]

        # Forward
        h_pred = head(h_t)  # [batch, hidden_size]

        # MSE loss：hidden state 数值接近
        v_loss = F.mse_loss(h_pred, h_true)

        # KL loss：token 概率分布接近
        with torch.no_grad():
            logits_true = model.lm_head(h_true)  # [batch, vocab_size]
            p_true = F.softmax(logits_true, dim=-1)  # target 分布

        logits_pred = model.lm_head(h_pred)
        p_pred = F.log_softmax(logits_pred, dim=-1)
        p_loss = F.kl_div(p_pred, p_true, reduction="batchmean")

        # 混合 loss
        loss = v_w * v_loss + p_w * p_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == num_steps - 1:
            print(
                f"  Step {step:4d}: loss={loss.item():.4f}, "
                f"v_loss={v_loss.item():.4f}, p_loss={p_loss.item():.4f}"
            )

    return head


def main():
    # 设备
    device = get_device()
    print(f"使用设备: {device}")

    # 模型目录（按你的实际路径调整）
    model_dir = Path("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B").expanduser()
    if not model_dir.exists():
        print(f"模型目录不存在: {model_dir}")
        print("请先运行: python scripts/preflight.py")
        return

    # 加载模型
    print(f"\n加载模型: {model_dir}")
    model = load_causal_lm_from_hf(model_dir)
    model.eval()  # 冻结，不训练
    model = model.to(device)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    # 加载训练数据
    prompts = load_prompts(num_prompts=100, min_length=50)

    # 收集 hidden states
    pairs = collect_hidden_states(
        model,
        prompts,
        tokenizer,
        device,
        max_length=128,
    )

    # 训练
    head = train_eagle_head(
        model,
        pairs,
        device,
        num_steps=500,
        batch_size=32,
    )

    # 保存
    save_path = Path("models/eagle_head.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), save_path)
    print(f"\n✅ 训练完成，保存到 {save_path}")


if __name__ == "__main__":
    main()
