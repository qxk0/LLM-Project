"""快速自检:不依赖数据,验证默认模型和所有结构变体都能前向、反向传播。"""

import torch

from model import GPT, GPTConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    variants = [
        GPTConfig(),  # 默认:学习式位置编码 + MHA + 共享词嵌入
        GPTConfig(pos_encoding="rope", attention_type="gqa", num_kv_heads=2),
        GPTConfig(pos_encoding="alibi", attention_type="mqa", tie_embeddings=False),
        GPTConfig(pos_encoding="rope", activation="swiglu", norm_type="rmsnorm"),
    ]
    for cfg in variants:
        model = GPT(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randint(0, cfg.vocab_size, (2, 64), device=device)
        logits, loss = model(x, x)
        loss.backward()
        print(
            f"OK [{cfg.pos_encoding}/{cfg.attention_type}/{cfg.activation}/{cfg.norm_type}] "
            f"参数 {n_params / 1e6:.2f}M,loss = {loss.item():.4f}"
        )
    if device == "cuda":
        print(f"显存占用: {torch.cuda.max_memory_allocated() / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main()
