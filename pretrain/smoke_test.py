"""快速自检:不依赖数据,只验证模型能前向、反向传播,GPU 能识别。"""

import torch

from model import GPT, GPTConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f}M")

    x = torch.randint(0, cfg.vocab_size, (2, 64), device=device)
    logits, loss = model(x, x)
    loss.backward()

    print(f"前向/反向 OK,logits 形状 {tuple(logits.shape)},loss = {loss.item():.4f}")
    if device == "cuda":
        print(f"显存占用: {torch.cuda.max_memory_allocated() / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main()
