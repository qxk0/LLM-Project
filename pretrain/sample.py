"""用训练好的模型生成小故事(推理演示)。"""

import argparse
import os

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description="生成小故事")
    parser.add_argument("--ckpt", type=str, default="models/pretrain/best.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model_cfg = GPTConfig(**ckpt["config"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer_path = ckpt["config"].get("tokenizer_path") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt))), "tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)

    ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
        )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
