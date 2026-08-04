"""生成质量评估:输出若干条故事,并计算 token 多样性指标(distinct-1/2)。

loss 只能反映"预测得多准",生成质量需要单独看:
  - 通顺程度(人看)
  - 多样性:distinct-n = 不重复 n-gram 数 / 总 n-gram 数,越高越不单调
"""

import argparse
import os

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return len(ngrams) / (len(tokens) - n + 1)


def main():
    parser = argparse.ArgumentParser(description="生成质量评估")
    parser.add_argument("--ckpt", type=str, default="models/pretrain/best.pt")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--save", type=str, default="models/pretrain/samples.md")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    prompt_ids = tokenizer.encode(args.prompt).ids

    all_tokens = []
    samples = []
    with torch.no_grad():
        for i in range(args.num_samples):
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            out = model.generate(
                idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
            )
            text = tokenizer.decode(out[0].tolist())
            samples.append(text)
            all_tokens.extend(out[0][len(prompt_ids) :].tolist())
            print(f"[{i + 1}] {text[:150]}\n")

    d1 = distinct_n(all_tokens, 1)
    d2 = distinct_n(all_tokens, 2)
    print(f"=== 多样性指标(distinct) ===")
    print(f"distinct-1: {d1:.3f} | distinct-2: {d2:.3f} | 生成 token 总数: {len(all_tokens)}")

    with open(args.save, "w", encoding="utf-8") as f:
        f.write(f"# 生成样本(prompt: {args.prompt})\n\n")
        for i, s in enumerate(samples, 1):
            f.write(f"## 样本 {i}\n\n{s}\n\n")
        f.write(f"distinct-1: {d1:.3f} | distinct-2: {d2:.3f}\n")
    print(f"样本已保存: {args.save}")


if __name__ == "__main__":
    main()
