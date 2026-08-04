"""训练 BPE tokenizer 并生成预训练数据文件。

(文件名特意不叫 tokenize.py,避免与 Python 标准库的 tokenize 模块重名)

做什么:
1. 从 HuggingFace 下载 TinyStories 小故事数据集(首次运行自动下载,直连官方源)
2. 用一部分故事训练一个字节级 BPE tokenizer(GPT-2 同款方案)
3. 把故事切成 token,拼成一个 uint16 大数组存成 train.bin
   (训练时用内存映射读取,快且省内存,nanoGPT 同款格式)

为什么自己训 tokenizer:
  真实工作中"数据处理"是核心环节之一。简历上写"训练了自定义 BPE tokenizer"
  比"用了现成 tokenizer"更能体现基本功。
"""

import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

from config import PretrainConfig


def main():
    parser = argparse.ArgumentParser(description="训练 tokenizer 并生成训练数据")
    parser.add_argument("--max-stories", type=int, default=None, help="取多少篇故事")
    parser.add_argument("--vocab-size", type=int, default=None, help="词表大小")
    args = parser.parse_args()

    cfg = PretrainConfig()
    if args.max_stories:
        cfg.max_stories = args.max_stories
    if args.vocab_size:
        cfg.vocab_size = args.vocab_size

    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1. 下载数据集。streaming 模式按需取数据,不会一次性占满磁盘。
    print(f"正在加载数据集 {cfg.dataset_name} (首次运行会下载)...")
    ds = load_dataset(cfg.dataset_name, split=cfg.data_split, streaming=True)
    iterator = iter(ds)

    # 把需要的故事先读进内存(100k 篇约 100~150MB,可以接受)
    texts = []
    for _ in range(cfg.max_stories):
        texts.append(next(iterator)["text"])
    print(f"已读取 {len(texts)} 篇故事")

    # 2. 训练 BPE tokenizer
    print(f"用前 {cfg.tokenizer_sample} 篇故事训练 BPE tokenizer(词表 {cfg.vocab_size})...")
    tokenizer = ByteLevelBPETokenizer()
    # 新版 tokenizers(0.22+)直接在 train_from_iterator 里传训练参数
    tokenizer.train_from_iterator(
        texts[: cfg.tokenizer_sample],
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=["<|endoftext|>"],  # 故事结束符,训练/生成时用来隔开故事
    )

    tokenizer_path = os.path.join(cfg.out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"tokenizer 已保存: {tokenizer_path} (结束符 id={eot_id})")

    # 3. 把所有故事切成 token,拼成一个大数组
    print(f"正在切分 {len(texts)} 篇故事为 token...")
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(np.array(tokenizer.encode(text).ids, dtype=np.uint16))
        chunks.append(np.array([eot_id], dtype=np.uint16))
        if (i + 1) % 20000 == 0:
            print(f"  已处理 {i + 1} 篇")
    flat = np.concatenate(chunks)

    bin_path = os.path.join(cfg.data_dir, "train.bin")
    flat.tofile(bin_path)

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "total_tokens": int(flat.size),
        "num_stories": len(texts),
        "eot_id": eot_id,
        "tokenizer_path": tokenizer_path,
    }
    with open(os.path.join(cfg.data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"完成!共 {flat.size / 1e6:.1f}M tokens,{len(texts)} 篇故事")
    print(f"数据文件: {bin_path}")


if __name__ == "__main__":
    main()
