# 预训练模块 pretrain/ (SwiGLU + RMSNorm + FlashAttention)

> 生成时间：2026-10-14 17:27:04

## 目录结构

```text
pretrain/
  build_data.py
  config.py
  eval_gen.py
  model.py
  plot_results.py
  sample.py
  smoke_test.py
  train.py
```

## 文件内容

### `pretrain/build_data.py`

```python
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
```

### `pretrain/config.py`

```python
"""预训练配置:所有超参数集中在这里,改配置不用改代码。"""

from dataclasses import dataclass


@dataclass
class PretrainConfig:
    # ---------- 数据 ----------
    dataset_name: str = "roneneldan/TinyStories"  # HuggingFace 上的小故事数据集
    data_split: str = "train"                     # 使用训练集
    max_stories: int = 100_000                    # 取前多少篇故事(控制训练规模)
    tokenizer_sample: int = 40_000                # 用多少篇故事训练 tokenizer
    data_dir: str = "data/pretrain"               # 处理后的数据保存目录

    # ---------- Tokenizer ----------
    vocab_size: int = 4096                        # BPE 词表大小(微型模型够用)
    min_frequency: int = 2                        # 词条至少出现几次才保留

    # ---------- 模型 ----------
    n_layer: int = 6                              # Transformer 层数
    n_head: int = 6                               # 注意力头数
    n_embd: int = 384                             # 隐藏维度
    block_size: int = 256                         # 上下文长度(一次看多少个 token)
    dropout: float = 0.1                          # 随机失活,防止过拟合
    # ---------- 结构消融(实验用) ----------
    pos_encoding: str = "learned"                 # learned | rope | alibi
    attention_type: str = "mha"                   # mha | mqa | gqa
    num_kv_heads: int = 0                         # GQA 的 KV 头数(0=自动取一半)
    tie_embeddings: bool = True                   # 是否共享输入/输出词嵌入
    activation: str = "gelu"                       # gelu | swiglu(Llama/Qwen 同款)
    norm_type: str = "layernorm"                   # layernorm | rmsnorm(Llama/Qwen 同款)

    # ---------- 训练 ----------
    batch_size: int = 32                          # 每个 batch 的样本数
    grad_accum: int = 2                           # 梯度累积步数,等效 batch 翻倍
    max_steps: int = 6000                         # 总训练步数
    lr: float = 3e-4                              # 最大学习率
    warmup_steps: int = 200                       # 学习率预热步数
    weight_decay: float = 0.1                     # 权重衰减(正则化)
    grad_clip: float = 1.0                        # 梯度裁剪,防止梯度爆炸
    fp16: bool = True                             # 混合精度训练(省显存、加速)
    seed: int = 1337

    # ---------- 日志 / 保存 ----------
    log_interval: int = 50                        # 每多少步打印一次 loss
    eval_interval: int = 250                      # 每多少步做一次验证
    eval_iters: int = 20                          # 验证时取几个 batch 求平均
    eval_tokens: int = 500_000                    # 预留多少 token 做验证(不参与训练)
    sample_interval: int = 250                    # 每多少步生成一段样本文本
    save_interval: int = 1000                     # 每多少步保存一次检查点
    out_dir: str = "models/pretrain"              # 模型保存目录
```

### `pretrain/eval_gen.py`

```python
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
```

### `pretrain/model.py`

```python
"""微型 GPT 模型(decoder-only Transformer)。

结构参考 Andrej Karpathy 的 nanoGPT,改成了教学友好的单文件版本:
  token embedding + position embedding
  -> N 个 Transformer Block(因果自注意力 + 前馈网络)
  -> LayerNorm -> 输出头(与 token embedding 共享权重)

参数量约 1200 万,单张 4GB 显存的显卡可以轻松训练。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    pos_encoding: str = "learned"     # learned | rope | alibi
    attention_type: str = "mha"       # mha | mqa | gqa
    num_kv_heads: int = 0             # GQA 的 KV 头数,0=自动取一半
    tie_embeddings: bool = True       # 输入/输出词嵌入是否共享
    activation: str = "gelu"          # gelu | swiglu(Llama/Qwen 同款)
    norm_type: str = "layernorm"      # layernorm | rmsnorm(Llama/Qwen 同款)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """RMSNorm:去掉均值中心化只做缩放(Llama/Qwen 同款,计算量更小)。"""

    def __init__(self, ndim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps
        return x / rms * self.weight


class CausalSelfAttention(nn.Module):
    """因果自注意力,支持 MHA / MQA / GQA 三种头配置,
    以及 RoPE / ALiBi / 学习式三种位置编码(消融实验用)。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.pos_encoding = config.pos_encoding

        # KV 头数:MHA=全部头,MQA=1,GQA=一半(或手动指定)
        if config.attention_type == "mqa":
            self.n_kv_heads = 1
        elif config.attention_type == "gqa":
            self.n_kv_heads = config.num_kv_heads or max(1, config.n_head // 2)
        else:
            self.n_kv_heads = config.n_head
        assert config.n_head % self.n_kv_heads == 0
        self.n_rep = config.n_head // self.n_kv_heads

        # QKV 一次算出;MQA/GQA 时 K、V 的头数更少,省显存省计算
        self.c_attn = nn.Linear(
            config.n_embd, (config.n_head + 2 * self.n_kv_heads) * self.head_dim, bias=False
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码:下三角为 1,保证训练时不会"偷看未来"
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("bias", mask)

        # RoPE:预计算每个位置的 cos/sin(旋转角度)
        if config.pos_encoding == "rope":
            inv_freq = 1.0 / (
                10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
            )
            t = torch.arange(config.block_size).float()
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
            self.register_buffer("cos", emb.cos())
            self.register_buffer("sin", emb.sin())
        # ALiBi:每个注意力头一个固定斜率,位置越远惩罚越大
        elif config.pos_encoding == "alibi":
            slopes = torch.tensor(
                [2.0 ** (-8 * (i + 1) / self.n_head) for i in range(self.n_head)]
            )
            dist = torch.arange(config.block_size).view(1, 1, config.block_size, 1) - (
                torch.arange(config.block_size).view(1, 1, 1, config.block_size)
            )
            self.register_buffer("alibi_bias", -slopes.view(1, self.n_head, 1, 1) * dist)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q = qkv[:, :, : self.n_head * self.head_dim]
        k = qkv[
            :, :, self.n_head * self.head_dim : (self.n_head + self.n_kv_heads) * self.head_dim
        ]
        v = qkv[:, :, (self.n_head + self.n_kv_heads) * self.head_dim :]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 旋转位置编码只作用于 Q、K,不影响 V
        if self.pos_encoding == "rope":
            q = self._apply_rope(q, self.cos[:T], self.sin[:T])
            k = self._apply_rope(k, self.cos[:T], self.sin[:T])

        # GQA/MQA:把 KV 头复制成和 Q 一样多
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if self.pos_encoding == "alibi":
            # ALiBi 需要自定义偏置,走手写路径
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att + self.alibi_bias[:, :, :T, :T]
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        else:
            # PyTorch 内置 Flash Attention(SDPA),显存 O(T) 且更快
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

    @staticmethod
    def _apply_rope(x, cos, sin):
        """RoPE:把相邻两个分量看成一个复数对,旋转 theta 角。"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class MLP(nn.Module):
    """前馈网络:GELU 版(线性->GELU->线性)或 SwiGLU 版(门控线性单元)。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.activation == "swiglu":
            # SwiGLU 有两个投影(gate+up),用 8/3 倍宽度使参数量与 GELU(4x) 持平
            swiglu_width = int(4 * config.n_embd * 2 / 3)
            self.gate_proj = nn.Linear(config.n_embd, swiglu_width, bias=False)
            self.up_proj = nn.Linear(config.n_embd, swiglu_width, bias=False)
            self.down_proj = nn.Linear(swiglu_width, config.n_embd, bias=False)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if hasattr(self, "gate_proj"):
            # SwiGLU: Swish(xW_gate) * (xW_up),信息经门控选择性通过
            return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """一个 Transformer 层 = 注意力 + 前馈,各自带残差连接。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm
        self.ln_1 = norm_cls(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = norm_cls(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # 学习式位置编码才需要位置表;RoPE/ALiBi 不需要
        if config.pos_encoding == "learned":
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.position_embedding = None
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm
        self.ln_f = norm_cls(config.n_embd)

        # 输出头;tie_embeddings=True 时与输入 embedding 共享权重(参数减半、训练更稳)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "输入长度超过 block_size"

        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            pos = self.position_embedding(torch.arange(T, device=idx.device))
            x = x + pos
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 每个位置预测下一个 token,所以取交叉熵
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成:每步把新 token 拼回输入,再预测下一个。"""
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

### `pretrain/plot_results.py`

```python
"""把实验日志画成图:训练/验证 loss 曲线、参数缩放曲线。

用法:
  python pretrain/plot_results.py --log models/exp_a/log.jsonl --log models/exp_b/log.jsonl
  python pretrain/plot_results.py --scaling models/*/log.jsonl
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def plot_curves(log_paths, out="models/pretrain/loss_curves.png"):
    plt.figure(figsize=(8, 5))
    for path in log_paths:
        events = load_events(path)
        steps = [e["step"] for e in events if e["type"] == "train"]
        losses = [e["loss"] for e in events if e["type"] == "train"]
        if not steps:
            continue
        label = os.path.basename(os.path.dirname(path))
        plt.plot(steps, losses, label=label, lw=1.5)
        val_steps = [e["step"] for e in events if e["type"] == "eval"]
        val_losses = [e["val_loss"] for e in events if e["type"] == "eval"]
        if val_steps:
            plt.plot(val_steps, val_losses, "o--", alpha=0.6)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.title("训练/验证 loss 曲线")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"曲线图已保存: {out}")


def plot_scaling(log_paths, out="models/pretrain/scaling.png"):
    """横轴参数量、纵轴最终验证 loss,看缩放规律。"""
    points = []
    for path in log_paths:
        events = load_events(path)
        meta = next((e for e in events if e["type"] == "meta"), None)
        evals = [e for e in events if e["type"] == "eval"]
        if not meta or not evals:
            continue
        points.append((meta["n_params"], evals[-1]["val_loss"], path))
    if not points:
        print("没有找到带 meta 的日志,跳过缩放图")
        return
    points.sort()
    plt.figure(figsize=(8, 5))
    plt.plot([p[0] / 1e6 for p in points], [p[1] for p in points], "o-")
    for x, y, p in points:
        label = os.path.basename(os.path.dirname(p))
        plt.annotate(label, (x / 1e6, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
    plt.xlabel("参数量 (M)")
    plt.ylabel("最终验证 loss")
    plt.title("缩放定律(小规模复现)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"缩放曲线已保存: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", default=[], help="日志文件,可多次指定")
    parser.add_argument("--scaling", action="store_true", help="画缩放曲线(需要多个实验)")
    parser.add_argument("--out", type=str, default="models/pretrain/loss_curves.png")
    args = parser.parse_args()

    if args.scaling:
        paths = sorted(glob.glob("models/exp_*/log.jsonl"))
        plot_scaling(paths, out=args.out.replace(".png", "_scaling.png"))
    elif args.log:
        plot_curves(args.log, out=args.out)
    else:
        parser.error("请用 --log 指定日志文件,或用 --scaling 画缩放曲线")


if __name__ == "__main__":
    main()
```

### `pretrain/sample.py`

```python
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
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
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
```

### `pretrain/smoke_test.py`

```python
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
```

### `pretrain/train.py`

```python
"""微型 GPT 预训练脚本。

流程:
1. 加载 tokenizer 和 train.bin(内存映射,不占内存)
2. 搭建千万级参数的 GPT
3. AdamW + 余弦退火学习率训练
4. 每 N 步打印 loss、验证 loss、生成一段样本文本、保存检查点

训练结束后 models/pretrain/ 里会有:
  tokenizer.json   词表
  ckpt_XXXX.pt     阶段检查点(模型 + 优化器状态)
  best.pt          验证 loss 最低的检查点
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from config import PretrainConfig
from model import GPT, GPTConfig


def get_batch(data, start, end, block_size, batch_size, device):
    """从 [start, end) 区间随机取一个 batch,每个样本是长度为 block_size 的连续切片。"""
    ix = torch.randint(end - block_size - 1 - start, (batch_size,)) + start
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix.tolist()]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix.tolist()]
    )
    return x.to(device), y.to(device)


def configure_optimizer(model, weight_decay, lr, betas=(0.9, 0.95)):
    """nanoGPT 同款优化器:二维以上参数做权重衰减,偏置/归一化参数不做。"""
    param_dict = {pn: p for pn, p in model.named_parameters()}  # 去重(共享权重只算一次)
    decay = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay = [p for p in param_dict.values() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def get_lr(step, max_steps, lr, warmup_steps):
    """学习率:先线性预热,再按余弦曲线降到接近 0(训练更稳)。"""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, data, start, end, block_size, batch_size, device, eval_iters):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, start, end, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def generate_sample(model, tokenizer, device, prompt="Once upon a time,", max_new=120):
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new, temperature=0.8, top_k=40)
    return tokenizer.decode(out[0].tolist())


def save_checkpoint(path, model, optimizer, step, model_config):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": model_config.__dict__,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="微型 GPT 预训练")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None, help="模型层数(实验用)")
    parser.add_argument("--n-head", type=int, default=None, help="注意力头数(实验用)")
    parser.add_argument("--n-embd", type=int, default=None, help="隐藏维度(实验用)")
    parser.add_argument("--pos-encoding", type=str, default=None, choices=["learned", "rope", "alibi"])
    parser.add_argument("--attention-type", type=str, default=None, choices=["mha", "mqa", "gqa"])
    parser.add_argument("--tie-embeddings", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--activation", type=str, default=None, choices=["gelu", "swiglu"])
    parser.add_argument("--norm-type", type=str, default=None, choices=["layernorm", "rmsnorm"])
    parser.add_argument("--out-dir", type=str, default=None, help="实验输出目录")
    parser.add_argument("--resume", type=str, default=None, help="从检查点继续训练")
    args = parser.parse_args()

    cfg = PretrainConfig()
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.vocab_size:
        cfg.vocab_size = args.vocab_size
    if args.n_layer:
        cfg.n_layer = args.n_layer
    if args.n_head:
        cfg.n_head = args.n_head
    if args.n_embd:
        cfg.n_embd = args.n_embd
    if args.pos_encoding:
        cfg.pos_encoding = args.pos_encoding
    if args.attention_type:
        cfg.attention_type = args.attention_type
    if args.tie_embeddings:
        cfg.tie_embeddings = args.tie_embeddings == "true"
    if args.activation:
        cfg.activation = args.activation
    if args.norm_type:
        cfg.norm_type = args.norm_type
    if args.out_dir:
        cfg.out_dir = args.out_dir

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 数据与 tokenizer
    meta_path = os.path.join(cfg.data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"找不到 {meta_path},请先运行: python pretrain/build_data.py"
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    data = np.memmap(os.path.join(cfg.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    tokenizer = Tokenizer.from_file(meta["tokenizer_path"])
    eot_id = meta["eot_id"]

    total_tokens = int(meta["total_tokens"])
    eval_start = total_tokens - cfg.eval_tokens
    assert eval_start > cfg.block_size + 10, "数据量太小,请增加 max_stories"

    # 模型
    model_cfg = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        pos_encoding=cfg.pos_encoding,
        attention_type=cfg.attention_type,
        num_kv_heads=cfg.num_kv_heads,
        tie_embeddings=cfg.tie_embeddings,
        activation=cfg.activation,
        norm_type=cfg.norm_type,
    )
    model = GPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    optimizer = configure_optimizer(model, cfg.weight_decay, cfg.lr)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"已从 {args.resume} 恢复,继续第 {start_step} 步")

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device == "cuda")
    use_amp = cfg.fp16 and device == "cuda"

    os.makedirs(cfg.out_dir, exist_ok=True)
    # 实验日志:每行一个 JSON 事件,供绘图/分析脚本读取
    log_path = os.path.join(cfg.out_dir, "log.jsonl")
    log_file = open(log_path, "a", encoding="utf-8")

    def log_event(event):
        log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        log_file.flush()

    log_event(
        {
            "type": "meta",
            "n_params": n_params,
            "config": cfg.__dict__,
        }
    )
    best_val = float("inf")

    print(f"开始训练: {cfg.max_steps} 步,等效 batch = {cfg.batch_size * cfg.grad_accum} * {cfg.block_size} tokens")
    model.train()
    t0 = time.time()
    tokens_seen = 0

    for step in range(start_step, cfg.max_steps):
        # 梯度累积:多次前向/反向后再更新一次参数
        for _ in range(cfg.grad_accum):
            x, y = get_batch(data, 0, eval_start, cfg.block_size, cfg.batch_size, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # 梯度裁剪 + 参数更新
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        lr = get_lr(step, cfg.max_steps, cfg.lr, cfg.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        tokens_seen += cfg.batch_size * cfg.block_size * cfg.grad_accum

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            cur_loss = loss.item() * cfg.grad_accum
            print(
                f"step {step:5d} | loss {loss.item() * cfg.grad_accum:.4f} | "
                f"lr {lr:.2e} | {tokens_seen / dt / 1000:.0f}k tok/s"
            )
            log_event(
                {
                    "type": "train",
                    "step": step,
                    "loss": cur_loss,
                    "lr": lr,
                    "tok_per_s": tokens_seen / dt,
                }
            )

        if step > 0 and step % cfg.eval_interval == 0:
            train_loss = estimate_loss(
                model, data, 0, eval_start, cfg.block_size, cfg.batch_size, device, cfg.eval_iters
            )
            val_loss = estimate_loss(
                model, data, eval_start, total_tokens, cfg.block_size, cfg.batch_size, device, cfg.eval_iters
            )
            print(f"  [eval] train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
            log_event(
                {
                    "type": "eval",
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    os.path.join(cfg.out_dir, "best.pt"), model, optimizer, step, model_cfg
                )
                print(f"  [eval] 新最佳,已保存 best.pt (val_loss {val_loss:.4f})")

        if step > 0 and step % cfg.sample_interval == 0:
            text = generate_sample(model, tokenizer, device)
            print(f"  [sample] {text[:300]}")
            log_event({"type": "sample", "step": step, "text": text})

        if step > 0 and step % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, model_cfg)
            print(f"  已保存检查点: {ckpt_path}")

    # 结束时保存最终检查点
    final_path = os.path.join(cfg.out_dir, f"ckpt_{cfg.max_steps}.pt")
    save_checkpoint(final_path, model, optimizer, cfg.max_steps - 1, model_cfg)
    log_file.close()
    print(f"训练完成!总用时 {(time.time() - t0) / 60:.1f} 分钟,最终检查点: {final_path}")


if __name__ == "__main__":
    main()
```
