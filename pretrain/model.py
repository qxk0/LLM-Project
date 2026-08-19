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
