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
