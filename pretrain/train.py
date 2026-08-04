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
    )
    model = GPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    optimizer = configure_optimizer(model, cfg.weight_decay, cfg.lr)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
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
