"""SimPO: reference-free, length-normalized preference optimization.

L = -log sigmoid( beta * (logp_w/|w| - logp_l/|l|) - gamma )

使用与 DPO 相同的 best-of-N 偏好对;不加载 reference model。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def parse_args():
    parser = argparse.ArgumentParser(description="SimPO(reference-free)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sft-adapter", type=str, default="runs/2026-10-12/models/sft")
    parser.add_argument("--pairs", type=str, default="runs/2026-10-12/alignment/pairs.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/alignment_simpo")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--sft-weight", type=float, default=0.0,
                        help="可选 chosen NLL 正则;0=纯 SimPO")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_pairs(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_answer(tokenizer, prompt: str, answer: str, max_length: int) -> Tuple[List[int], List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(
        prompt + answer + (tokenizer.eos_token or ""),
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    ).input_ids
    labels = list(full_ids)
    mask_len = min(len(prompt_ids), len(labels))
    labels[:mask_len] = [-100] * mask_len
    return full_ids, labels


def main():
    args = parse_args()
    if args.smoke_test:
        args.max_steps = min(args.max_steps, 2)
        args.grad_accum = min(args.grad_accum, 2)
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_pairs(args.pairs)
    print(f"[simpo] pairs={len(rows)} beta={args.beta} gamma={args.gamma} sft_weight={args.sft_weight}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    model.train()
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    logs: List[Dict[str, float]] = []
    step = 0
    micro = 0
    optimizer.zero_grad()
    while step < args.max_steps:
        random.shuffle(rows)
        for row in rows:
            chosen_ids, chosen_labels = encode_answer(
                tokenizer, row["prompt"], row["chosen"], args.max_length
            )
            rejected_ids, rejected_labels = encode_answer(
                tokenizer, row["prompt"], row["rejected"], args.max_length
            )
            if all(x == -100 for x in chosen_labels) or all(x == -100 for x in rejected_labels):
                continue

            chosen = torch.tensor([chosen_ids], dtype=torch.long, device=model.device)
            chosen_y = torch.tensor([chosen_labels], dtype=torch.long, device=model.device)
            rejected = torch.tensor([rejected_ids], dtype=torch.long, device=model.device)
            rejected_y = torch.tensor([rejected_labels], dtype=torch.long, device=model.device)

            out_w = model(
                input_ids=chosen,
                attention_mask=torch.ones_like(chosen),
                labels=chosen_y,
            )
            out_l = model(
                input_ids=rejected,
                attention_mask=torch.ones_like(rejected),
                labels=rejected_y,
            )
            logp_w = -out_w.loss
            logp_l = -out_l.loss
            margin = args.beta * (logp_w - logp_l) - args.gamma
            loss = -F.logsigmoid(margin)
            if args.sft_weight:
                loss = loss + args.sft_weight * out_w.loss
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                record = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "margin": float(margin.detach().cpu()),
                    "logp_chosen": float(logp_w.detach().cpu()),
                    "logp_rejected": float(logp_l.detach().cpu()),
                }
                logs.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if args.save_steps and step % args.save_steps == 0 and step < args.max_steps:
                    model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{step}"))
                if step >= args.max_steps:
                    break

    if micro % args.grad_accum:
        optimizer.step()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "simpo_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"SimPO 完成: {args.output_dir}")


if __name__ == "__main__":
    main()

