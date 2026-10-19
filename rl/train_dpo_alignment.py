"""标准 DPO(sigmoid + reference model),使用 best-of-N 偏好对。"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def parse_args():
    parser = argparse.ArgumentParser(description="标准 DPO(best-of-N 偏好对)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sft-adapter", type=str, default="runs/2026-10-12/models/sft")
    parser.add_argument("--pairs", type=str, default="runs/2026-10-12/alignment/pairs.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/alignment_dpo")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        args.max_steps = min(args.max_steps, 2)
        args.grad_accum = min(args.grad_accum, 2)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset("json", data_files=args.pairs, split="train")
    print(f"[dpo] pairs={len(dataset)}")
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
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model.config.use_cache = False

    config = DPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        loss_type="sigmoid",
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=1 if args.smoke_test else 10,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    # TRL 为 PEFT 自动创建 ref adapter 后,活动 adapter 必须切回 default。
    model.set_adapter("default")
    print(f"[dpo] steps={args.max_steps} beta={args.beta} loss=sigmoid")
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"DPO 完成: {args.output_dir}")


if __name__ == "__main__":
    main()

