"""多轮工具客服 SFT。

数据格式为 `sft_*.jsonl` 的 messages:
  system / user / assistant(JSON action) / tool / user / assistant ...

只对 assistant 的 JSON 动作和最终回答计算 loss,user/tool token 全部 -100。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl.multiturn.formatting import tokenize_multiturn_example  # noqa: E402
from rl.multiturn.modeling import (  # noqa: E402
    add_lora,
    load_model_and_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="多轮工具客服 SFT")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-file", type=str, default="data/multiturn/sft_train.jsonl")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--device-map", choices=["auto", "cuda"], default="cuda")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    return parser.parse_args()


def apply_smoke_defaults(args):
    if not args.smoke_test:
        return args
    if args.max_examples == 200:
        args.max_examples = 12
    if args.max_steps == 30:
        args.max_steps = 5
    if args.max_seq_len == 768:
        args.max_seq_len = 512
    if args.lora_r == 16:
        args.lora_r = 8
    if args.lora_alpha == 32:
        args.lora_alpha = 16
    if args.grad_accum == 8:
        args.grad_accum = 4
    if args.output_dir is None:
        args.output_dir = "models/sft_multiturn_smoke"
    return args


def main():
    args = apply_smoke_defaults(parse_args())
    if args.output_dir is None:
        args.output_dir = "models/sft_multiturn"
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    model, tokenizer, is_qwen3 = load_model_and_tokenizer(
        args.model_name,
        adapter=None,
        for_training=True,
        precision=args.precision,
        device_map=args.device_map,
    )
    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    model.config.use_cache = False

    raw = load_dataset("json", data_files=args.data_file, split="train")
    n = min(args.max_examples, len(raw))
    data = raw.select(range(n))

    def tokenize_fn(ex):
        return tokenize_multiturn_example(
            tokenizer,
            ex["messages"],
            max_len=args.max_seq_len,
            enable_thinking=args.enable_thinking,
        )

    dataset = data.map(tokenize_fn, remove_columns=data.column_names)
    sample = dataset[0]
    print(
        f"[data] rows={len(dataset)} | first assistant_tokens={sample['assistant_tokens']} "
        f"| first seq_len={len(sample['input_ids'])} | qwen3={is_qwen3}"
    )

    use_bf16 = args.precision == "bf16" or (args.precision == "auto" and is_qwen3)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=max(1, int(0.03 * args.max_steps)),
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim=args.optim,
        logging_steps=1 if args.smoke_test else 10,
        logging_first_step=True,
        save_steps=5 if args.smoke_test else 10,
        save_total_limit=1 if args.smoke_test else 2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        processing_class=tokenizer,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(
        f"开始多轮 SFT: steps={args.max_steps} | seq_len={args.max_seq_len} "
        f"| batch={args.batch_size} | grad_accum={args.grad_accum} "
        f"| precision={'bf16' if use_bf16 else 'fp16'}"
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    if torch.cuda.is_available():
        print(
            f"[vram] peak allocated={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB "
            f"| reserved={torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB"
        )
    print(f"多轮 SFT 完成: {args.output_dir}")


if __name__ == "__main__":
    main()

