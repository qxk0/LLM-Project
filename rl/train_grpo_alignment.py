"""标准 GRPO baseline(不是 DAPO):online rollout + group relative advantage。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl.alignment.rewards import domain_reward  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="标准 GRPO baseline")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sft-adapter", type=str, default="runs/2026-10-12/models/sft")
    parser.add_argument("--data-file", type=str, default="runs/2026-10-12/data/clean/train.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/alignment_grpo")
    parser.add_argument("--max-prompts", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--ood-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_rows(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_dataset(path: str, max_prompts: int, ood_ratio: float, seed: int):
    rows = load_rows(path)
    ood = [r for r in rows if r.get("ood")]
    ind = [r for r in rows if not r.get("ood")]
    rng = random.Random(seed)
    rng.shuffle(ood)
    rng.shuffle(ind)
    n_ood = min(len(ood), int(max_prompts * ood_ratio))
    n_ind = min(len(ind), max_prompts - n_ood)
    # 若 OOD 不够,保持 OOD 占比,不强行凑满 max_prompts。
    if n_ood and n_ind > int(n_ood * (1 - ood_ratio) / ood_ratio):
        n_ind = int(n_ood * (1 - ood_ratio) / ood_ratio)
    selected = ood[:n_ood] + ind[:n_ind]
    rng.shuffle(selected)
    return selected


def main():
    args = parse_args()
    if args.smoke_test:
        args.max_prompts = min(args.max_prompts, 16)
        args.max_steps = min(args.max_steps, 2)
        args.num_generations = min(args.num_generations, 2)
        args.max_completion_length = min(args.max_completion_length, 48)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = make_dataset(args.data_file, args.max_prompts, args.ood_ratio, args.seed)
    print(f"[grpo] prompts={len(rows)} ood={sum(bool(r.get('ood')) for r in rows)}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(
        [
            {
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["q"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "answer_keywords": row.get("keywords", []),
                "ood": bool(row.get("ood")),
            }
            for row in rows
        ]
    )

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

    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=0.95,
        top_k=50,
        beta=args.beta,
        loss_type="grpo",
        importance_sampling_level="token",
        mask_truncated_completions=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=1 if args.smoke_test else 5,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[domain_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    model.set_adapter("default")
    print(
        f"[grpo] steps={args.max_steps} loss={config.loss_type} "
        f"IS={config.importance_sampling_level} G={args.num_generations}"
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"GRPO 完成: {args.output_dir}")


if __name__ == "__main__":
    main()

