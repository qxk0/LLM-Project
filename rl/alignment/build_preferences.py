"""用同一 SFT 采样 best-of-N,构造 DPO/SimPO 偏好对。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if os.path.join(REPO, "sft") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "sft"))

from eval_domain import build_model  # noqa: E402
from rl.train_dpo import sample_completions_batched  # noqa: E402
from rl.alignment.rewards import score_response  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="构造 on-policy best-of-N 偏好对")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sft-adapter", type=str, default="runs/2026-10-12/models/sft")
    parser.add_argument("--data-file", type=str, default="runs/2026-10-12/data/clean/train.jsonl")
    parser.add_argument("--output", type=str, default="runs/2026-10-12/alignment/pairs.jsonl")
    parser.add_argument("--max-prompts", type=int, default=400)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def stratified_rows(path: str, limit: int, seed: int):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit >= len(rows):
        return rows
    by_intent = {}
    for row in rows:
        by_intent.setdefault(row.get("intent", "unknown"), []).append(row)
    rng = random.Random(seed)
    quotas = {
        intent: max(1, int(limit * len(items) / len(rows)))
        for intent, items in by_intent.items()
    }
    while sum(quotas.values()) > limit:
        intent = max(quotas, key=lambda k: (quotas[k], k))
        if quotas[intent] <= 1:
            break
        quotas[intent] -= 1
    selected, used = [], {}
    for intent, items in by_intent.items():
        rng.shuffle(items)
        take = min(quotas[intent], len(items))
        selected.extend(items[:take])
        used[intent] = take
    if len(selected) < limit:
        pool = [r for intent, items in by_intent.items() for r in items[used[intent]:]]
        rng.shuffle(pool)
        selected.extend(pool[:limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


def main():
    args = parse_args()
    if args.smoke_test:
        args.max_prompts = min(args.max_prompts, 8)
        args.num_samples = min(args.num_samples, 2)
        args.batch_size = min(args.batch_size, 2)
        args.max_new_tokens = min(args.max_new_tokens, 48)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    rows = stratified_rows(args.data_file, args.max_prompts, args.seed)
    print(f"[pairs] prompts={len(rows)} | OOD={sum(bool(r.get('ood')) for r in rows)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(args.model_name, args.sft_adapter)
    model.eval()

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["q"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    completions = sample_completions_batched(
        model,
        tokenizer,
        prompts,
        args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=0.95,
        tag="alignment-pairs",
    )

    pairs, records, skipped = [], [], 0
    for row, prompt, samples in zip(rows, prompts, completions):
        scored = [
            (score_response(sample, row.get("keywords", []), bool(row.get("ood"))), sample)
            for sample in samples
        ]
        best_score, chosen = max(scored, key=lambda x: x[0])
        worst_score, rejected = min(scored, key=lambda x: x[0])
        kept = best_score > worst_score
        records.append(
            {
                "q": row["q"],
                "intent": row.get("intent"),
                "ood": bool(row.get("ood")),
                "keywords": row.get("keywords", []),
                "chosen_score": best_score,
                "rejected_score": worst_score,
                "chosen": chosen,
                "rejected": rejected,
                "kept": kept,
            }
        )
        if not kept:
            skipped += 1
            continue
        pairs.append(
            {
                "prompt": prompt,
                "prompt_messages": [{"role": "user", "content": row["q"]}],
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": best_score,
                "rejected_score": worst_score,
                "intent": row.get("intent"),
                "ood": bool(row.get("ood")),
                "keywords": row.get("keywords", []),
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    stats = {
        "prompts": len(rows),
        "pairs": len(pairs),
        "skipped": skipped,
        "num_samples": args.num_samples,
        "intent_distribution": dict(Counter(p["intent"] for p in pairs)),
        "ood_pairs": sum(1 for p in pairs if p["ood"]),
        "mean_chosen_score": sum(p["chosen_score"] for p in pairs) / max(1, len(pairs)),
        "mean_rejected_score": sum(p["rejected_score"] for p in pairs) / max(1, len(pairs)),
    }
    with open(os.path.splitext(args.output)[0] + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"偏好对已保存: {args.output}")


if __name__ == "__main__":
    main()

