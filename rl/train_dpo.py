"""DPO(Direct Preference Optimization)对齐训练。

这是对齐技术演进线中的第二环:SFT -> DPO -> GRPO。
DPO 不需要奖励模型和 RL 采样,直接用"偏好对"做二分类式训练,
数学上等价于 RLHF,但简单得多。

偏好对怎么来(本脚本的核心):
  1. 对每条领域问题,让 SFT 模型采样 4 个回答
  2. 用与 GRPO 相同的规则奖励打分(答对/拒答 +1,否则 -1;简洁 +0.1)
  3. 得分最高的当 chosen,最低的当 rejected —— 自动构造偏好数据集
训练产物:models/dpo/ 下的 LoRA 适配器(与 SFT/GRPO 同评测集对比)。
"""

import argparse
import json
import os
import random
import time
from collections import Counter

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

REFUSAL_MARKERS = ["抱歉", "不好意思", "无法回答", "超出", "服务范围", "咨询"]


def parse_args():
    parser = argparse.ArgumentParser(description="DPO 对齐训练(规则奖励构造偏好对)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", type=str, default="models/sft")
    parser.add_argument("--data-file", type=str, default="data_eng/output/train.jsonl")
    parser.add_argument("--max-pairs", type=int, default=400, help="最多构造多少偏好对")
    parser.add_argument("--num-samples", type=int, default=4, help="每题采样几个回答")
    parser.add_argument("--pairs-batch-size", type=int, default=8,
                        help="构造偏好对时的批量采样 batch(只影响吞吐,不影响采样参数)")
    parser.add_argument("--pairs-out", type=str, default=None,
                        help="把偏好对与每题打分落盘成 JSONL(便于复现与审计)")
    parser.add_argument("--seed", type=int, default=42, help="分层抽样与训练的随机种子")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=str, default="models/dpo")
    return parser.parse_args()


def load_rows(path, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
    return rows


def load_rows_stratified(path, limit, seed=42):
    """按 intent 分层抽样,保证每个意图(尤其 OOD 拒答)都有覆盖。

    ⚠️ 修复点:旧实现 load_rows() 只取文件"前 N 行",而训练数据是按 intent 分块排布的,
    导致 RFT / DPO 实际只看到 price(recommend 后来才出现)、**完全看不到 OOD 拒答样本**,
    因此偏好对/RFT 数据无法覆盖"该拒答"这一核心场景。
    """
    all_rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))
    if limit >= len(all_rows):
        return all_rows
    by_intent = {}
    for r in all_rows:
        by_intent.setdefault(r.get("intent", "unknown"), []).append(r)
    rng = random.Random(seed)
    quotas = {i: max(1, int(limit * len(v) / len(all_rows))) for i, v in by_intent.items()}
    while sum(quotas.values()) > limit:
        i = max(quotas, key=lambda x: (quotas[x], x))
        if quotas[i] <= 1:
            break
        quotas[i] -= 1
    out, used = [], {}
    for i, v in by_intent.items():
        rng.shuffle(v)
        take = min(quotas[i], len(v))
        out.extend(v[:take])
        used[i] = take
    remainder = limit - len(out)
    if remainder > 0:
        pool = [r for i, v in by_intent.items() for r in v[used[i]:]]
        rng.shuffle(pool)
        out.extend(pool[:remainder])
    rng.shuffle(out)
    return out[:limit]


def sample_completions_batched(model, tokenizer, prompts, num_samples, batch_size=8,
                               max_new_tokens=64, temperature=0.8, top_p=0.95,
                               progress_every=50, tag="sample"):
    """左 padding 批量采样:每个 prompt 采样 num_samples 条。

    只改变吞吐(批量),**采样参数与旧实现完全一致**(temperature=0.8 / top_p=0.95 /
    num_return_sequences=num_samples / max_new_tokens=64)。
    """
    old_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    all_completions = []
    t0 = time.time()
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_samples,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = out[:, enc.input_ids.shape[1]:]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i in range(len(chunk)):
                all_completions.append(
                    [t.strip() for t in texts[i * num_samples:(i + 1) * num_samples]]
                )
            done = min(start + batch_size, len(prompts))
            if progress_every and done % progress_every == 0:
                print(f"  [{tag}] {done}/{len(prompts)} | {time.time() - t0:.0f}s", flush=True)
    finally:
        tokenizer.padding_side = old_side
    return all_completions


def score_response(comp, keywords, is_ood):
    """与 GRPO 相同的规则奖励:答对/拒答 +1,否则 -1;简洁 +0.1。"""
    score = 0.0
    if is_ood:
        score += 1.0 if any(m in comp for m in REFUSAL_MARKERS) else -1.0
    else:
        score += 1.0 if any(k in comp for k in keywords) else -1.0
    score += 0.1 if len(comp.strip()) <= 80 else 0.0
    return score


def build_preference_pairs(model, tokenizer, rows, num_samples, batch_size=8):
    """采样 + 规则奖励打分 -> chosen/rejected 偏好对。

    筛选逻辑与旧实现完全一致(chosen=知识库 gold;rejected=采样中规则分最低者;
    仅当 worst_score < gold_score 时保留),只把"逐题串行采样"改为批量采样,
    并额外记录每题的采样打分,便于事后审计。
    """
    completions_per_prompt = sample_completions_batched(
        model, tokenizer, [r["q"] for r in rows], num_samples,
        batch_size=batch_size, tag="dpo-pairs",
    )
    pairs, records, skipped = [], [], 0
    for r, completions in zip(rows, completions_per_prompt):
        gold_score = score_response(r["a"], r["keywords"], r["ood"])
        scored = [(score_response(c, r["keywords"], r["ood"]), c) for c in completions]
        worst_score, rejected = min(scored)
        kept = worst_score < gold_score
        records.append(
            {
                "prompt": r["q"],
                "gold_answer": r["a"],
                "intent": r.get("intent"),
                "ood": bool(r.get("ood")),
                "keywords": r.get("keywords", []),
                "gold_score": gold_score,
                "worst_score": worst_score,
                "sampled_scores": [s for s, _ in scored],
                "rejected": rejected,
                "kept": kept,
            }
        )
        if kept:
            pairs.append({"prompt": r["q"], "chosen": r["a"], "rejected": rejected})
        else:
            skipped += 1
    stats = {
        "rows": len(rows),
        "num_samples": num_samples,
        "pairs": len(pairs),
        "skipped": skipped,
        "intent_distribution": dict(Counter(r.get("intent") for r in rows)),
        "ood_rows": sum(1 for r in rows if r.get("ood")),
    }
    print(
        f"偏好对总数:{len(pairs)} | 因分数相同跳过:{skipped} | "
        f"覆盖意图:{list(stats['intent_distribution'])} | OOD 题数:{stats['ood_rows']}"
    )
    return pairs, records, stats


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = load_rows_stratified(args.data_file, args.max_pairs, seed=args.seed)
    print(f"加载 {len(rows)} 条领域数据(按 intent 分层抽样,保证 OOD 覆盖)")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"加载基座 {args.model_name} + SFT 适配器 {args.adapter} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto", local_files_only=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print(f"构造偏好对:每题采样 {args.num_samples} 个回答(批量采样),用规则奖励打分...")
    pairs, pair_records, pair_stats = build_preference_pairs(
        model, tokenizer, rows, args.num_samples, batch_size=args.pairs_batch_size
    )
    dataset = Dataset.from_list(pairs)
    if args.pairs_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.pairs_out)), exist_ok=True)
        with open(args.pairs_out, "w", encoding="utf-8") as f:
            for rec in pair_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(os.path.splitext(args.pairs_out)[0] + "_stats.json", "w", encoding="utf-8") as f:
            json.dump(pair_stats, f, ensure_ascii=False, indent=2)
        print(f"偏好对已落盘:{args.pairs_out}({len(pair_records)} 条逐题记录)")

    # ---------- DPO 训练 ----------
    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        beta=0.1,  # DPO 的 KL 强度
        max_length=128,
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    # 同 GRPO 的坑:ref 适配器创建后活动适配器会变成 ref,必须切回 default
    model.set_adapter("default")

    print(f"开始 DPO 训练:{args.max_steps} 步,beta={dpo_args.beta}")
    trainer.train()

    if "ref" in model.peft_config:
        model.delete_adapter("ref")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"DPO 训练完成!适配器已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
