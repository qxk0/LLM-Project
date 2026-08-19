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
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
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


def score_response(comp, keywords, is_ood):
    """与 GRPO 相同的规则奖励:答对/拒答 +1,否则 -1;简洁 +0.1。"""
    score = 0.0
    if is_ood:
        score += 1.0 if any(m in comp for m in REFUSAL_MARKERS) else -1.0
    else:
        score += 1.0 if any(k in comp for k in keywords) else -1.0
    score += 0.1 if len(comp.strip()) <= 80 else 0.0
    return score


def build_preference_pairs(model, tokenizer, rows, num_samples):
    """采样 + 规则奖励打分 -> chosen/rejected 偏好对。"""
    pairs = []
    skipped = 0
    for i, r in enumerate(rows):
        prompt = r["q"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.eos_token_id,
            )
        completions = [
            tokenizer.decode(o[inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
            for o in out
        ]
        scored = [(score_response(c, r["keywords"], r["ood"]), c) for c in completions]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, chosen = scored[0]
        worst_score, rejected = scored[-1]
        if best_score > worst_score:  # 分数不同才保留,避免噪声对
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        else:
            skipped += 1
        if (i + 1) % 50 == 0:
            print(f"  已构造 {len(pairs)} 对 (跳过 {skipped})...")
    print(f"偏好对总数:{len(pairs)} | 因分数相同跳过:{skipped}")
    return Dataset.from_list(pairs)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_rows(args.data_file, args.max_pairs)
    print(f"加载 {len(rows)} 条领域数据")

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

    print(f"构造偏好对:每题采样 {args.num_samples} 个回答,用规则奖励打分...")
    dataset = build_preference_pairs(model, tokenizer, rows, args.num_samples)

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
