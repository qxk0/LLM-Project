"""RFT / 拒绝采样:on-policy 采样 + 规则验证器筛选高质量 SFT 数据。

思路(对应面试建议"on-policy 采样数据 + verifier 筛选好数据"):
  1. 让当前 SFT 模型对每条领域问题采样 K 个回答(on-policy)
  2. 用与 GRPO 相同的规则验证器打分:关键实体命中 / 正确拒答 / 回答简洁
  3. 只保留验证器通过的"好回答"作为额外 SFT 数据

这比纯模板数据更多样,且保证质量 —— 是 RFT(Rejection Fine-Tuning)的简化实现。
输出:data_eng/output/onpolicy.jsonl,由 build_dataset.py --extra-file 合并。
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train_dpo import load_rows_stratified, sample_completions_batched, score_response


def parse_args():
    parser = argparse.ArgumentParser(description="on-policy 采样 + 验证器筛选")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", type=str, default="models/sft")
    parser.add_argument("--data-file", type=str, default="data_eng/output/train.jsonl")
    parser.add_argument("--max-rows", type=int, default=400)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=1.0,
                        help="验证器通过阈值:1.0=答对/正确拒答")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批量采样 batch(只影响吞吐,采样参数不变)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data_eng/output/onpolicy.jsonl")
    parser.add_argument("--rejected-out", type=str, default=None,
                        help="把未通过验证器的采样也落盘(含 reward),便于审计通过率")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rows = load_rows_stratified(args.data_file, args.max_rows, seed=args.seed)
    from collections import Counter as _Counter
    print(f"加载 {len(rows)} 条领域数据(按 intent 分层抽样)"
          f" | 意图覆盖 {dict(_Counter(r['intent'] for r in rows))}"
          f" | OOD {sum(1 for r in rows if r['ood'])} 条")

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
    print(f"加载 SFT 模型进行 on-policy 采样 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto", local_files_only=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    completions_per_prompt = sample_completions_batched(
        model, tokenizer, [r["q"] for r in rows], args.num_samples,
        batch_size=args.batch_size, tag="rft",
    )
    passed, rejected, total = [], [], 0
    # ⚠️ 修复点:旧实现进度打印用 `i`(未定义,循环变量是 `r`),第一条样本就 NameError;
    # 改为 enumerate 并对齐 train_dpo 的进度打印节奏。
    for i, (r, completions) in enumerate(zip(rows, completions_per_prompt)):
        for resp in completions:
            total += 1
            sc = score_response(resp, r["keywords"], r["ood"])
            if sc >= args.min_score:
                passed.append(
                    {
                        "intent": r["intent"],
                        "q": r["q"],
                        "a": resp,
                        "keywords": r["keywords"],
                        "ood": r["ood"],
                        "source": "onpolicy",
                        # ---- 样本血缘:继承采样来源行的元数据(data_eng/output_clean 自带) ----
                        "seed_id": r.get("seed_id"),
                        "family_id": r.get("family_id"),
                        "split_group_id": r.get("split_group_id"),
                        "entity_key": r.get("entity_key"),
                        "base_question": r.get("base_question"),
                        "parent_q": r.get("q"),
                        "parent_sample_id": r.get("sample_id"),
                        "variant_type": "onpolicy",
                        "lineage_confidence": "self_reported" if r.get("family_id") else "unknown",
                    }
                )
            else:
                rejected.append({
                    "intent": r["intent"],
                    "q": r["q"],
                    "a": resp,
                    "keywords": r["keywords"],
                    "ood": r["ood"],
                    "source": "onpolicy_rejected",
                    "reward": sc,
                })
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(rows)},累计通过 {len(passed)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in passed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.rejected_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.rejected_out)), exist_ok=True)
        with open(args.rejected_out, "w", encoding="utf-8") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter as _C
    stats = {
        "rows": len(rows),
        "num_samples": args.num_samples,
        "total_samples": total,
        "passed": len(passed),
        "rejected": len(rejected),
        "pass_rate": round(len(passed) / max(total, 1), 4),
        "min_score": args.min_score,
        "intent_distribution": dict(_C(r["intent"] for r in rows)),
        "ood_rows": sum(1 for r in rows if r["ood"]),
    }
    with open(os.path.splitext(args.out)[0] + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"完成:共采样 {total} 条,验证器通过 {len(passed)} 条(通过率 {len(passed) / max(total, 1):.0%})")
    print("下一步:python data_eng/build_dataset.py --extra-file " + args.out)


if __name__ == "__main__":
    main()
