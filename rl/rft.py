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

from train_dpo import load_rows, score_response


def parse_args():
    parser = argparse.ArgumentParser(description="on-policy 采样 + 验证器筛选")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", type=str, default="models/sft")
    parser.add_argument("--data-file", type=str, default="data_eng/output/train.jsonl")
    parser.add_argument("--max-rows", type=int, default=400)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=1.0,
                        help="验证器通过阈值:1.0=答对/正确拒答")
    parser.add_argument("--out", type=str, default="data_eng/output/onpolicy.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.data_file, args.max_rows)
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
    print(f"加载 SFT 模型进行 on-policy 采样 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto", local_files_only=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    passed, total = [], 0
    for i, r in enumerate(rows):
        inputs = tokenizer(r["q"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.eos_token_id,
            )
        for o in out:
            resp = tokenizer.decode(o[inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
            total += 1
            if score_response(resp, r["keywords"], r["ood"]) >= args.min_score:
                passed.append(
                    {
                        "intent": r["intent"],
                        "q": r["q"],
                        "a": resp,
                        "keywords": r["keywords"],
                        "ood": r["ood"],
                        "source": "onpolicy",
                    }
                )
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(rows)},累计通过 {len(passed)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in passed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成:共采样 {total} 条,验证器通过 {len(passed)} 条(通过率 {len(passed) / max(total, 1):.0%})")
    print("下一步:python data_eng/build_dataset.py --extra-file " + args.out)


if __name__ == "__main__":
    main()
