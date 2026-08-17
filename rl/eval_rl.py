"""RL 前后效果对比:同一批未见过的数学题,SFT 模型 vs RL 模型,统计正确率。

生成 models/rl/rl_comparison.md 报告,这是简历上"RL 提升了 X%"的量化证据。
"""

import argparse
import os
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(description="RL 前后正确率对比")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sft-adapter", type=str, default="models/sft")
    parser.add_argument("--rl-adapter", type=str, default="models/rl")
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--report", type=str, default="models/rl/rl_comparison.md")
    return parser.parse_args()


def make_questions(num, seed):
    rng = random.Random(seed)
    qs = []
    for _ in range(num):
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        if rng.random() < 0.5:
            qs.append((f"请计算:{a} + {b} = ?", str(a + b)))
        else:
            a, b = max(a, b), min(a, b)
            qs.append((f"请计算:{a} - {b} = ?", str(a - b)))
    return qs


def build_model(model_name, adapter):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", local_files_only=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
    return PeftModel.from_pretrained(model, adapter)


def answer(model, tokenizer, prompt, max_new=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,  # 贪婪解码,评测更稳定
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)


def evaluate(model, tokenizer, questions):
    correct = 0
    details = []
    for prompt, ans in questions:
        resp = answer(model, tokenizer, prompt)
        nums = re.findall(r"-?\d+", resp)
        pred = int(nums[-1]) if nums else None
        ok = pred == int(ans)
        correct += ok
        details.append((prompt, ans, resp, ok))
    return correct / len(questions), details


def main():
    args = parse_args()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    questions = make_questions(args.num_questions, args.seed)
    print(f"生成 {len(questions)} 道评测题(与训练题不同随机种子)")

    print("评测 SFT 模型...")
    sft_model = build_model(args.model_name, args.sft_adapter)
    sft_acc, sft_details = evaluate(sft_model, tokenizer, questions)
    del sft_model
    torch.cuda.empty_cache()

    print("评测 RL 模型...")
    rl_model = build_model(args.model_name, args.rl_adapter)
    rl_acc, rl_details = evaluate(rl_model, tokenizer, questions)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    lines = ["# RL 前后正确率对比\n"]
    lines.append(f"- SFT 模型正确率:**{sft_acc * 100:.1f}%**")
    lines.append(f"- RL 模型正确率:**{rl_acc * 100:.1f}%**")
    lines.append(f"- 提升:**{(rl_acc - sft_acc) * 100:+.1f} 个百分点**\n")
    for i, ((prompt, ans, resp, ok), (_, _, rl_resp, rl_ok)) in enumerate(
        zip(sft_details, rl_details), 1
    ):
        lines.append(f"## 题 {i}:{prompt}(答案 {ans})")
        lines.append(f"- SFT:{'对' if ok else '错'} -> {resp[:100]}")
        lines.append(f"- RL :{'对' if rl_ok else '错'} -> {rl_resp[:100]}\n")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"SFT 正确率 {sft_acc * 100:.1f}% | RL 正确率 {rl_acc * 100:.1f}%")
    print(f"对比报告已保存: {args.report}")


if __name__ == "__main__":
    main()
