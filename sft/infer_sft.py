"""用微调后的 LoRA 模型对话,并和基座模型做对比(生成 comparison.md)。"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_PROMPTS = [
    "用三句话介绍一下什么是大语言模型。",
    "太阳为什么从东边升起?",
    "写一首关于春天的五言绝句。",
    "1+1 等于几?请给出理由。",
    "怎么养成早起的好习惯?给出三条建议。",
]


def parse_args():
    parser = argparse.ArgumentParser(description="SFT 模型推理与对比")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", type=str, default="models/sft")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--compare", action="store_true", help="同时用基座模型回答做对比")
    parser.add_argument("--report", type=str, default="models/sft/comparison.md")
    return parser.parse_args()


def build_model(model_name, adapter=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model


def chat(model, tokenizer, prompt, max_new_tokens, temperature):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return answer.strip()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先记录基座模型回答,再挂上 LoRA 回答(避免同时占两份显存)
    base_answers = {}
    sft_answers = {}
    if args.compare:
        print("正在用基座模型回答...")
        base_model = build_model(args.model_name)
        for p in DEFAULT_PROMPTS:
            base_answers[p] = chat(base_model, tokenizer, p, args.max_new_tokens, args.temperature)
            print(f"  [base] {p[:20]}... -> {base_answers[p][:60]}")

    print("正在用微调后模型回答...")
    sft_model = build_model(args.model_name, args.adapter)
    for p in DEFAULT_PROMPTS:
        sft_answers[p] = chat(sft_model, tokenizer, p, args.max_new_tokens, args.temperature)
        print(f"  [sft ] {p[:20]}... -> {sft_answers[p][:60]}")

    # 生成对比报告
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    lines = ["# SFT 前后效果对比\n"]
    for i, p in enumerate(DEFAULT_PROMPTS, 1):
        lines.append(f"## 问题 {i}:{p}\n")
        if args.compare:
            lines.append(f"**基座模型:**\n\n{base_answers[p]}\n")
        lines.append(f"**SFT 微调后:**\n\n{sft_answers[p]}\n")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"对比报告已保存: {args.report}")


if __name__ == "__main__":
    main()
