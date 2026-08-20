"""GRPO 强化学习对齐脚本(DeepSeek-R1 同款思路)。

做什么:
1. 从 SFT 微调后的模型继续(4bit 基座 + LoRA 适配器)
2. 用 GRPO 做强化学习:模型先自己生成答案,再按"规则奖励"打分
3. 奖励函数(纯规则,不依赖大模型):
   - 正确率奖励:答案数字对不对(+1.0)
   - 格式奖励:有没有写出"答案是 ..."(+0.5)
   模型通过试错学会"把步骤写出来、答案写对",这就是对齐的核心思想。

任务:两位数以内的加减法(数据本地生成,不用下载)。
训练产物:models/rl/ 下保存 RL 训练后的 LoRA 适配器。
"""

import argparse
import json
import os
import random
import re

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


def load_tokenizer(model_name):
    """优先读本地缓存,避免联网(训练时不需要网络)。"""
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    except Exception:
        print("本地缓存未找到,尝试联网加载(网络受限时可设置 HF_ENDPOINT 镜像)...")
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_model(model_name, bnb_config):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        print("本地缓存未找到,尝试联网加载...")
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO 强化学习(数学题对齐)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", type=str, default="models/sft", help="SFT 微调好的适配器")
    parser.add_argument("--mode", type=str, default="domain", choices=["domain", "math"],
                        help="domain=领域客服对齐(默认);math=数学题对齐")
    parser.add_argument("--data-file", type=str, default="data_eng/output/train.jsonl",
                        help="domain 模式的数据管道产物")
    parser.add_argument("--num-samples", type=int, default=400, help="训练题数量")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4, help="累积步数(与 batch 乘积需能被候选数整除)")
    parser.add_argument("--num-generations", type=int, default=4, help="每题生成几个候选答案")
    parser.add_argument("--max-completion-length", type=int, default=64, help="回答上限,给答案留足空间")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="models/rl")
    parser.add_argument("--loss-type", type=str, default="dapo", choices=["dapo", "grpo"],
                        help="dapo=动态裁剪(默认,clip-higher);grpo=经典GRPO")
    parser.add_argument("--seq-level", action="store_true",
                        help="GSPO 风格:序列级损失,消除'长回答梯度更大'的长度偏差(需 --loss-type grpo)")
    return parser.parse_args()


def make_math_dataset(num_samples, seed):
    """本地生成加减法题目:prompt 是问题,answer 是正确答案。"""
    rng = random.Random(seed)
    rows = []
    for _ in range(num_samples):
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        if rng.random() < 0.5:
            prompt, answer = f"请计算:{a} + {b} = ?", str(a + b)
        else:
            a, b = max(a, b), min(a, b)  # 保证减法结果非负
            prompt, answer = f"请计算:{a} - {b} = ?", str(a - b)
        rows.append({"prompt": prompt, "answer": answer})
    return Dataset.from_list(rows)


def make_domain_dataset(path, ood_ratio=0.2, seed=42):
    """读取数据管道产物,并把拒答样本占比提到 ood_ratio(聚焦拒答对齐)。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.append(
                    {
                        "prompt": r["q"],
                        "answer_keywords": r["keywords"],
                        "ood": r["ood"],
                    }
                )
    ood = [r for r in rows if r["ood"]]
    ind = [r for r in rows if not r["ood"]]
    rng = random.Random(seed)
    n_ind = min(len(ind), int(len(ood) * (1 - ood_ratio) / ood_ratio))
    sampled = rng.sample(ind, n_ind) + ood
    rng.shuffle(sampled)
    return Dataset.from_list(sampled)


def extract_last_number(text):
    """从模型输出中提取最后一个整数,作为它给出的答案。"""
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def correctness_reward(prompts, completions, answer, **kwargs):
    """答案出现在回答中即得 1 分(独立数字,防止被 25 误匹配 125)。"""
    rewards = []
    for comp, ans in zip(completions, answer):
        found = re.search(rf"(?<!\d){int(ans)}(?!\d)", comp)
        rewards.append(1.0 if found else 0.0)
    return rewards


def format_reward(prompts, completions, **kwargs):
    """回答简洁(≤80 字符)得 0.1 分(低权重,别压过答对)。"""
    return [0.1 if len(c.strip()) <= 80 else 0.0 for c in completions]


def domain_reward(prompts, completions, answer_keywords, ood, **kwargs):
    """领域客服对齐奖励(对称惩罚,梯度更强):
    知识库内问题 -> 包含答案关键实体 +1,否则 -1;
    知识库外问题 -> 正确拒答 +1,瞎编 -1。
    """
    rewards = []
    for comp, kws, is_ood in zip(completions, answer_keywords, ood):
        if is_ood:
            refused = any(m in comp for m in ["抱歉", "不好意思", "无法回答", "超出", "服务范围", "咨询"])
            rewards.append(1.0 if refused else -1.0)
        else:
            rewards.append(1.0 if any(k in comp for k in kws) else -1.0)
    return rewards


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 1. 数据 ----------
    if args.mode == "domain":
        dataset = make_domain_dataset(args.data_file)
        reward_funcs = [domain_reward, format_reward]
        ood_count = sum(1 for r in dataset if r["ood"])
        print(f"加载领域数据 {len(dataset)} 条(拒答样本 {ood_count} 条,占比 {ood_count / len(dataset):.0%})")
    else:
        dataset = make_math_dataset(args.num_samples, args.seed)
        reward_funcs = [correctness_reward, format_reward]
        print(f"生成 {len(dataset)} 道数学题(本地生成,无需下载)")

    # ---------- 2. 模型:SFT 微调后的 LoRA 继续 ----------
    tokenizer = load_tokenizer(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"加载基座 {args.model_name} + SFT 适配器 {args.adapter} ...")
    model = load_model(args.model_name, bnb_config)
    model = PeftModel.from_pretrained(model, args.adapter)

    # ---------- 3. GRPO 训练 ----------
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.7,  # 降低采样噪声,让优势估计更稳
        top_p=0.95,
        top_k=50,
        # 对齐技巧:
        #  - loss_type="dapo": DAPO 动态裁剪(正负优势分开裁剪,缓解奖励饱和)
        #  - importance_sampling_level="sequence": GSPO 序列级损失,去掉长度偏置
        #  - mask_truncated_completions=True: 被 max_completion_length 截断的回答不计入损失
        loss_type="grpo" if args.seq_level else args.loss_type,
        importance_sampling_level="sequence" if args.seq_level else "token",
        mask_truncated_completions=True,
        beta=0.04,  # KL 惩罚系数:约束 RL 后的模型别离 SFT 太远
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    # TRL 创建 ref 参考适配器后,活动适配器会变成 "ref";
    # 必须切回 "default",否则训练更新的是 ref,而保存时又会被删掉(等于白训)。
    model.set_adapter("default")

    print(
        f"开始 GRPO 训练:{args.max_steps} 步,loss={training_args.loss_type}"
        f"{'(序列级)' if args.seq_level else ''},截断掩码已开启"
    )
    trainer.train()

    # ---------- 4. 保存 ----------
    # 删掉训练用的参考适配器,只保留 RL 更新后的 default 适配器
    if "ref" in model.peft_config:
        model.delete_adapter("ref")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"RL 训练完成!适配器已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
