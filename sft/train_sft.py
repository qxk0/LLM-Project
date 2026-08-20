"""Qwen2.5-0.5B LoRA 监督微调(SFT)脚本。

做什么:
1. 以 4bit 量化(NF4)加载 Qwen2.5-0.5B 基座模型 —— 4GB 显存才装得下
2. 冻结全部原始参数,只训练 LoRA 低秩适配器(可训练参数约 1%)
3. 用中文指令数据(Alpaca 格式)微调,让模型学会"听指令回答问题"

训练产物:models/sft/ 下保存 LoRA 适配器(几 MB,不是完整模型)。

为什么用 LoRA:
  全量微调 5 亿参数模型在 4GB 显存上会爆显存;LoRA 只训练少量旁路参数,
  效果接近全量微调,显存占用却只有零头。这是当下微调的事实标准。
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B LoRA 监督微调")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", type=str, default="shibing624/alpaca-zh")
    parser.add_argument("--data-file", type=str, default=None,
                        help="本地 JSONL 数据(数据管道产物),格式:{q,a,keywords,ood}")
    parser.add_argument("--mix-general", type=float, default=0.2,
                        help="混合通用指令数据比例(相对领域数据),防止领域过拟合、保持通用对话能力")
    parser.add_argument("--max-examples", type=int, default=2000, help="取多少条训练样本")
    parser.add_argument("--max-steps", type=int, default=300, help="训练步数")
    parser.add_argument("--batch-size", type=int, default=1, help="每设备 batch(4GB 显存用 1)")
    parser.add_argument("--grad-accum", type=int, default=8, help="梯度累积,等效 batch 翻倍")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="models/sft")
    return parser.parse_args()


def load_tokenizer(model_name):
    """优先读本地缓存,避免联网。"""
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    except Exception:
        print("本地缓存未找到,尝试联网加载(网络受限时可设置 HF_ENDPOINT 镜像)...")
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_model(model_name, bnb_config):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
    except Exception:
        print("本地缓存未找到,尝试联网加载...")
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )


def format_example(row):
    """把 Alpaca 格式(instruction/input/output)转成对话消息。"""
    instruction = row["instruction"]
    inp = row.get("input") or ""
    content = instruction + ("\n" + inp if inp.strip() else "")
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 1. tokenizer ----------
    tokenizer = load_tokenizer(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen 没有 pad token,用 eos 代替

    # ---------- 2. 4bit 量化加载基座模型 ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"正在以 4bit 量化加载 {args.model_name} (首次运行会下载模型)...")
    model = load_model(args.model_name, bnb_config)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # 训练时关闭 KV 缓存,配合梯度检查点省显存

    # ---------- 3. LoRA ----------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        # Qwen2.5 所有线性层都挂 LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------- 4. 数据 ----------
    if args.data_file:
        # 自建数据管道产物:领域客服数据,直接本地加载
        print(f"正在加载本地数据 {args.data_file} ...")
        raw = load_dataset("json", data_files=args.data_file, split="train")
        n = min(args.max_examples, len(raw))
        data = raw.select(range(n)).map(
            lambda r: {
                "messages": [
                    {"role": "user", "content": r["q"]},
                    {"role": "assistant", "content": r["a"]},
                ]
            }
        )
    else:
        print(f"正在加载数据集 {args.dataset} (首次运行会下载)...")
        raw = load_dataset(args.dataset, split="train")
        n = min(args.max_examples, len(raw))
        data = raw.select(range(n)).map(format_example)
    # 领域数据 + 通用数据混合:先训得稳,再在领域上收敛
    if args.data_file and args.mix_general > 0:
        try:
            general_raw = load_dataset("shibing624/alpaca-zh", split="train")
            m = min(int(n * args.mix_general), len(general_raw))
            general = general_raw.select(range(m)).map(format_example)
            from datasets import concatenate_datasets
            data = concatenate_datasets([data, general])
            n = len(data)
            print(f"混合通用指令数据 {m} 条,总样本 {len(data)} 条(通用占比 {m / len(data):.0%})")
        except Exception as e:
            print(f"[警告] 通用数据混合失败,仅用领域数据:{e}")
    print(f"使用 {n} 条指令样本")

    def tokenize_fn(ex):
        """chat template 转 token;assistant 回答之前的位置在标签里全部屏蔽(-100)。"""
        msgs = ex["messages"]
        full = tokenizer.apply_chat_template(msgs, tokenize=False)
        # 只到"assistant 开始"为止,用于定位回答的起点
        prompt = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        full_ids = tokenizer(full, truncation=True, max_length=args.max_seq_len).input_ids
        prompt_ids = tokenizer(prompt).input_ids
        labels = full_ids.copy()
        mask_len = min(len(prompt_ids), len(full_ids))
        labels[:mask_len] = [-100] * mask_len  # 只让模型学习回答部分
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    dataset = data.map(tokenize_fn, remove_columns=data.column_names)

    # ---------- 5. 训练 ----------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=int(0.03 * args.max_steps),
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        processing_class=tokenizer,
    )

    print(f"开始 SFT 训练: {args.max_steps} 步,等效 batch = {args.batch_size * args.grad_accum}")
    trainer.train()

    # ---------- 6. 保存 ----------
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"训练完成!LoRA 适配器已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
