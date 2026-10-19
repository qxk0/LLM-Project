"""Qwen2.5 / Qwen3 LoRA 监督微调(SFT)脚本。

做什么:
1. 以 4bit 量化(NF4)加载 Qwen2.5-0.5B / Qwen3-0.6B 基座模型 —— 4GB 显存才装得下
2. 冻结全部原始参数,只训练 LoRA 低秩适配器(可训练参数约 1%)
3. 用中文指令数据(Alpaca 格式)微调,让模型学会"听指令回答问题"

训练产物:models/sft/ 下保存 LoRA 适配器(几 MB,不是完整模型)。

Qwen3 说明:
- Qwen3 默认会进入 thinking 模式,生成较长的思考内容。客服 SFT 默认关闭 thinking,
  使用 `--enable-thinking` 才打开。
- 非 thinking 模式下,chat template 会在 assistant 开头插入空的 <think></think> 块;
  本脚本用 prompt + answer + EOS 拼接 full 序列,保证 prompt/full 的前缀严格对齐。

低显存 smoke test:
  python sft/train_sft.py --model-name Qwen/Qwen3-0.6B --smoke-test
  或直接运行: powershell -ExecutionPolicy Bypass -File sft/smoke_qwen3.ps1

为什么用 LoRA:
  全量微调 6 亿参数模型在 4GB 显存上会爆显存;LoRA 只训练少量旁路参数,
  效果接近全量微调,显存占用却只有零头。这是当下微调的事实标准。
"""

import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5 / Qwen3 LoRA 监督微调")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", type=str, default="shibing624/alpaca-zh")
    parser.add_argument("--data-file", type=str, default=None,
                        help="本地 JSONL 数据(数据管道产物),格式:{q,a,keywords,ood}")
    parser.add_argument("--mix-general", type=float, default=0.0,
                        help="混合通用指令数据比例(相对领域数据),防止领域过拟合、保持通用对话能力")
    parser.add_argument("--max-examples", type=int, default=2000, help="取多少条训练样本")
    parser.add_argument("--max-steps", type=int, default=300, help="训练步数")
    parser.add_argument("--batch-size", type=int, default=1, help="每设备 batch(4GB 显存用 1)")
    parser.add_argument("--grad-accum", type=int, default=8, help="梯度累积,等效 batch 翻倍")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="默认: 正式训练 models/sft; smoke test models/sft_qwen3_smoke")
    parser.add_argument("--smoke-test", action="store_true",
                        help="低显存冒烟测试: 32 条 / 10 步 / seq 256 / LoRA r=8")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Qwen3 使用 thinking 模式(默认关闭,客服 SFT 不需要长思考)")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16"], default="auto",
                        help="auto: Qwen3 用 bf16,其余用 fp16")
    parser.add_argument("--device-map", choices=["auto", "cuda"], default="auto",
                        help="auto 允许 CPU offload;cuda 强制单卡,显存不够时直接 OOM")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit",
                        help="优化器;4GB 显存建议 paged_adamw_8bit")
    return parser.parse_args()


def apply_smoke_defaults(args):
    """只覆盖仍处于默认值的参数,用户显式传入的值优先。"""
    if not args.smoke_test:
        return args
    if args.max_examples == 2000:
        args.max_examples = 32
    if args.max_steps == 300:
        args.max_steps = 10
    if args.max_seq_len == 1024:
        args.max_seq_len = 256
    if args.lora_r == 16:
        args.lora_r = 8
    if args.lora_alpha == 32:
        args.lora_alpha = 16
    if args.grad_accum == 8:
        args.grad_accum = 4
    if args.lr == 2e-4:
        args.lr = 1e-4
    if args.device_map == "auto":
        args.device_map = "cuda"
    if args.output_dir is None:
        args.output_dir = "models/sft_qwen3_smoke"
    return args


def load_tokenizer(model_name):
    """优先读本地缓存,避免联网。"""
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    except Exception:
        print("本地缓存未找到,尝试联网加载(网络受限时可设置 HF_ENDPOINT 镜像)...")
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_model(model_name, bnb_config, device_map="auto"):
    resolved_device_map = {"": 0} if device_map == "cuda" else device_map
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map=resolved_device_map,
            trust_remote_code=True, local_files_only=True,
        )
    except Exception:
        print("本地缓存未找到,尝试联网加载...")
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map=resolved_device_map,
            trust_remote_code=True,
        )


def render_chat_template(tokenizer, messages, add_generation_prompt, enable_thinking=None):
    """渲染 chat template。

    Qwen3 的模板支持 enable_thinking;Qwen2.5 不接受该变量,因此只在 Qwen3 上传入。
    """
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


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
    args = apply_smoke_defaults(parse_args())
    if args.output_dir is None:
        args.output_dir = "models/sft"
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 1. tokenizer ----------
    tokenizer = load_tokenizer(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen 没有 pad token,用 eos 代替

    # ---------- 2. 4bit 量化加载基座模型 ----------
    model_is_qwen3_hint = "qwen3" in args.model_name.lower()
    try:
        model_config = AutoConfig.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    is_qwen3 = getattr(model_config, "model_type", "").lower() == "qwen3" or model_is_qwen3_hint
    use_bf16 = args.precision == "bf16" or (args.precision == "auto" and is_qwen3)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"正在以 4bit 量化加载 {args.model_name} (首次运行会下载模型)...")
    model = load_model(args.model_name, bnb_config, device_map=args.device_map)
    print(
        f"[config] qwen3={is_qwen3} | thinking={args.enable_thinking if is_qwen3 else 'n/a'} "
        f"| precision={'bf16' if use_bf16 else 'fp16'} | seq_len={args.max_seq_len} "
        f"| batch={args.batch_size} | grad_accum={args.grad_accum} "
        f"| smoke={args.smoke_test}"
    )
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
        # 分层抽样(修复):旧实现 raw.select(range(n)) 只取文件前 n 行,而数据是按 intent 分块排列的,
        # 会导致 SFT 训练集里一条 OOD 拒答样本都没有 -> 拒答能力根本学不出来。
        torch.manual_seed(args.seed)
        idx_by_intent = {}
        for _i, _ex in enumerate(raw):
            idx_by_intent.setdefault(_ex.get("intent", "unknown"), []).append(_i)
        rng = random.Random(args.seed)
        quotas = {k: max(1, int(n * len(v) / len(raw))) for k, v in idx_by_intent.items()}
        while sum(quotas.values()) > n:
            k_max = max(quotas, key=lambda x: (quotas[x], x))
            if quotas[k_max] <= 1:
                break
            quotas[k_max] -= 1
        picked, used = [], {}
        for k, v in idx_by_intent.items():
            rng.shuffle(v)
            take = min(quotas[k], len(v))
            picked.extend(v[:take])
            used[k] = take
        if n - len(picked) > 0:
            pool = [i for k, v in idx_by_intent.items() for i in v[used[k]:]]
            rng.shuffle(pool)
            picked.extend(pool[: n - len(picked)])
        rng.shuffle(picked)
        picked = sorted(picked)[:n]
        print(f"      分层抽样 {len(picked)} 条 | intent 覆盖 {sorted(idx_by_intent)}"
              f" | OOD 条数 {sum(1 for i in picked if raw[i].get('ood'))}")
        data = raw.select(picked).map(
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
        # 只到"assistant 开始"为止,用于定位回答的起点。
        # Qwen3 非 thinking 模式会在这里插入空的 <think></think> 块;
        # full 直接复用 prompt 前缀,避免 full/prompt 分别渲染导致 token 错位。
        prompt = render_chat_template(
            tokenizer,
            msgs[:-1],
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking if is_qwen3 else None,
        )
        full_from_template = render_chat_template(
            tokenizer,
            msgs,
            add_generation_prompt=False,
            enable_thinking=args.enable_thinking if is_qwen3 else None,
        )
        if full_from_template.startswith(prompt):
            # Qwen2.5 等模板:full 天然以 prompt 为前缀,直接使用,保留 <|im_end|>。
            full = full_from_template
        else:
            # Qwen3 非 thinking:prompt 多出空的 <think></think> 块,
            # 用 prompt + answer + chat 结束符拼接,避免 full/prompt 错位。
            chat_end = "<|im_end|>" if "<|im_end|>" in tokenizer.get_vocab() else (tokenizer.eos_token or "")
            full = prompt + msgs[-1]["content"] + chat_end
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
        warmup_steps=max(1, int(0.03 * args.max_steps)),
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim=args.optim,
        logging_steps=1 if args.smoke_test else 10,
        logging_first_step=True,
        save_steps=5 if args.smoke_test else 100,
        save_total_limit=1 if args.smoke_test else 2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        processing_class=tokenizer,
    )

    print(
        f"开始 SFT 训练: {args.max_steps} 步,等效 batch = {args.batch_size * args.grad_accum} "
        f"| precision={'bf16' if use_bf16 else 'fp16'} | optim={args.optim}"
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()

    # ---------- 6. 保存 ----------
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"[vram] peak allocated={peak_alloc:.2f} GiB | peak reserved={peak_reserved:.2f} GiB")
    print(f"训练完成!LoRA 适配器已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
