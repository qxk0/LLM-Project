"""标准 GRPO(Group Relative Policy Optimization)baseline —— 独立于现有 DAPO 实验。

目的:在**不修改**现有 DAPO(`rl/train_grpo.py` + `models/rl/`)的前提下,补一个标准 GRPO
baseline,形成 SFT → DPO → GRPO → DAPO 的对比链路。

与 DAPO 的唯一算法差异(其余尽量保持一致):
    DAPO  : loss_type="dapo"  (TRL 默认;按 global 有效 token 数归一化,含 clip-higher 动态裁剪)
    GRPO  : loss_type="grpo"  (标准 GRPO objective:按序列长度归一化的 token-level loss)
    两者 importance_sampling_level 都是 "token";不使用 --seq-level(那是 GSPO 风格,本脚本不涉及)

复用(不重新实现):
    * 模型/tokenizer 加载      -> 直接从 rl/train_grpo.py 导入
    * 数据构造(make_domain_dataset)、reward(domain_reward / format_reward)同样是同一份实现
    * 日志/checkpoint 保存约定与 DAPO 保持一致

泄漏防护(重要):
    训练数据默认使用 **data_eng/output_clean/train.jsonl**(clean split),
    启动时会强制校验 `train ∩ clean test = ∅`,不通过则直接退出。
    原因:旧的 data_eng/output/train.jsonl 与 clean test 有 77/183 条精确同题(旧随机切分导致),
    用它训练会把 clean test 的题目喂进模型。

用法:
    python rl/train_grpo_standard.py --max-steps 300 --seed 42
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # 复用 rl/train_grpo.py 的实现
sys.path.insert(0, os.path.join(HERE, "..", "data_eng"))  # 复用 data_eng/leakage_audit.py 的归一化

import train_grpo as base  # noqa: E402
from leakage_audit import base_question, load_jsonl, normalize_text  # noqa: E402

ALGORITHM = "GRPO"          # 标准 GRPO(区别于 DAPO)
LOSS_TYPE = "grpo"         # 关键:不是 "dapo"
DEFAULT_DATA = "data_eng/output_clean/train.jsonl"
CLEAN_TEST = "data_eng/output_clean/test.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="标准 GRPO baseline(对比 DAPO)")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--adapter", type=str, default="models/sft", help="从 SFT 适配器继续(与 DAPO 一致)")
    p.add_argument("--data-file", type=str, default=DEFAULT_DATA, help="RL 训练数据(默认 clean train)")
    p.add_argument("--max-steps", type=int, default=300, help="与 DAPO 一致")
    p.add_argument("--batch-size", type=int, default=2, help="与 DAPO 一致")
    p.add_argument("--grad-accum", type=int, default=4, help="与 DAPO 一致")
    p.add_argument("--num-generations", type=int, default=4, help="与 DAPO 一致(G 个 response/组)")
    p.add_argument("--max-completion-length", type=int, default=64, help="与 DAPO 一致")
    p.add_argument("--lr", type=float, default=1.5e-4, help="与 DAPO 一致")
    p.add_argument("--temperature", type=float, default=0.7, help="与 DAPO 一致")
    p.add_argument("--beta", type=float, default=0.04, help="KL 系数,与 DAPO 一致")
    p.add_argument("--seed", type=int, default=42, help="与 DAPO 一致")
    p.add_argument("--ood-ratio", type=float, default=0.2, help="拒答样本占比,与 DAPO 一致")
    p.add_argument("--output-dir", type=str, default="models/grpo", help="独立目录,不覆盖 models/rl")
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                   help="断点续训(例如 models/grpo/checkpoint-200);会恢复 optimizer/scheduler/RNG 状态")
    p.add_argument("--skip-leak-check", action="store_true", help="跳过 train ∩ clean test 校验(不建议)")
    return p.parse_args()


def check_no_leak(train_path, test_path=CLEAN_TEST):
    """强制校验:训练集与 clean test 在 exact / base question 层面完全不相交。"""
    train = load_jsonl(train_path)
    test = load_jsonl(test_path)
    tq = {normalize_text(r["q"]) for r in train}
    tb = {normalize_text(base_question(r["q"])) for r in train}
    rows_exact = sum(1 for r in test if normalize_text(r["q"]) in tq)
    overlap_exact = len(tq & {normalize_text(r["q"]) for r in test})
    overlap_base = len(tb & {normalize_text(base_question(r["q"])) for r in test})
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_ood": sum(1 for r in train if r["ood"]),
        "exact_question_overlap": overlap_exact,
        "base_question_overlap": overlap_base,
        "test_rows_seen_in_train": rows_exact,
    }


def main():
    args = parse_args()

    # ---------- 0. 泄漏校验(硬门槛) ----------
    leak = check_no_leak(args.data_file)
    print(f"[leak-check] train={args.data_file} rows={leak['train_rows']} (ood={leak['train_ood']})")
    print(f"[leak-check] clean test rows={leak['test_rows']} | exact overlap={leak['exact_question_overlap']} "
          f"| base overlap={leak['base_question_overlap']} | test rows seen in train={leak['test_rows_seen_in_train']}")
    if (leak["exact_question_overlap"] or leak["base_question_overlap"]) and not args.skip_leak_check:
        raise SystemExit(
            "[leak-check 失败] 训练数据与 clean test 存在重叠,拒绝训练。"
            f"(exact={leak['exact_question_overlap']}, base={leak['base_question_overlap']})"
        )

    if os.path.abspath(args.output_dir) == os.path.abspath("models/rl"):
        raise SystemExit("拒绝写入 models/rl(那是现有 DAPO 的 checkpoint);请用 --output-dir models/grpo")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 0.5 续训检查点校验 ----------
    if args.resume_from_checkpoint:
        need = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "adapter_model.safetensors"]
        missing = [f for f in need if not os.path.exists(os.path.join(args.resume_from_checkpoint, f))]
        if missing:
            raise SystemExit(f"[resume] 检查点 {args.resume_from_checkpoint} 缺少续训所需文件: {missing}")
        prev_state = json.load(open(os.path.join(args.resume_from_checkpoint, "trainer_state.json"), encoding="utf-8"))
        print(f"[resume] 从 {args.resume_from_checkpoint} 续训:global_step={prev_state.get('global_step')} "
              f"→ max_steps={args.max_steps};optimizer/scheduler/RNG state 均存在")

    # ---------- 1. 数据(与 DAPO 同一份实现) ----------
    dataset = base.make_domain_dataset(args.data_file, ood_ratio=args.ood_ratio, seed=args.seed)
    ood_count = sum(1 for r in dataset if r["ood"])
    print(f"[data] RL prompt 数={len(dataset)}(拒答 {ood_count},占比 {ood_count/len(dataset):.0%})")

    # ---------- 2. 模型(tokenizer/loader 与 DAPO 完全同一实现) ----------
    tokenizer = base.load_tokenizer(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[model] 加载基座 {args.model_name} + SFT 适配器 {args.adapter} ...")
    model = base.load_model(args.model_name, bnb_config)
    model = PeftModel.from_pretrained(model, args.adapter)

    # ---------- 3. GRPO 训练(唯一与 DAPO 不同之处:loss_type) ----------
    training_args = GRPOConfig(
        output_dir=output_dir,
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
        temperature=args.temperature,
        top_p=0.95,
        top_k=50,
        seed=args.seed,
        # ===== 与 DAPO 的唯一算法差异 =====
        loss_type=LOSS_TYPE,                      # "grpo" = 标准 GRPO objective
        importance_sampling_level="token",        # 标准 GRPO:token 级重要性采样
        mask_truncated_completions=False,         # 与 DAPO 保持一致
        beta=args.beta,
    )
    assert training_args.loss_type != "dapo", "必须使用标准 GRPO loss,不能是 dapo"
    assert training_args.importance_sampling_level == "token"

    print(f"[algorithm] {ALGORITHM} | loss_type={training_args.loss_type} "
          f"| importance_sampling_level={training_args.importance_sampling_level}")
    print(f"[reward] domain_reward(命中/拒答 +1 否则 -1) + format_reward(<=80 字 +0.1),与 DAPO 相同")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[base.domain_reward, base.format_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    # 与 DAPO 相同的坑:TRL 创建 ref 适配器后活动适配器会变成 "ref",必须切回 default
    model.set_adapter("default")

    print(f"[train] 开始标准 GRPO 训练:{args.max_steps} 步,loss_type={training_args.loss_type}")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # ---------- 4. 保存(独立目录) ----------
    if "ref" in model.peft_config:
        model.delete_adapter("ref")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 续训时把中断前的 log_history 一并合并,保证 training_log.json 覆盖完整 300 步
    new_log = trainer.state.log_history
    full_log = new_log
    prev_state_path = os.path.join(args.resume_from_checkpoint or "", "trainer_state.json")
    if args.resume_from_checkpoint and os.path.exists(prev_state_path):
        # 注意:TrainingArguments(save_total_limit) 可能已经把旧 checkpoint 删掉,
        # 因此必须先判断文件是否存在;缺少时退化为"只写续训部分",不让训练结果丢失。
        prev_state = json.load(open(prev_state_path, encoding="utf-8"))
        merged = {}
        for e in prev_state.get("log_history", []):
            merged[e.get("step", -1)] = e
        for e in new_log:
            merged[e.get("step", -1)] = e
        full_log = [merged[k] for k in sorted(merged)]
        with open(os.path.join(output_dir, "training_log_resumed_part.json"), "w", encoding="utf-8") as f:
            json.dump(new_log, f, ensure_ascii=False, indent=2)
    elif args.resume_from_checkpoint:
        print(f"[warn] 续训起点 {args.resume_from_checkpoint} 已被 save_total_limit 清理,"
              f"training_log.json 只包含本次续训的 {len(new_log)} 条记录")
    with open(os.path.join(output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(full_log, f, ensure_ascii=False, indent=2)

    args_dump = {
        "algorithm": ALGORITHM,
        "loss_type": training_args.loss_type,
        "importance_sampling_level": training_args.importance_sampling_level,
        "is_dapo": training_args.loss_type == "dapo",
        "base_model": args.model_name,
        "init_adapter": args.adapter,
        "train_data": args.data_file.replace("\\", "/"),
        "leak_check": leak,
        "resumed_from": args.resume_from_checkpoint,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": training_args.top_p,
        "top_k": training_args.top_k,
        "beta": args.beta,
        "ood_ratio": args.ood_ratio,
        "reward": "domain_reward(hit/refusal ±1) + format_reward(<=80 chars +0.1) [same as DAPO]",
        "trl_version": __import__("trl").__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
        "torch_version": torch.__version__,
    }
    with open(os.path.join(output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dump, f, ensure_ascii=False, indent=2)

    log = full_log
    rewards = [e["reward"] for e in log if "reward" in e]
    losses = [e["loss"] for e in log if "loss" in e]
    print("=" * 78)
    print(f"完成 | steps={args.max_steps} | loss_type={training_args.loss_type} "
          f"| reward: first={rewards[0]:.3f} last={rewards[-1]:.3f} "
          f"| loss: first={losses[0]:.4f} last={losses[-1]:.4f} | 输出={output_dir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
