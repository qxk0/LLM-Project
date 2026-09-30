"""评测速度诊断 benchmark(只读推理,不修改 evaluator / 数据 / 权重)。

目的:定位 sft/eval_domain.py 在 183 条 clean test 上为什么慢。
做法:复用 eval_domain.build_model / load_test_set 加载模型,用与 eval_domain.answer()
完全一致的 generation 参数(greedy / max_new_tokens / pad_token_id)计时,
并额外记录 token 数、EOS 命中、GPU 显存、每秒 token。

用法:
  python sft/eval_bench.py --data-file data_eng/output_clean/test.jsonl \
      --adapters models/sft models/dpo models/rl --n 5
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_domain as ev  # noqa: E402  复用评测器的加载逻辑与判定口径


def parse_args():
    parser = argparse.ArgumentParser(description="评测速度诊断 benchmark")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data-file", type=str, default="data_eng/output_clean/test.jsonl")
    parser.add_argument("--adapters", nargs="*", default=["models/sft"])
    parser.add_argument("--n", type=int, default=5, help="每个模型测多少题")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="与 eval_domain 新默认一致")
    parser.add_argument("--out", type=str, default="data_eng/output_clean/eval_bench.json")
    return parser.parse_args()


def gpu_mem_mb():
    if not torch.cuda.is_available():
        return None
    return {
        "allocated": round(torch.cuda.memory_allocated() / 2**20, 1),
        "reserved": round(torch.cuda.memory_reserved() / 2**20, 1),
        "max_allocated": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }


def timed_generate(model, tokenizer, q, max_new):
    """与修正后的 sft/eval_domain.py 走同一条推理路径:chat template + <|im_end|> 停止符。

    直接复用 evaluator 的 generate_answer,保证 benchmark 与正式评测口径完全一致。
    """
    stop_ids = ev.resolve_stop_ids(tokenizer)
    gen = ev.generate_answer(model, tokenizer, q, max_new, prompt_format="chat", stop_ids=stop_ids)
    return {
        "seconds": gen["elapsed_seconds"],
        "prompt_tokens": gen["prompt_tokens"],
        "new_tokens": gen["generated_tokens"],
        "hit_eos": bool(gen["hit_eos"]),
        "text": gen["prediction"],
    }


def pick_questions(rows, n):
    """确定性挑题:优先覆盖不同 intent(price/recommend/customize/hours/ood...),再做补齐。"""
    order = ["price", "recommend", "customize", "hours", "ood", "delivery", "member", "after_sale", "allergen", "store"]
    picked, seen = [], set()
    for intent in order:
        for i, r in enumerate(rows):
            if i in seen or r.get("intent") != intent:
                continue
            picked.append((i, r))
            seen.add(i)
            break
        if len(picked) >= n:
            break
    for i, r in enumerate(rows):
        if len(picked) >= n:
            break
        if i not in seen:
            picked.append((i, r))
            seen.add(i)
    return picked[:n]


def main():
    args = parse_args()
    print("cuda available:", torch.cuda.is_available(), "| device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    rows = ev.load_test_set(args.data_file)
    picked = pick_questions(rows, args.n)
    print(f"benchmark 题数: {len(picked)} / 总 {len(rows)}")
    for i, r in picked:
        print(f"  [{i}] intent={r['intent']} ood={r['ood']} q={r['q']}")

    tokenizer = ev.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    configs = [("base", None)] + [(os.path.basename(a.rstrip("/\\")), a) for a in args.adapters]
    report = {"cuda": torch.cuda.is_available(), "max_new_tokens": args.max_new_tokens, "models": {}}

    for name, adapter in configs:
        print(f"\n=== {name} ===")
        torch.cuda.reset_peak_memory_stats()
        t_load0 = time.perf_counter()
        model = ev.build_model(args.model_name, adapter)
        load_s = time.perf_counter() - t_load0
        print(f"  模型+adapter 加载: {load_s:.2f}s | GPU {gpu_mem_mb()}")

        # 预热一次(不计入统计),排除首次 kernel 编译 / 显存分配的影响
        _, warm = picked[0]
        warmup = timed_generate(model, tokenizer, warm["q"], args.max_new_tokens)
        print(f"  warmup: {warmup['seconds']:.2f}s, {warmup['new_tokens']} tokens")

        per_q = []
        for idx, r in picked:
            res = timed_generate(model, tokenizer, r["q"], args.max_new_tokens)
            res.update({"index": idx, "intent": r["intent"], "ood": r["ood"], "q": r["q"]})
            per_q.append(res)
            print(
                f"  [{idx}] {r['intent']:<11} {res['seconds']:6.2f}s  "
                f"new_tokens={res['new_tokens']:3d}  eos={res['hit_eos']}"
            )

        total_s = sum(x["seconds"] for x in per_q)
        total_tok = sum(x["new_tokens"] for x in per_q)
        report["models"][name] = {
            "adapter": adapter,
            "load_seconds": round(load_s, 2),
            "num_questions": len(per_q),
            "total_generation_seconds": round(total_s, 2),
            "seconds_per_question": round(total_s / len(per_q), 3),
            "tokens_generated": total_tok,
            "tokens_per_second": round(total_tok / total_s, 2) if total_s else None,
            "mean_new_tokens": round(statistics.mean(x["new_tokens"] for x in per_q), 1),
            "max_new_tokens_observed": max(x["new_tokens"] for x in per_q),
            "eos_hit_rate": round(sum(1 for x in per_q if x["hit_eos"]) / len(per_q), 3),
            "gpu_mem_mb": gpu_mem_mb(),
            "samples": [
                {k: v for k, v in x.items() if k != "text"} for x in per_q
            ],
        }
        del model
        torch.cuda.empty_cache()

    # 额外:同一题在 8 / 32 / 64 token 下的耗时,用来分离"单次调用固定开销"与"每 token 成本"
    print("\n=== token 长度扫描(sft,同一题)===")
    sweep = []
    model = ev.build_model(args.model_name, "models/sft")
    q = picked[0][1]["q"]
    for mnt in (8, 32, 64):
        res = timed_generate(model, tokenizer, q, mnt)
        sweep.append({"max_new_tokens": mnt, "seconds": round(res["seconds"], 3), "new_tokens": res["new_tokens"]})
        print(f"  max_new_tokens={mnt:3d} -> {res['seconds']:.3f}s ({res['new_tokens']} tokens)")
    del model
    torch.cuda.empty_cache()
    report["token_sweep_sft"] = sweep

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nbenchmark 结果已保存:", args.out)

    print("\n| model | load(s) | n | s/question | tokens | tok/s | mean_new_tok | EOS hit |")
    print("|---|---|---|---|---|---|---|---|")
    for name, m in report["models"].items():
        print(
            f"| {name} | {m['load_seconds']} | {m['num_questions']} | {m['seconds_per_question']} | "
            f"{m['tokens_generated']} | {m['tokens_per_second']} | {m['mean_new_tokens']} | {m['eos_hit_rate']} |"
        )

    # 用实测速率外推 183 条
    print("\n外推(183 条 / 每个模型):")
    for name, m in report["models"].items():
        est = m["seconds_per_question"] * 183 + m["load_seconds"]
        print(f"  {name}: {est:.0f}s = {est/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
