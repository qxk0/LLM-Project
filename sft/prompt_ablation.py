"""prompt format 消融实验(diagnostic only):chat vs raw × base/sft/dpo/rl × 5 固定题。

约束:
  * 只做 4 models × 2 formats × 5 questions = **40 次 generation**,不跑全量;
  * 两种 format 的 generation 参数完全相同:max_new_tokens=32 / do_sample=False /
    num_beams=1(HF 默认即 1,未启用 beam search),stop_ids 沿用 evaluator 的 [151643, 151645];
  * chat = tokenizer.apply_chat_template([{"role":"user","content":q}], tokenize=False,
    add_generation_prompt=True);raw = 直接用 question,两者之间不插入任何额外 prompt;
  * 复用 eval_domain.generate_answer / judge_row,保证与正式 evaluator 同一代码路径与判定口径;
  * 结果写入独立的 prompt_ablation/ 目录,**不覆盖**正式 predictions/;
  * 不做任何排名结论,只报告"谁对 prompt format 敏感"。

用法:
  python sft/prompt_ablation.py --data-file data_eng/output_clean/test.jsonl
"""

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eval_domain as ev  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="prompt format 消融(chat vs raw)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data-file", type=str, default="data_eng/output_clean/test.jsonl")
    parser.add_argument("--adapters", nargs="*", default=["models/sft", "models/dpo", "models/rl"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="data_eng/output_clean/prompt_ablation")
    parser.add_argument("--report", type=str, default="data_eng/output_clean/PROMPT_ABLATION.md")
    parser.add_argument("--report-only", action="store_true",
                        help="只读 out-dir 下已保存的 jsonl 生成汇总/报告,不加载模型、不推理")
    return parser.parse_args()


def pick_five(rows):
    """固定 5 条:price / recommend / customize / hours / ood 各一条(按文件顺序,确定性)。"""
    wanted = ["price", "recommend", "customize", "hours", "ood"]
    picked = []
    for intent in wanted:
        for r in rows:
            if r.get("intent") == intent:
                picked.append(r)
                break
    missing = [i for i in wanted if i not in {r.get("intent") for r in picked}]
    if missing:
        raise SystemExit(f"测试集缺少 intent: {missing}")
    return picked


def main():
    args = parse_args()
    rows = ev.load_test_set(args.data_file)
    questions = pick_five(rows)
    print(f"固定 5 条(price/recommend/customize/hours/ood):")
    for i, r in enumerate(questions, 1):
        print(f"  {i}. [{r['intent']}] ood={r['ood']} {r['q']}")

    tokenizer = ev.AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = ev.resolve_stop_ids(tokenizer)
    print(f"stop_ids = {stop_ids} | max_new_tokens = {args.max_new_tokens} | "
          f"do_sample=False | num_beams=1(HF 默认,无 beam search)")

    os.makedirs(args.out_dir, exist_ok=True)
    configs = [("base", None)] + [(os.path.basename(a.rstrip("/\\")), a) for a in args.adapters]

    records_by_combo = {}       # (model, fmt) -> list[record]
    gpu_mem = {}

    if args.report_only:
        # 离线重建:直接读已保存的 JSONL,不加载模型、不推理
        for model_name, _ in configs:
            for fmt in ("chat", "raw"):
                path = os.path.join(args.out_dir, f"{model_name}_{fmt}.jsonl")
                records_by_combo[(model_name, fmt)] = ev.load_predictions(path)
        ref = records_by_combo[(configs[0][0], "chat")]
        by_index = {r["index"]: r for r in ref}
        questions = [
            {
                "q": by_index[i]["question"],
                "intent": by_index[i]["intent"],
                "a": by_index[i]["gold_answer"],
                "keywords": by_index[i]["keywords"],
                "ood": by_index[i]["ood"],
            }
            for i in sorted(by_index)
        ]
        print(f"[report-only] 读取 {sum(len(v) for v in records_by_combo.values())} 条已保存记录"
              f"(未加载模型、未推理)| stop_ids 沿用 {stop_ids}")

    for model_name, adapter in ([] if args.report_only else configs):
        model = ev.build_model(args.model_name, adapter)
        gpu_mem[model_name] = round(torch.cuda.memory_allocated() / 2**20, 1)
        for fmt in ("chat", "raw"):
            records = []
            print(f"\n=== {model_name} / {fmt} ===")
            for idx, r in enumerate(questions):
                gen = ev.generate_answer(
                    model, tokenizer, r["q"], args.max_new_tokens,
                    prompt_format=fmt, stop_ids=stop_ids,
                )
                hit, refused = ev.judge_row(r, gen["prediction"])
                rec = {
                    "model": model_name,
                    "prompt_format": fmt,
                    "index": idx,
                    "question": r["q"],
                    "intent": r.get("intent"),
                    "gold_answer": r.get("a", ""),
                    "keywords": r.get("keywords", []),
                    "ood": bool(r.get("ood")),
                    "prediction": gen["prediction"],
                    "generated_tokens": gen["generated_tokens"],
                    "hit_eos": bool(gen["hit_eos"]),
                    "elapsed_seconds": gen["elapsed_seconds"],
                    "keyword_hit": bool(hit),
                    "refused": bool(refused),
                    "correct": bool(refused) if r.get("ood") else bool(hit),
                }
                records.append(rec)
                print(f"  [{idx}] {r['intent']:<10} {rec['elapsed_seconds']:5.2f}s "
                      f"tok={rec['generated_tokens']:2d} eos={rec['hit_eos']} "
                      f"hit={rec['keyword_hit']} refused={rec['refused']}")
            path = os.path.join(args.out_dir, f"{model_name}_{fmt}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records_by_combo[(model_name, fmt)] = records
            print(f"  -> 已保存 {path}")
        del model
        torch.cuda.empty_cache()

    # ---------------- 汇总表 ----------------
    def summarize(records):
        domain = [r for r in records if not r["ood"]]
        ood = [r for r in records if r["ood"]]
        return {
            "accuracy": sum(1 for r in records if r["correct"]) / len(records),
            "domain_accuracy": sum(1 for r in domain if r["keyword_hit"]) / max(len(domain), 1),
            "ood_refusal": (sum(1 for r in ood if r["refused"]) / len(ood)) if ood else None,
            "eos_hit": sum(1 for r in records if r["hit_eos"]) / len(records),
            "avg_tokens": statistics.mean(r["generated_tokens"] for r in records),
            "avg_seconds": statistics.mean(r["elapsed_seconds"] for r in records),
            "keyword_hit_rate": sum(1 for r in records if r["keyword_hit"]) / len(records),
        }

    summary = {f"{m}/{f}": summarize(recs) for (m, f), recs in records_by_combo.items()}

    header = (f"{'model':<7}{'format':<7}{'accuracy':>10}{'domainAcc':>11}{'OOD refuse':>12}"
              f"{'EOS hit':>9}{'avg tok':>9}{'avg sec':>9}")
    print("\n" + "=" * 80)
    print(header)
    print("-" * 80)
    for model_name, _ in configs:
        for fmt in ("chat", "raw"):
            s = summary[f"{model_name}/{fmt}"]
            ood = "n/a" if s["ood_refusal"] is None else f"{s['ood_refusal']*100:.0f}%"
            print(f"{model_name:<7}{fmt:<7}{s['accuracy']*100:>9.0f}%{s['domain_accuracy']*100:>10.0f}%"
                  f"{ood:>12}{s['eos_hit']*100:>8.0f}%{s['avg_tokens']:>9.1f}{s['avg_seconds']:>9.2f}")
    print("=" * 80)

    # ---------------- 逐题预测对照 ----------------
    print("\n逐题预测对照(同一题下比较 chat/raw):")
    for idx, r in enumerate(questions):
        print(f"\n--- [{idx}] intent={r['intent']} ood={r['ood']} | Q: {r['q']}")
        print(f"    gold: {r.get('a','')}")
        for model_name, _ in configs:
            for fmt in ("chat", "raw"):
                rec = records_by_combo[(model_name, fmt)][idx]
                pred = rec["prediction"].replace("\n", " ")[:70]
                flag = "对" if rec["correct"] else "错"
                print(f"    {model_name:<5}/{fmt:<4} [{flag}] eos={int(rec['hit_eos'])} "
                      f"tok={rec['generated_tokens']:2d} | {pred}")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "gpu_mem_mb": gpu_mem,
                   "stop_ids": stop_ids, "max_new_tokens": args.max_new_tokens,
                   "questions": [{"intent": r["intent"], "q": r["q"]} for r in questions]},
                  f, ensure_ascii=False, indent=2)

    # ---------------- markdown 报告 ----------------
    lines = ["# prompt format 消融(diagnostic only)", "",
             f"- 固定 5 题(price/recommend/customize/hours/ood),4 模型 × 2 格式 × 5 题 = **40 次 generation**",
             f"- generation 参数:max_new_tokens={args.max_new_tokens}, do_sample=False, num_beams=1(HF 默认)",
             f"- stop_ids = {stop_ids}(沿用 evaluator,未改动 EOS 逻辑)",
             "- chat = `apply_chat_template([{role:user}], add_generation_prompt=True)`;raw = 原始 question",
             "- 本实验仅作诊断,**不代表最终评测口径**", "",
             "## 汇总", "",
             "| model | format | accuracy | domain accuracy | OOD refusal | EOS hit | avg tokens | avg seconds |",
             "|---|---|---|---|---|---|---|---|"]
    for model_name, _ in configs:
        for fmt in ("chat", "raw"):
            s = summary[f"{model_name}/{fmt}"]
            ood = "n/a" if s["ood_refusal"] is None else f"{s['ood_refusal']*100:.0f}%"
            lines.append(f"| {model_name} | {fmt} | {s['accuracy']*100:.0f}% | {s['domain_accuracy']*100:.0f}% | "
                         f"{ood} | {s['eos_hit']*100:.0f}% | {s['avg_tokens']:.1f} | {s['avg_seconds']:.2f} |")
    lines.append("")
    lines.append("## 逐题预测")
    lines.append("")
    for idx, r in enumerate(questions):
        lines.append(f"### [{idx}] {r['intent']}{' (OOD)' if r['ood'] else ''}")
        lines.append("")
        lines.append(f"- Q:{r['q']}")
        lines.append(f"- gold:{r.get('a','')}")
        lines.append("")
        lines.append("| model | format | correct | EOS | tokens | seconds | prediction |")
        lines.append("|---|---|---|---|---|---|---|")
        for model_name, _ in configs:
            for fmt in ("chat", "raw"):
                rec = records_by_combo[(model_name, fmt)][idx]
                pred = rec["prediction"].replace("\n", " ").replace("|", "/")[:80]
                lines.append(f"| {model_name} | {fmt} | {'✓' if rec['correct'] else '✗'} | "
                             f"{int(rec['hit_eos'])} | {rec['generated_tokens']} | "
                             f"{rec['elapsed_seconds']:.2f} | {pred} |")
        lines.append("")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存: {args.report}")
    print(f"逐条记录目录: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
