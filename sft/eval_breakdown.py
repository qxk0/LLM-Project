"""[已废弃 · 不再推理] 细分统计请改用离线模式:

    python sft/eval_domain.py --from-predictions data_eng/output_clean/predictions \
        --report data_eng/output_clean/eval_domain_clean.md

本文件保留仅作历史记录:其入口已被改为转发到上面的离线统计(只读 predictions JSONL),
不会再调用 model.generate。下面保留的旧实现函数改名为 *_deprecated_inference,不会被入口调用。

--- 原始说明(已过时) ---
评测结果细分(只读推理):按 intent / source / gold_eval 维度拆分。

约束:
  * 不修改 sft/eval_domain.py,不修改任何数据集,不训练任何模型;
  * 直接 import eval_domain 并复用它的 build_model / answer / evaluate,
    保证解码方式(greedy, max_new_tokens 同参)与判定口径(关键词命中 / 拒答话术)完全一致;
  * 只额外做两件事:① 把每条样本的模型回答落盘成 jsonl 便于人工复核;
    ② 按 intent 与 source 分组统计同口径指标。

用法:
  python sft/eval_breakdown.py --data-file data_eng/output_clean/test.jsonl \
      --adapters models/sft models/dpo models/rl models/dpo/checkpoint-200
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_domain as ev  # noqa: E402  复用评测器逻辑,避免口径漂移


def parse_args():
    parser = argparse.ArgumentParser(description="按 intent/source 拆分的评测明细")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data-file", type=str, default="data_eng/output_clean/test.jsonl")
    parser.add_argument("--adapters", nargs="*", default=["models/sft"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--report", type=str, default="data_eng/output_clean/eval_domain_clean_breakdown.md")
    parser.add_argument(
        "--predictions", type=str, default="data_eng/output_clean/eval_predictions_clean.jsonl"
    )
    return parser.parse_args()


def source_group(row):
    """gold_eval = 本轮新增的人工评测题;legacy_clean = 上一轮 clean test 的 gold 样本。"""
    return "gold_eval" if row.get("source") == "gold_eval" else "legacy_clean"


def main_deprecated_inference():
    args = parse_args()
    rows = ev.load_test_set(args.data_file)
    print(f"评测集:{len(rows)} 条(领域 {sum(1 for r in rows if not r['ood'])} / 拒答 {sum(1 for r in rows if r['ood'])})")

    tokenizer = ev.AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    configs = [("base", None)] + [
        (os.path.basename(a.rstrip("/\\")), a) for a in args.adapters
    ]

    all_predictions = []
    summary = {}
    for name, adapter in configs:
        print(f"评测 {name} ...")
        model = ev.build_model(args.model_name, adapter)
        metrics, details = ev.evaluate(model, tokenizer, rows, args.max_new_tokens)
        summary[name] = {"overall": metrics, "by_intent": {}, "by_source": {}}

        # 逐条落盘(便于人工复核)
        for r, resp, hit, refused in details:
            all_predictions.append(
                {
                    "model": name,
                    "eval_id": r.get("eval_id"),
                    "sample_id": r.get("sample_id"),
                    "intent": r.get("intent"),
                    "source": r.get("source"),
                    "source_group": source_group(r),
                    "ood": bool(r["ood"]),
                    "q": r["q"],
                    "a": r["a"],
                    "keywords": r["keywords"],
                    "prediction": resp,
                    "keyword_hit": bool(hit),
                    "refused": bool(refused),
                    "correct": bool(refused) if r["ood"] else bool(hit),
                }
            )

        # 按 intent 汇总(同口径:非 OOD 看关键词命中, OOD 看拒答)
        buckets = defaultdict(lambda: {"total": 0, "correct": 0, "domain_total": 0, "domain_hit": 0, "ood_total": 0, "refused": 0})
        for r, resp, hit, refused in details:
            b = buckets[r["intent"]]
            b["total"] += 1
            if r["ood"]:
                b["ood_total"] += 1
                b["refused"] += 1 if refused else 0
                b["correct"] += 1 if refused else 0
            else:
                b["domain_total"] += 1
                b["domain_hit"] += 1 if hit else 0
                b["correct"] += 1 if hit else 0
        for intent, b in buckets.items():
            summary[name]["by_intent"][intent] = {
                "total": b["total"],
                "domain_total": b["domain_total"],
                "domain_hit": b["domain_hit"],
                "domain_acc": round(b["domain_hit"] / max(b["domain_total"], 1), 4),
                "ood_total": b["ood_total"],
                "refused": b["refused"],
                "refusal_rate": round(b["refused"] / max(b["ood_total"], 1), 4) if b["ood_total"] else None,
                "accuracy": round(b["correct"] / max(b["total"], 1), 4),
            }

        # 按 source 分组 + gold_eval vs legacy_clean
        for group_name in ("gold_eval", "legacy_clean"):
            sub = [(r, resp, hit, refused) for r, resp, hit, refused in details if source_group(r) == group_name]
            dom = [x for x in sub if not x[0]["ood"]]
            ood = [x for x in sub if x[0]["ood"]]
            summary[name]["by_source"][group_name] = {
                "total": len(sub),
                "domain_total": len(dom),
                "domain_acc": round(sum(1 for x in dom if x[2]) / max(len(dom), 1), 4),
                "ood_total": len(ood),
                "refusal_rate": round(sum(1 for x in ood if x[3]) / max(len(ood), 1), 4) if ood else None,
                "overall_acc": round(
                    (sum(1 for x in dom if x[2]) + sum(1 for x in ood if x[3])) / max(len(sub), 1), 4
                ),
            }
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.predictions), exist_ok=True)
    with open(args.predictions, "w", encoding="utf-8") as f:
        for p in all_predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    lines = ["# 评测结果细分(intent / source / gold_eval)\n"]
    lines.append(f"- 数据:{args.data_file}({len(rows)} 条)")
    lines.append("- 解码:greedy,max_new_tokens=%d;判定口径与 sft/eval_domain.py 完全一致(直接复用其函数)\n" % args.max_new_tokens)
    lines.append("## 1. 总体(与 eval_domain_clean.md 一致)\n")
    lines.append("| model | domain_acc | refusal_rate | overall_acc | miss_rate | avg_len | samples |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, _ in configs:
        m = summary[name]["overall"]
        lines.append(
            f"| {name} | {m['domain_acc']*100:.1f}% | {m['refusal_rate']*100:.1f}% | "
            f"{m['overall_acc']*100:.1f}% | {m['miss_rate']*100:.1f}% | {m['avg_len']:.1f} | {m['samples']} |"
        )
    lines.append("")
    lines.append("## 2. 按 intent 拆分(领域正确率 / 拒答率)\n")
    intents = [i for i in ["price", "recommend", "customize", "hours", "delivery", "member", "after_sale", "allergen", "store", "ood"]]
    for name, _ in configs:
        lines.append(f"### {name}\n")
        lines.append("| intent | n | domain_acc | ood 拒答率 | intent 准确率 |")
        lines.append("|---|---|---|---|---|")
        for intent in intents:
            b = summary[name]["by_intent"].get(intent)
            if not b:
                continue
            ref = "—" if b["refusal_rate"] is None else f"{b['refusal_rate']*100:.1f}%"
            dom = f"{b['domain_acc']*100:.1f}%" if b["domain_total"] else "—"
            lines.append(f"| {intent} | {b['total']} | {dom} | {ref} | {b['accuracy']*100:.1f}% |")
        lines.append("")
    lines.append("## 3. gold_eval(本轮新增人工评测题)vs legacy_clean(上一轮 clean test)\n")
    lines.append("| model | 分组 | n | domain_acc | 拒答率 | overall_acc |")
    lines.append("|---|---|---|---|---|---|")
    for name, _ in configs:
        for group_name in ("gold_eval", "legacy_clean"):
            b = summary[name]["by_source"][group_name]
            ref = "—" if b["refusal_rate"] is None else f"{b['refusal_rate']*100:.1f}%"
            dom = f"{b['domain_acc']*100:.1f}%" if b["domain_total"] else "—"
            lines.append(
                f"| {name} | {group_name} | {b['total']} | {dom} | {ref} | {b['overall_acc']*100:.1f}% |"
            )
    lines.append("")
    lines.append(f"逐条预测见:`{args.predictions}`\n")
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(args.report.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("细分报告:", args.report)
    print("逐条预测:", args.predictions)
    return 0


if __name__ == "__main__":
    # 只允许离线统计:直接转发到 eval_domain.py 的 --from-predictions,绝不重新推理
    print(
        "[deprecated] 本脚本不再跑推理;转发到 eval_domain.py --from-predictions(纯离线统计)。",
        file=sys.stderr,
    )
    sys.argv = [
        "eval_domain.py",
        "--from-predictions",
        "data_eng/output_clean/predictions",
        "--report",
        "data_eng/output_clean/eval_domain_clean.md",
    ]
    raise SystemExit(ev.main())
