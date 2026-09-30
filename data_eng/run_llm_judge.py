"""LLM-as-Judge 全量评测(复用 sft/eval_domain.py 的 judge_llm 与原有评分 prompt)。

约束:
  * **不重新推理**:只读取 data_eng/output_clean/predictions/*.jsonl 里已有的回答;
  * **不修改任何既有文件**:所有输出写入 data_eng/output_clean/llm_judge/
  * **不改 prompt / 数据 / 模型**:评分标准完全沿用 eval_domain.judge_llm
      - 知识库内问题:包含关键事实 4-5 分,部分正确 3 分,错误或编造 1-2 分
      - 知识库外问题:正确拒答 5 分,硬答或编造 1 分
  * 支持**断点续跑**:已评过的 (model, index) 会跳过,中断后可原命令重跑续上。

用法:
  python data_eng/run_llm_judge.py
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sft"))

import eval_domain as ev  # noqa: E402  复用 judge_llm(含原有 prompt)与 _load_api_key


def parse_args():
    p = argparse.ArgumentParser(description="LLM-as-Judge 全量评测(只读取已有 predictions)")
    p.add_argument("--predictions-dir", type=str, default="data_eng/output_clean/predictions")
    p.add_argument("--out-dir", type=str, default="data_eng/output_clean/llm_judge")
    p.add_argument("--judge-model", type=str, default="glm-4-flash", help="与 eval_domain 默认一致")
    p.add_argument("--sleep", type=float, default=0.3, help="每次调用后的间隔(与原实现一致)")
    p.add_argument("--retries", type=int, default=2, help="单条失败后的重试次数")
    p.add_argument("--limit-per-model", type=int, default=None, help="调试用:每个模型最多评多少条")
    return p.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done(path):
    """已完成的索引集合(断点续跑用)。"""
    done = set()
    if os.path.exists(path):
        for r in load_jsonl(path):
            done.add(r["index"])
    return done


def fmt_dur(sec):
    sec = int(max(sec, 0))
    return f"{sec // 60}m{sec % 60:02d}s"


def judge_one(rec, model, retries, sleep_s):
    """调用现有 judge_llm;返回 (score|None, error|None, elapsed)。"""
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            s = ev.judge_llm(rec["question"], rec.get("gold_answer", ""), rec["prediction"], model)
            return s, None, round(time.perf_counter() - t0, 3)
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(1.0 * (attempt + 1), 5.0))
    return None, last_err, None


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        ev._load_api_key()
    except Exception as e:
        raise SystemExit(f"[judge] API key 不可用:{e}")

    pred_files = sorted(glob.glob(os.path.join(args.predictions_dir, "*.jsonl")))
    if not pred_files:
        raise SystemExit(f"[judge] 在 {args.predictions_dir} 下没找到 predictions jsonl")

    print(f"[judge] judge_model = {args.judge_model} (来自 eval_domain 默认值)")
    print(f"[judge] 预测来源 = {args.predictions_dir} | 模型文件 {[os.path.basename(p) for p in pred_files]}")

    failures = defaultdict(list)
    for pred_file in pred_files:
        model = os.path.splitext(os.path.basename(pred_file))[0]
        out_path = os.path.join(args.out_dir, f"{model}.jsonl")
        records = load_jsonl(pred_file)
        if args.limit_per_model:
            records = records[: args.limit_per_model]
        done = load_done(out_path)
        todo = [(i, r) for i, r in enumerate(records) if i not in done]
        print(f"\n[model={model}] 总 {len(records)} 条 | 已完成 {len(done)} | 待评 {len(todo)}")

        handle = open(out_path, "a", encoding="utf-8")
        t_start = time.perf_counter()
        try:
            for n, (idx, rec) in enumerate(todo, 1):
                score, err, elapsed = judge_one(rec, args.judge_model, args.retries, args.sleep)
                row = {
                    "model": model,
                    "index": idx,
                    "intent": rec.get("intent"),
                    "is_gold_eval": bool(rec.get("is_gold_eval")),
                    "ood": bool(rec.get("ood")),
                    "question": rec["question"],
                    "gold_answer": rec.get("gold_answer", ""),
                    "prediction": rec["prediction"],
                    "judge_score": score,
                    "judge_error": err,
                    "judge_model": args.judge_model,
                    "elapsed_seconds": elapsed,
                    # 规则评测侧的判定(仅作交叉对照,不改判定逻辑)
                    "keyword_hit": bool(rec.get("keyword_hit")),
                    "refused": bool(rec.get("refused")),
                    "rule_correct": bool(rec.get("correct")),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if err:
                    failures[model].append({"index": idx, "error": err})
                elapsed_total = time.perf_counter() - t_start
                avg = elapsed_total / n
                eta = avg * (len(todo) - n)
                tag = f"score={score}" if score is not None else f"FAIL({err})"
                print(f"[model={model}] {n}/{len(todo)} | {tag} | avg={avg:.2f}s "
                      f"| elapsed={fmt_dur(elapsed_total)} | ETA={fmt_dur(eta)}")
                time.sleep(args.sleep)
        finally:
            handle.close()
        print(f"[model={model}] 本轮完成 {len(todo)} 条,失败 {len(failures[model])} 条")

    # ---------------- 汇总 ----------------
    summary = {}
    for pred_file in pred_files:
        model = os.path.splitext(os.path.basename(pred_file))[0]
        rows = load_jsonl(os.path.join(args.out_dir, f"{model}.jsonl"))
        ok = [r for r in rows if isinstance(r.get("judge_score"), int)]
        bad = [r for r in rows if not isinstance(r.get("judge_score"), int)]
        dom = [r for r in ok if not r["ood"]]
        ood = [r for r in ok if r["ood"]]
        gold = [r for r in ok if r.get("is_gold_eval")]
        refused_rule = [r for r in ok if r["refused"]]
        answered_rule = [r for r in ok if not r["refused"]]

        def mean(xs):
            return round(statistics.mean(x["judge_score"] for x in xs), 3) if xs else None

        summary[model] = {
            "judged": len(ok),
            "failed": len(bad),
            "attempted": len(rows),
            "mean_score": mean(ok),
            "mean_score_domain": mean(dom),
            "mean_score_ood": mean(ood),
            "mean_score_gold_eval": mean(gold),
            "n_domain": len(dom),
            "n_ood": len(ood),
            "n_gold_eval": len(gold),
            "semantic_correct_rate_domain(score>=4)": round(
                sum(1 for x in dom if x["judge_score"] >= 4) / len(dom), 3) if dom else None,
            "ood_refusal_reasonable_rate(score=5)": round(
                sum(1 for x in ood if x["judge_score"] == 5) / len(ood), 3) if ood else None,
            "score_distribution": dict(sorted(Counter(x["judge_score"] for x in ok).items())),
            "mean_score_when_rule_refused": mean(refused_rule),
            "mean_score_when_rule_not_refused": mean(answered_rule),
            "avg_judge_seconds": round(statistics.mean(r["elapsed_seconds"] for r in ok), 2)
            if any(r.get("elapsed_seconds") for r in ok) else None,
            "failures": failures.get(model, []),
        }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = ["# LLM-as-Judge 评测结果(clean 183 test,复用项目原有 judge prompt)", ""]
    lines.append(f"- Judge model:`{args.judge_model}`(与 `sft/eval_domain.py --judge-model` 默认一致)")
    lines.append("- 数据来源:`data_eng/output_clean/predictions/*.jsonl`(**直接复用已有回答,未重新推理**)")
    lines.append("- 评分标准(沿用原实现):知识库内问题 4-5 分=含关键事实、3 分=部分正确、1-2 分=错误/编造;"
                 "知识库外问题 5 分=正确拒答、1 分=硬答/编造")
    lines.append("")
    lines.append("| model | 实际评测 | 失败 | 平均分 | 领域均分 | OOD 均分 | gold_eval 均分 | 领域语义正确率(≥4) | OOD 拒答合理率(=5) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model, s in summary.items():
        def fmt(v, pct=False):
            if v is None:
                return "—"
            return f"{v*100:.1f}%" if pct else f"{v:.2f}"
        lines.append(
            f"| {model} | {s['judged']} | {s['failed']} | {fmt(s['mean_score'])} | {fmt(s['mean_score_domain'])} | "
            f"{fmt(s['mean_score_ood'])} | {fmt(s['mean_score_gold_eval'])} | "
            f"{fmt(s['semantic_correct_rate_domain(score>=4)'], True)} | "
            f"{fmt(s['ood_refusal_reasonable_rate(score=5)'], True)} |"
        )
    lines.append("")
    lines.append("## 分数分布")
    lines.append("")
    lines.append("| model | " + " | ".join(f"{i}分" for i in range(1, 6)) + " |")
    lines.append("|---|" + "---|" * 5)
    for model, s in summary.items():
        d = s["score_distribution"]
        lines.append(f"| {model} | " + " | ".join(str(d.get(i, 0)) for i in range(1, 6)) + " |")
    lines.append("")
    lines.append("## 与规则评测的交叉对照(均分)")
    lines.append("")
    lines.append("| model | 规则判为拒答的样本均分 | 规则判为未拒答的样本均分 |")
    lines.append("|---|---|---|")
    for model, s in summary.items():
        a = "—" if s["mean_score_when_rule_refused"] is None else f"{s['mean_score_when_rule_refused']:.2f}"
        b = "—" if s["mean_score_when_rule_not_refused"] is None else f"{s['mean_score_when_rule_not_refused']:.2f}"
        lines.append(f"| {model} | {a} | {b} |")
    lines.append("")
    lines.append("## 失败样本")
    lines.append("")
    any_fail = False
    for model, s in summary.items():
        if s["failures"]:
            any_fail = True
            lines.append(f"- **{model}**:{len(s['failures'])} 条失败,示例:{s['failures'][0]['error']}")
    if not any_fail:
        lines.append("- 无失败样本(全部返回有效 1-5 分)。")
    lines.append("")
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[judge] 汇总已写入:{os.path.join(args.out_dir, 'summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
