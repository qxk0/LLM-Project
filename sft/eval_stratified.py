"""分层评测报告:把 clean test 按"训练集是否覆盖了本题所需事实"切成两片,分别统计。

为什么需要这个:
    本项目的 clean 切分是按样本血缘 group 做的 —— 同一个 (实体, 意图) 只能进一个 split。
    好处是零泄漏,代价是**测试集里必然存在训练集从未覆盖过的场景**
    (最典型的是"布丁推荐"那一族:60 条题、训练集里一条都没有)。
    把这两类题混在一起算一个正确率,会得到一个既不能反映能力、也不能反映泛化的数字。

因此本脚本把 domain 题拆成两片:
    * seen   「熟场景新问法」:本题判定所需的关键词(判定器要求命中任一)
                             在**同一 intent** 的训练集里出现过 → 属于"学过、换个问法"。
    * unseen 「未见过场景」  :上述事实在训练集里完全没出现 → 模型无从得知,只能靠泛化或拒答。
    * ood    知识库外拒答    :单独统计,不与上面两片混算正确率。

切片口径之所以用"关键词 × intent",是因为它和评测器的判定方式一致:
    `eval_domain.keyword_hit` 只要命中任一关键词即判对,所以"能不能被判对"的前提
    就是这些关键词至少有一个在训练集同 intent 下出现过。

**本脚本只读 predictions JSONL,不做任何模型推理**(不 import torch,不加载模型)。

用法:
    python sft/eval_stratified.py \
        --data-file runs/2026-10-12/data/clean/test.jsonl \
        --train-file runs/2026-10-12/data/clean/train.jsonl \
        --pred-dir runs/2026-10-12/eval/predictions --judge-dir runs/2026-10-12/judge \
        --pred-dir runs/2026-10-13/gspo_ablation/eval/predictions --judge-dir runs/2026-10-13/gspo_ablation/judge \
        --out-dir runs/2026-10-14/stratified_eval
"""

import argparse
import glob
import json
import os
import statistics

SLICES = ("seen", "unseen", "ood")
SLICE_LABEL = {
    "seen": "熟场景新问法",
    "unseen": "未见过场景",
    "ood": "OOD 拒答",
}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_slice_map(train_rows, test_rows):
    """返回 {test_index: slice},以及每个切片的题号列表。"""
    seen_facts = {(r["intent"], k) for r in train_rows for k in r.get("keywords", [])}
    slice_of = {}
    for i, r in enumerate(test_rows):
        if r.get("ood"):
            slice_of[i] = "ood"
        elif any((r["intent"], k) in seen_facts for k in r.get("keywords", [])):
            slice_of[i] = "seen"
        else:
            slice_of[i] = "unseen"
    return slice_of, seen_facts


def summarize(records):
    n = len(records)
    if n == 0:
        return None
    return {
        "n": n,
        "accuracy": round(sum(1 for r in records if r.get("correct")) / n, 4),
        "avg_chars": round(statistics.fmean(len(r.get("prediction", "")) for r in records), 1),
        "avg_generated_tokens": round(
            statistics.fmean(r.get("generated_tokens", 0) for r in records), 1
        ),
        "eos_hit_rate": round(sum(1 for r in records if r.get("hit_eos")) / n, 4),
    }


def collect(args):
    test_rows = load_jsonl(args.data_file)
    train_rows = load_jsonl(args.train_file)
    slice_of, seen_facts = build_slice_map(train_rows, test_rows)

    # 自动发现 predictions(多个目录时按 basename 去重,保留先出现的那个)
    pred_map = {}
    for d in args.pred_dir:
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            pred_map.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    if not pred_map:
        raise SystemExit("没有找到任何 predictions jsonl")

    judge_map = {}
    for d in args.judge_dir:
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            judge_map.setdefault(os.path.splitext(os.path.basename(p))[0], p)

    report = {"data_file": args.data_file, "train_file": args.train_file,
              "slice_definition": "keywords×intent:该题所需关键词是否在同 intent 训练集中出现过",
              "slice_sizes": {s: sum(1 for v in slice_of.values() if v == s) for s in SLICES},
              "models": {}}

    for name, pred_path in pred_map.items():
        preds = load_jsonl(pred_path)
        # 逐条 judge 分数(可能不存在,或含 None 的失败条目)
        judge_scores = {}
        if name in judge_map:
            for r in load_jsonl(judge_map[name]):
                if isinstance(r.get("judge_score"), int):
                    judge_scores[r["index"]] = r["judge_score"]

        entry = {"pred_file": os.path.relpath(pred_path),
                 "judge_file": os.path.relpath(judge_map[name]) if name in judge_map else None,
                 "overall": summarize(preds), "slices": {}}
        ov_js = [judge_scores[r["index"]] for r in preds if r["index"] in judge_scores]
        entry["overall"]["judge_mean"] = round(statistics.fmean(ov_js), 2) if ov_js else None
        entry["overall"]["judge_n"] = len(ov_js)
        for s in SLICES:
            sub = [r for r in preds if slice_of.get(r["index"]) == s]
            stats = summarize(sub)
            if stats is not None:
                js = [judge_scores[r["index"]] for r in sub if r["index"] in judge_scores]
                stats["judge_mean"] = round(statistics.fmean(js), 2) if js else None
                stats["judge_n"] = len(js)
            entry["slices"][s] = stats
        report["models"][name] = entry
    return report, seen_facts


def render_markdown(report):
    sz = report["slice_sizes"]
    L = ["# 分层评测报告(clean test)", "",
         "评测集:`%s`" % report["data_file"],
         "训练集:`%s`(仅用于判定「该题所需事实是否学过」,不参与任何推理)" % report["train_file"], "",
         "切片口径:%s" % report["slice_definition"], "",
         "- **熟场景新问法**(n=%d):判定所需关键词在同 intent 训练集中出现过 —— 学过、换个问法" % sz["seen"],
         "- **未见过场景**(n=%d):该事实在训练集中完全没出现 —— 模型无从得知" % sz["unseen"],
         "- **OOD 拒答**(n=%d):知识库外问题,单独统计,不与上面两片混算" % sz["ood"], "",
         "## 规则评测:关键词命中准确率", "",
         "| 模型 | 熟场景新问法(n=%d) | 未见过场景(n=%d) | OOD 拒答(n=%d) | 总体(n=%d) |" % (
             sz["seen"], sz["unseen"], sz["ood"], sum(sz.values())),
         "|---|---|---|---|---|"]
    for name, e in report["models"].items():
        s = e["slices"]
        def pct(x):
            return "—" if x is None else "%.1f%%" % (x["accuracy"] * 100)
        L.append("| %s | %s | %s | %s | %s |" % (
            name, pct(s["seen"]), pct(s["unseen"]), pct(s["ood"]),
            pct(e["overall"])))
    L += ["", "## LLM-as-Judge:语义均分(1-5)", "",
          "| 模型 | 熟场景新问法 | 未见过场景 | OOD 拒答 | 总体 |", "|---|---|---|---|---|"]
    for name, e in report["models"].items():
        s = e["slices"]
        def jm_(x):
            if x is None or x.get("judge_mean") is None:
                return "—"
            return "%.2f (n=%d)" % (x["judge_mean"], x["judge_n"])
        L.append("| %s | %s | %s | %s | %s |" % (
            name, jm_(s["seen"]), jm_(s["unseen"]), jm_(s["ood"]), jm_(e["overall"])))
    L += ["", "## 回答长度与终止(EOS)", "",
          "| 模型 | 切片 | 平均回答长度(字符) | 平均生成 token | EOS 命中率 |",
          "|---|---|---|---|---|"]
    for name, e in report["models"].items():
        for s in SLICES:
            st = e["slices"][s]
            if st is None:
                continue
            L.append("| %s | %s | %.1f | %.1f | %.1f%% |" % (
                name, SLICE_LABEL[s], st["avg_chars"], st["avg_generated_tokens"],
                st["eos_hit_rate"] * 100))
    L += ["", "## 说明", "",
          "- 「未见过场景」这一片正确率普遍是个位数,属于**预期结果**:该事实没有出现在训练集中,",
          "  模型没有可依赖的学习信号;这一片应该被读作「模型的泛化/拒答边界」,而不是「模型能力差」。",
          "- 两片分开报告后,「熟场景新问法」才是与 SFT/DPO/RL 训练目标直接对应的能力指标。",
          "- 本报告由 `sft/eval_stratified.py` 从已保存的 predictions 计算得到,**未做任何模型推理**。",
          ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="分层评测报告(只读 predictions,不推理)")
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--pred-dir", action="append", required=True)
    ap.add_argument("--judge-dir", action="append", default=[])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    report, _ = collect(args)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "stratified_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md = render_markdown(report)
    with open(os.path.join(args.out_dir, "STRATIFIED_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print("已写入:%s" % args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
