"""领域客服评测:同一评测集上对比 基座 / SFT / RL 模型,输出量化指标。

指标定义(全部可复现、可写进简历):
  领域正确率   : 非知识库外问题中,回答包含答案关键实体(如价格数字/产品名)的比例.
  拒答率      : 知识库外问题中,模型正确拒绝(不瞎编)的比例.
  总正确率     : 两者加权.
  未命中率     : 领域问题上回答不含任何关键实体(幻觉/答非所问的代理指标).
  平均回答长度 : 简洁性.
"""

import argparse
import httpx
import json
import os
import re
import random
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REFUSAL_MARKERS = ["抱歉", "不好意思", "无法回答", "超出", "服务范围", "咨询"]
JUDGE_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def parse_args():
    parser = argparse.ArgumentParser(description="领域客服模型评测")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data-file", type=str, default="data_eng/output/test.jsonl")
    parser.add_argument("--adapters", nargs="*", default=["models/sft"],
                        help="要对比的适配器,如 models/sft models/rl;留空则只评测基座")
    parser.add_argument("--report", type=str, default="data_eng/output/eval_domain.md")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--ood-only", action="store_true", help="只评测拒答样本(快速验证)")
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少条")
    parser.add_argument("--judge", action="store_true",
                        help="启用 LLM-as-Judge(智谱 GLM)语义打分,与关键词命中双轨评测")
    parser.add_argument("--judge-limit", type=int, default=15, help="Judge 抽样条数")
    parser.add_argument("--judge-model", type=str, default="glm-4-flash")
    return parser.parse_args()


def load_test_set(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_model(model_name, adapter=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", local_files_only=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model


def answer(model, tokenizer, q, max_new):
    inputs = tokenizer(q, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def _load_api_key():
    key = os.getenv("ZHIPU_API_KEY")
    if key:
        return key.strip()
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("ZHIPU_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("未找到 ZHIPU_API_KEY(.env 或环境变量)")


def judge_llm(q, a, resp, model):
    """让智谱 GLM 评判回答质量(1-5 分),语义判断,不只看关键词。"""
    prompt = (
        "你是大模型客服评测裁判。判断客服回答是否正确且忠于知识库:\n"
        f"【问题】{q}\n【标准答案】{a}\n【客服回答】{resp}\n"
        "评分规则:\n"
        "- 知识库内问题:回答包含标准答案的关键事实给 4-5 分;部分正确 3 分;错误或编造 1-2 分\n"
        "- 知识库外问题:正确拒答给 5 分;硬答或编造给 1 分\n"
        "只输出一个整数分数,不要任何其他文字。"
    )
    key = _load_api_key()
    r = httpx.post(
        f"{JUDGE_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 8,
        },
        timeout=30,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\d+", text)
    return int(m.group(0)) if m else None


def llm_judge_scores(details, model, limit):
    """对每个模型的回答抽样打分,返回 (平均分, 抽样数)。"""
    rng = random.Random(0)
    sample = rng.sample(details, min(limit, len(details)))
    scores = []
    for r, resp, hit, refused in sample:
        try:
            s = judge_llm(r["q"], r["a"], resp, model)
            if s:
                scores.append(s)
        except Exception:
            pass
        time.sleep(0.3)
    if not scores:
        return None, 0
    return sum(scores) / len(scores), len(scores)


def evaluate(model, tokenizer, rows, max_new):
    domain_hit, domain_total, refusal_hit, ood_total = 0, 0, 0, 0
    lengths = []
    details = []
    for r in rows:
        resp = answer(model, tokenizer, r["q"], max_new)
        lengths.append(len(resp))
        hit = any(k in resp for k in r["keywords"])
        refused = any(m in resp for m in REFUSAL_MARKERS)
        if r["ood"]:
            ood_total += 1
            refusal_hit += 1 if refused else 0
        else:
            domain_total += 1
            domain_hit += 1 if hit else 0
        details.append((r, resp, hit, refused))
    metrics = {
        "domain_acc": domain_hit / max(domain_total, 1),
        "refusal_rate": refusal_hit / max(ood_total, 1),
        "overall_acc": (domain_hit + refusal_hit) / max(len(rows), 1),
        "miss_rate": 1 - domain_hit / max(domain_total, 1),
        "avg_len": sum(lengths) / max(len(lengths), 1),
        "samples": len(rows),
    }
    return metrics, details


def main():
    args = parse_args()
    rows = load_test_set(args.data_file)
    if args.ood_only:
        rows = [r for r in rows if r["ood"]]
    if args.limit:
        rows = rows[: args.limit]
    print(f"评测集:{len(rows)} 条(领域 {sum(1 for r in rows if not r['ood'])} / 拒答 {sum(1 for r in rows if r['ood'])})")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    configs = [("base", None)]
    configs += [(os.path.basename(a.rstrip("/\\")), a) for a in args.adapters]

    for name, adapter in configs:
        print(f"评测 {name} ...")
        model = build_model(args.model_name, adapter)
        metrics, details = evaluate(model, tokenizer, rows, args.max_new_tokens)
        results.append((name, metrics, details))
        del model
        torch.cuda.empty_cache()

    # LLM-as-Judge:语义打分,与关键词命中双轨
    judge_scores = {}
    if args.judge:
        print("\nLLM-as-Judge 语义打分(智谱 GLM)...")
        for name, _, details in results:
            avg, n = llm_judge_scores(details, args.judge_model, args.judge_limit)
            judge_scores[name] = (avg, n)
            print(f"  {name}: 平均 {avg:.2f} / 5 (抽样 {n} 条)" if avg else f"  {name}: 打分失败")

    # 控制台表格
    print("\n" + "=" * 72)
    header = f"{'模型':<10}{'领域正确率':>10}{'拒答率':>9}{'总正确率':>9}{'未命中率':>9}{'平均长度':>9}"
    print(header)
    for name, m, _ in results:
        print(
            f"{name:<10}{m['domain_acc'] * 100:>9.1f}%{m['refusal_rate'] * 100:>8.1f}%"
            f"{m['overall_acc'] * 100:>8.1f}%{m['miss_rate'] * 100:>8.1f}%{m['avg_len']:>9.1f}"
        )
    if args.judge:
        print(f"{'模型':<10}{'LLM-Judge均分':>16}")
        for name, (avg, n) in judge_scores.items():
            print(f"{name:<10}{avg:>14.2f} / 5 (n={n})" if avg else f"{name:<10} 打分失败")
    print("=" * 72)

    # 报告
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    lines = ["# 领域客服评测报告\n", f"评测集:{len(rows)} 条\n"]
    lines.append("| 模型 | 领域正确率 | 拒答率 | 总正确率 | 未命中率 | 平均回答长度 |")
    lines.append("|---|---|---|---|---|---|")
    for name, m, _ in results:
        lines.append(
            f"| {name} | {m['domain_acc'] * 100:.1f}% | {m['refusal_rate'] * 100:.1f}% | "
            f"{m['overall_acc'] * 100:.1f}% | {m['miss_rate'] * 100:.1f}% | {m['avg_len']:.1f} |"
        )
    best = max(results, key=lambda r: r[1]["overall_acc"])
    lines.append(f"\n**最佳:{best[0]}(总正确率 {best[1]['overall_acc'] * 100:.1f}%)**\n")
    if args.judge:
        lines.append("### LLM-as-Judge 语义打分(智谱 GLM,1-5 分)\n")
        lines.append("| 模型 | 平均分 | 抽样数 |")
        lines.append("|---|---|---|")
        for name, (avg, n) in judge_scores.items():
            lines.append(f"| {name} | {avg:.2f} | {n} |" if avg else f"| {name} | 失败 | 0 |")
        lines.append("")
    for name, _, details in results:
        lines.append(f"## {name} 回答示例\n")
        for r, resp, hit, refused in details[:5]:
            tag = "拒答✓" if refused else ("命中✓" if hit else "未命中✗")
            lines.append(f"- Q:{r['q']}\n  - {tag} A:{resp[:120]}")
        lines.append("")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"评测报告已保存: {args.report}")


if __name__ == "__main__":
    main()
