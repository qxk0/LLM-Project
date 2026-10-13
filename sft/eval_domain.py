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
    # 默认报告路径指向 clean 目录,避免误覆盖历史报告 data_eng/output/eval_domain.md
    parser.add_argument("--report", type=str, default="data_eng/output_clean/eval_domain_clean.md")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--ood-only", action="store_true", help="只评测拒答样本(快速验证)")
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少条")
    parser.add_argument("--prompt-format", type=str, default="chat", choices=["chat", "raw"],
                        help="chat(默认)=与 SFT 训练一致的 chat template;raw=旧的裸 prompt 行为")
    parser.add_argument("--predictions-dir", type=str, default="data_eng/output_clean/predictions",
                        help="逐条预测落盘目录(每个模型一个 jsonl)")
    parser.add_argument("--from-predictions", type=str, default=None,
                        help="离线模式:只读取该目录/文件下的 predictions jsonl 做统计,不加载模型、不推理")
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


def resolve_stop_ids(tokenizer):
    """返回 tokenizer 中**实际存在**的停止 token id 列表(不硬编码 magic number)。

    Qwen2.5 的助手回答终止符是 <|im_end|>(151645),而 tokenizer.eos_token_id 往往是
    <|endoftext|>(151643)。只把 eos 当停止符,会导致模型回答完不停止、一路跑满
    max_new_tokens(实测 base/sft/rl 的 EOS 命中率为 0%),既慢又会被截断。
    """
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or int(tid) < 0:
            continue
        if unk_id is not None and int(tid) == int(unk_id) and token != getattr(tokenizer, "unk_token", None):
            continue  # token 不存在(被映射成 unk),跳过
        if int(tid) not in ids:
            ids.append(int(tid))
    return ids


def build_prompt(tokenizer, question, prompt_format="chat"):
    """按评测配置生成模型输入文本。

    chat(默认):与 SFT 训练一致 —— 用 tokenizer.apply_chat_template(add_generation_prompt=True),
                最终输入形如 <|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n
    raw       :兼容旧行为(裸问题),仅在显式指定时使用
    """
    if prompt_format == "raw" or getattr(tokenizer, "chat_template", None) is None:
        return question
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_answer(model, tokenizer, question, max_new, prompt_format="chat", stop_ids=None):
    """一次贪心生成;返回文本与统计信息(供进度显示 + 逐条落盘使用)。"""
    prompt = build_prompt(tokenizer, question, prompt_format)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    gen_kwargs = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids  # 生成到 <|im_end|> / <|endoftext|> 立即停止
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    elapsed = time.perf_counter() - t0
    gen_ids = out[0][prompt_len:]
    gen_list = gen_ids.tolist()
    stop_set = {int(t) for t in (stop_ids or [])}
    return {
        "prediction": tokenizer.decode(gen_ids, skip_special_tokens=True).strip(),
        "generated_tokens": int(gen_ids.shape[0]),
        "hit_eos": any(int(tok) in stop_set for tok in gen_list),
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": int(prompt_len),
        "prompt": prompt,
    }


def answer(model, tokenizer, q, max_new, prompt_format="chat", stop_ids=None):
    """兼容旧调用:只返回回答文本。"""
    return generate_answer(model, tokenizer, q, max_new, prompt_format, stop_ids)["prediction"]


def judge_row(row, resp):
    """rule-based 判定(**逻辑与修改前完全一致**,未改动判定规则)。"""
    hit = any(k in resp for k in row["keywords"])
    refused = any(m in resp for m in REFUSAL_MARKERS)
    return hit, refused


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


def format_duration(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 60}m{seconds % 60:02d}s"


def prediction_record(model_name, index, row, gen, prompt_format, max_new):
    """把一条样本的完整结果整理成可离线统计的记录。"""
    hit, refused = judge_row(row, gen["prediction"])
    return {
        "model": model_name,
        "index": index,
        "question": row["q"],
        "gold_answer": row.get("a", ""),
        "intent": row.get("intent"),
        "source": row.get("source"),
        "family_id": row.get("family_id"),
        "is_gold_eval": row.get("source") == "gold_eval",
        "ood": bool(row.get("ood")),
        "keywords": row.get("keywords", []),
        "prediction": gen["prediction"],
        "generated_tokens": gen["generated_tokens"],
        "hit_eos": bool(gen["hit_eos"]),
        "elapsed_seconds": gen["elapsed_seconds"],
        "keyword_hit": bool(hit),
        "refused": bool(refused),
        "correct": bool(refused) if row.get("ood") else bool(hit),
        "prompt_format": prompt_format,
        "max_new_tokens": max_new,
    }


def evaluate(model, tokenizer, rows, max_new, prompt_format="chat", model_name="model",
             predictions_path=None, progress=True, warn_seconds=15.0):
    """逐题生成 + 实时进度 + 逐条落盘(每题 flush,中断不丢已完成结果)。

    返回 (aggregate_metrics, records);records 可直接用于离线统计。
    """
    stop_ids = resolve_stop_ids(tokenizer)
    records = []
    details = []
    slow = 0
    handle = None
    if predictions_path:
        os.makedirs(os.path.dirname(os.path.abspath(predictions_path)), exist_ok=True)
        handle = open(predictions_path, "w", encoding="utf-8")
    if progress:
        print(f"[model={model_name}] 开始评测 {len(rows)} 条 | prompt_format={prompt_format} "
              f"| stop_ids={stop_ids} | max_new_tokens={max_new}")
    t_start = time.perf_counter()
    try:
        for i, r in enumerate(rows, 1):
            gen = generate_answer(model, tokenizer, r["q"], max_new, prompt_format, stop_ids)
            rec = prediction_record(model_name, i - 1, r, gen, prompt_format, max_new)
            records.append(rec)
            details.append((r, rec["prediction"], rec["keyword_hit"], rec["refused"]))
            if handle is not None:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.flush()  # 每完成一条就落盘
            if gen["elapsed_seconds"] > warn_seconds:
                slow += 1
                if progress:
                    print(f"  [WARN] 单题耗时 {gen['elapsed_seconds']:.1f}s (> {warn_seconds:.0f}s) "
                          f"| #{i} intent={r.get('intent')} | q={r['q'][:30]}")
            if progress:
                elapsed = time.perf_counter() - t_start
                avg = elapsed / i
                eta = avg * (len(rows) - i)
                print(
                    f"[model={model_name}] {i}/{len(rows)} | current={gen['elapsed_seconds']:.1f}s "
                    f"| avg={avg:.1f}s | elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}"
                )
    finally:
        if handle is not None:
            handle.close()

    metrics = aggregate_records(records)
    metrics["slow_questions_gt{}s".format(int(warn_seconds))] = slow
    metrics["wall_seconds"] = round(time.perf_counter() - t_start, 2)
    if progress:
        print(
            f"[model={model_name}] 完成 | samples={metrics['samples']} | 领域正确率={metrics['domain_acc']*100:.1f}% "
            f"| 拒答率={metrics['refusal_rate']*100:.1f}% | 总正确率={metrics['overall_acc']*100:.1f}% "
            f"| 平均生成token={metrics['avg_generated_tokens']:.1f} | EOS命中率={metrics['eos_hit_rate']*100:.1f}% "
            f"| 用时={format_duration(metrics['wall_seconds'])}"
            + (f" | 预测已保存:{predictions_path}" if predictions_path else "")
        )
    return metrics, details, records


def aggregate_records(records):
    """从 prediction records 计算全部指标(在线推理与离线统计共用同一实现)。"""
    domain = [r for r in records if not r["ood"]]
    ood = [r for r in records if r["ood"]]
    domain_hit = sum(1 for r in domain if r["keyword_hit"])
    refusal_hit = sum(1 for r in ood if r["refused"])
    n = len(records)
    by_intent = {}
    for intent in sorted({r.get("intent") for r in records}):
        sub = [r for r in records if r.get("intent") == intent]
        d = [r for r in sub if not r["ood"]]
        o = [r for r in sub if r["ood"]]
        by_intent[intent] = {
            "samples": len(sub),
            "domain_total": len(d),
            "domain_hit": sum(1 for r in d if r["keyword_hit"]),
            "domain_acc": round(sum(1 for r in d if r["keyword_hit"]) / max(len(d), 1), 4),
            "ood_total": len(o),
            "refused": sum(1 for r in o if r["refused"]),
            "refusal_rate": (
                round(sum(1 for r in o if r["refused"]) / len(o), 4) if o else None
            ),
        }
    gold = [r for r in records if r.get("is_gold_eval")]
    gold_domain = [r for r in gold if not r["ood"]]
    gold_ood = [r for r in gold if r["ood"]]
    return {
        "samples": n,
        "domain_total": len(domain),
        "domain_hit": domain_hit,
        "domain_acc": domain_hit / max(len(domain), 1),
        "ood_total": len(ood),
        "refusal_hit": refusal_hit,
        "refusal_rate": refusal_hit / max(len(ood), 1),
        "overall_acc": (domain_hit + refusal_hit) / max(n, 1),
        "miss_rate": 1 - domain_hit / max(len(domain), 1),
        "avg_len": sum(len(r["prediction"]) for r in records) / max(n, 1),
        "keyword_hit_rate": sum(1 for r in records if r["keyword_hit"]) / max(n, 1),
        "avg_generated_tokens": sum(r["generated_tokens"] for r in records) / max(n, 1),
        "total_generated_tokens": sum(r["generated_tokens"] for r in records),
        "eos_hit_rate": sum(1 for r in records if r.get("hit_eos")) / max(n, 1),
        "avg_elapsed_seconds": sum(r["elapsed_seconds"] for r in records) / max(n, 1),
        "by_intent": by_intent,
        "gold_eval": {
            "samples": len(gold),
            "domain_total": len(gold_domain),
            "domain_acc": sum(1 for r in gold_domain if r["keyword_hit"]) / max(len(gold_domain), 1),
            "ood_total": len(gold_ood),
            "refusal_rate": sum(1 for r in gold_ood if r["refused"]) / max(len(gold_ood), 1),
            "overall_acc": sum(1 for r in gold if r["correct"]) / max(len(gold), 1),
            "avg_generated_tokens": sum(r["generated_tokens"] for r in gold) / max(len(gold), 1),
            "eos_hit_rate": sum(1 for r in gold if r.get("hit_eos")) / max(len(gold), 1),
        },
    }


def load_predictions(path):
    """读取 predictions jsonl(文件或目录下的所有 *.jsonl)。"""
    records = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".jsonl"):
                records.extend(load_predictions(os.path.join(path, name)))
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_metrics_table(metrics_by_model):
    print("\n" + "=" * 96)
    print(f"{'模型':<16}{'领域正确率':>10}{'拒答率':>9}{'总正确率':>9}{'未命中率':>9}"
          f"{'平均长度':>9}{'平均token':>10}{'EOS命中':>9}")
    for name, m in metrics_by_model.items():
        print(
            f"{name:<16}{m['domain_acc'] * 100:>9.1f}%{m['refusal_rate'] * 100:>8.1f}%"
            f"{m['overall_acc'] * 100:>8.1f}%{m['miss_rate'] * 100:>8.1f}%{m['avg_len']:>9.1f}"
            f"{m['avg_generated_tokens']:>10.1f}{m['eos_hit_rate'] * 100:>8.1f}%"
        )
    print("=" * 96)


def render_report(path, metrics_by_model, dataset_line, examples_by_model=None, judge_scores=None):
    lines = ["# 领域客服评测报告", "", dataset_line, ""]
    lines.append("| 模型 | 领域正确率 | 拒答率 | 总正确率 | 未命中率 | 平均回答长度 | 平均生成token | EOS命中率 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, m in metrics_by_model.items():
        lines.append(
            f"| {name} | {m['domain_acc'] * 100:.1f}% | {m['refusal_rate'] * 100:.1f}% | "
            f"{m['overall_acc'] * 100:.1f}% | {m['miss_rate'] * 100:.1f}% | {m['avg_len']:.1f} | "
            f"{m['avg_generated_tokens']:.1f} | {m['eos_hit_rate'] * 100:.1f}% |"
        )
    lines.append("")
    best = max(metrics_by_model.items(), key=lambda kv: kv[1]["overall_acc"])
    lines.append(f"**最佳:{best[0]}(总正确率 {best[1]['overall_acc'] * 100:.1f}%)**")
    lines.append("")
    lines.append("## gold_eval(人工评测题)单独统计")
    lines.append("")
    lines.append("| 模型 | n | 领域正确率 | 拒答率 | 总正确率 | 平均生成token | EOS命中率 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, m in metrics_by_model.items():
        g = m["gold_eval"]
        lines.append(
            f"| {name} | {g['samples']} | {g['domain_acc'] * 100:.1f}% | {g['refusal_rate'] * 100:.1f}% | "
            f"{g['overall_acc'] * 100:.1f}% | {g['avg_generated_tokens']:.1f} | {g['eos_hit_rate'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## 按 intent 拆分")
    lines.append("")
    for name, m in metrics_by_model.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| intent | n | 领域正确率 | OOD 拒答率 |")
        lines.append("|---|---|---|---|")
        for intent, b in m["by_intent"].items():
            dom = f"{b['domain_acc'] * 100:.1f}%" if b["domain_total"] else "—"
            ref = "—" if b["refusal_rate"] is None else f"{b['refusal_rate'] * 100:.1f}%"
            lines.append(f"| {intent} | {b['samples']} | {dom} | {ref} |")
        lines.append("")
    if judge_scores:
        lines.append("## LLM-as-Judge 语义打分(智谱 GLM,1-5 分)")
        lines.append("")
        lines.append("| 模型 | 平均分 | 抽样数 |")
        lines.append("|---|---|---|")
        for name, (avg, n) in judge_scores.items():
            lines.append(f"| {name} | {avg:.2f} | {n} |" if avg else f"| {name} | 失败 | 0 |")
        lines.append("")
    if examples_by_model:
        for name, details in examples_by_model.items():
            lines.append(f"## {name} 回答示例")
            lines.append("")
            for r, resp, hit, refused in details[:5]:
                tag = "拒答✓" if refused else ("命中✓" if hit else "未命中✗")
                lines.append(f"- Q:{r['q']}")
                lines.append(f"  - {tag} A:{resp[:120]}")
            lines.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    args = parse_args()

    # ---------------- 离线模式:只读 predictions JSONL,不加载模型、不推理 ----------------
    if args.from_predictions:
        records = load_predictions(args.from_predictions)
        if not records:
            raise SystemExit(f"{args.from_predictions} 下没有可用的 predictions jsonl")
        by_model = {}
        for r in records:
            by_model.setdefault(r["model"], []).append(r)
        metrics_by_model = {name: aggregate_records(rs) for name, rs in by_model.items()}
        print(f"离线统计:读取 {len(records)} 条预测 | 模型 {list(metrics_by_model)} | 未运行任何推理")
        print_metrics_table(metrics_by_model)
        render_report(
            args.report,
            metrics_by_model,
            f"数据来源:`{args.from_predictions}`(离线统计,未推理)",
        )
        print(f"报告已保存: {args.report}")
        return 0

    rows = load_test_set(args.data_file)
    if args.ood_only:
        rows = [r for r in rows if r["ood"]]
    if args.limit:
        rows = rows[: args.limit]
    if os.path.abspath(args.data_file) == os.path.abspath("data_eng/output/test.jsonl"):
        print("[WARN] 当前用的是旧测试集 data_eng/output/test.jsonl(存在 family/前缀改写泄漏);"
              "clean 评测请加 --data-file data_eng/output_clean/test.jsonl")
    print(f"评测集:{len(rows)} 条(领域 {sum(1 for r in rows if not r['ood'])} / 拒答 {sum(1 for r in rows if r['ood'])})")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_ids = resolve_stop_ids(tokenizer)
    print(f"stop_ids = {stop_ids} | eos_token_id={tokenizer.eos_token_id} "
          f"| <|im_end|>={tokenizer.convert_tokens_to_ids('<|im_end|>')}")
    debug_prompt = build_prompt(tokenizer, rows[0]["q"], args.prompt_format)
    print(f"[DEBUG] prompt_format={args.prompt_format} | 首条渲染后的输入(转义,已截断): "
          f"{debug_prompt[:160]!r}")

    configs = [("base", None)]
    configs += [(os.path.basename(a.rstrip("/\\")), a) for a in args.adapters]

    metrics_by_model = {}
    examples_by_model = {}
    for name, adapter in configs:
        model = build_model(args.model_name, adapter)
        # ⚠️ 文件名带上 prompt_format:base 在 chat/raw 两种口径下都会评,
        # 若都用 "<name>.jsonl" 后跑的会覆盖先跑的,导致离线统计/Judge 拿到错格式的预测。
        # 只改落盘文件名,不改任何判定逻辑(report 里的 model 名仍是 name)。
        pred_path = os.path.join(args.predictions_dir, f"{name}_{args.prompt_format}.jsonl")
        metrics, details, _ = evaluate(
            model,
            tokenizer,
            rows,
            args.max_new_tokens,
            prompt_format=args.prompt_format,
            model_name=name,
            predictions_path=pred_path,
        )
        metrics_by_model[name] = metrics
        examples_by_model[name] = details
        del model
        torch.cuda.empty_cache()

    # LLM-as-Judge(默认关闭;本轮不启用,不调用任何外部 API)
    judge_scores = {}
    if args.judge:
        print("\nLLM-as-Judge 语义打分(智谱 GLM)...")
        for name, details in examples_by_model.items():
            avg, n = llm_judge_scores(details, args.judge_model, args.judge_limit)
            judge_scores[name] = (avg, n)
            print(f"  {name}: 平均 {avg:.2f} / 5 (抽样 {n} 条)" if avg else f"  {name}: 打分失败")

    print_metrics_table(metrics_by_model)
    dataset_line = (
        f"评测集:`{args.data_file}`({len(rows)} 条) | prompt_format={args.prompt_format} "
        f"| max_new_tokens={args.max_new_tokens} | stop_ids={stop_ids}"
    )
    report_path = render_report(
        args.report, metrics_by_model, dataset_line,
        examples_by_model=examples_by_model, judge_scores=judge_scores or None,
    )
    print(f"评测报告已保存: {report_path}")
    print(f"逐条预测已保存: {os.path.join(args.predictions_dir, '<model>.jsonl')}")
    print(f"离线复算(不推理): python sft/eval_domain.py --from-predictions {args.predictions_dir} "
          f"--report {args.report}")


if __name__ == "__main__":
    main()
