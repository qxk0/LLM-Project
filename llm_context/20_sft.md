# 监督微调模块 sft/ (SFT + 评测)

> 生成时间：2026-10-14 17:27:04

## 目录结构

```text
sft/
  eval_bench.py
  eval_breakdown.py
  eval_domain.py
  eval_stratified.py
  infer_sft.py
  prompt_ablation.py
  train_sft.py
```

## 文件内容

### `sft/eval_bench.py`

```python
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
```

### `sft/eval_breakdown.py`

```python
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
```

### `sft/eval_domain.py`

```python
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
```

### `sft/eval_stratified.py`

```python
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
```

### `sft/infer_sft.py`

```python
"""用微调后的 LoRA 模型对话,并和基座模型做对比(生成 comparison.md)。"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_PROMPTS = [
    "用三句话介绍一下什么是大语言模型。",
    "太阳为什么从东边升起?",
    "写一首关于春天的五言绝句。",
    "1+1 等于几?请给出理由。",
    "怎么养成早起的好习惯?给出三条建议。",
]


def parse_args():
    parser = argparse.ArgumentParser(description="SFT 模型推理与对比")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", type=str, default="models/sft")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--compare", action="store_true", help="同时用基座模型回答做对比")
    parser.add_argument("--report", type=str, default="models/sft/comparison.md")
    return parser.parse_args()


def build_model(model_name, adapter=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model


def chat(model, tokenizer, prompt, max_new_tokens, temperature):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return answer.strip()


def main():
    args = parse_args()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先记录基座模型回答,再挂上 LoRA 回答(避免同时占两份显存)
    base_answers = {}
    sft_answers = {}
    if args.compare:
        print("正在用基座模型回答...")
        base_model = build_model(args.model_name)
        for p in DEFAULT_PROMPTS:
            base_answers[p] = chat(base_model, tokenizer, p, args.max_new_tokens, args.temperature)
            print(f"  [base] {p[:20]}... -> {base_answers[p][:60]}")

    print("正在用微调后模型回答...")
    sft_model = build_model(args.model_name, args.adapter)
    for p in DEFAULT_PROMPTS:
        sft_answers[p] = chat(sft_model, tokenizer, p, args.max_new_tokens, args.temperature)
        print(f"  [sft ] {p[:20]}... -> {sft_answers[p][:60]}")

    # 生成对比报告
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    lines = ["# SFT 前后效果对比\n"]
    for i, p in enumerate(DEFAULT_PROMPTS, 1):
        lines.append(f"## 问题 {i}:{p}\n")
        if args.compare:
            lines.append(f"**基座模型:**\n\n{base_answers[p]}\n")
        lines.append(f"**SFT 微调后:**\n\n{sft_answers[p]}\n")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"对比报告已保存: {args.report}")


if __name__ == "__main__":
    main()
```

### `sft/prompt_ablation.py`

```python
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
```

### `sft/train_sft.py`

```python
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
import random

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
```
