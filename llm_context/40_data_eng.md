# 数据工程模块 data_eng/ (数据集构建 + 质检 + 评审)

> 生成时间：2026-10-14 17:27:04

## 目录结构

```text
data_eng/
  GRPO_EXPERIMENT_ARCHIVE.md
  INTERVIEW_FACTS.md
  PROJECT_FINAL_AUDIT.md
  build_dataset.py
  evolve.py
  gold_eval_seeds.py
  kb.py
  leakage_audit.py
  run_llm_judge.py
  run_pipeline.py
  seeds.py
```

## 文件内容

### `data_eng/build_dataset.py`

````python
"""领域后训练数据构建管道(样本血缘 lineage + family/group 级切分)。

流程:
  1. 种子:人工基于知识库编写高质量问答(seeds.py,含拒答样本)
  2. 扩展:模板 + 实体替换,把种子扩展到几十倍(问题多样性的来源)
  3. 前缀改写:每条问题生成 5 个口语化变体(含原样)
  4. 合并额外数据:evolved(LLM 蒸馏) / onpolicy(RFT 自采样),并尽力恢复其血缘
  5. 清洗:规范化、精确去重、长度过滤
  6. 切分:按血缘 group 切分(而不是按样本随机切分)
  7. 审计:输出 stats.json + leakage_audit.json + DATASET_LEAKAGE_AUDIT.md

两个层级的血缘概念(见 leakage_audit.py 的说明):
  * family = (intent, base_question):同一问题的不同口语化改写属于同一 family
  * group  = 血缘连通分量:同一 seed / 同一实体(产品、小料、OOD 主题)的
             seed + template + paraphrase + evolved + onpolicy 全部锁进同一个 split

切分规则(强约束):
  * 同一个 group 的所有样本只能进 train / val / test 其中一个
  * test 只允许 gold 数据(source ∈ {seed, template, paraphrase})且血缘可追溯
  * evolved / onpolicy / 血缘不可追溯的样本一律只进 train(保守策略)

输出:data_eng/output_clean/ 下的 train.jsonl / val.jsonl / test.jsonl
      旧的 data_eng/output/ 不会被本脚本修改(保留审计价值)。
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kb import PRODUCTS, REFUSAL_TEMPLATES, TOPPINGS
from seeds import SEEDS

import leakage_audit as audit_lib
from gold_eval_seeds import (
    GOLD_EVAL_SEEDS,
    UNSUPPORTED_GOLD_SCENARIOS,
    validate_gold_eval_seeds,
)
from leakage_audit import (
    MODEL_GENERATED_SOURCES,
    base_question,
    family_id_of,
    file_sha256,
    load_jsonl,
    normalize_text,
    sample_id_of,
    stable_id,
)

OUT_DIR_DEFAULT = "data_eng/output_clean"
LEGACY_OUT_DIR = "data_eng/output"
RANDOM_SEED = 42
SPLIT_TARGETS = {"train": 0.8, "val": 0.1, "test": 0.1}
GOLD_SOURCES = ("seed", "template", "paraphrase")

# ---------- 模板扩展 ----------

PRICE_TEMPLATES = [
    "你们家{name}多少钱?",
    "{name}多少钱一杯?",
    "请问{name}的价格是?",
    "{name}要多少钱?",
    "你们{name}卖多少钱?",
    "{name}怎么卖的?",
    "我想知道{name}多少钱",
    "一杯{name}多少钱?",
]

RECOMMEND_TEMPLATES = [
    "推荐一下{tag}的饮品",
    "有什么{tag}推荐吗?",
    "我想喝{tag}的,有什么推荐?",
    "有没有{tag}的好喝的?",
    "你们家{tag}的好喝吗?",
    "想试试{tag}的,推荐哪个?",
]

CUSTOMIZE_SUGAR_TEMPLATES = [
    "可以把{name}做成无糖吗?",
    "{name}能选半糖吗?",
    "{name}可以做少糖吗?",
]

CUSTOMIZE_ICE_TEMPLATES = [
    "{name}能去冰吗?",
    "{name}可以做热的吗?",
]

OOD_TOPICS = [
    "修手机", "装系统", "退机票", "订酒店", "买保险", "装修房子", "考驾照",
    "报培训班", "办签证", "代开发票", "卖二手车", "查违章", "打疫苗", "缴水电费",
    "修空调", "配钥匙", "寄快递", "开锁", "搬家", "找家教",
]

OOD_QUESTION_TEMPLATES = [
    "你们能{thing}吗?",
    "帮我{thing}一下",
    "请问哪里可以{thing}?",
    "你会{thing}吗?",
]

# 通用前缀改写:对全部样本做口语化增强,提升问题多样性
PARAPHRASE_TEMPLATES = [
    "{q}",
    "请问{q}",
    "你好,{q}",
    "麻烦问一下,{q}",
    "亲,{q}",
]


def normalize(text):
    """清洗:统一标点、去空白,便于去重。"""
    text = text.replace("?", "?").replace("?", "?").strip()
    text = re.sub(r"\s+", "", text)
    return text


# ---------------------------------------------------------------------------
# 以下 expand_seeds_legacy / clean_and_split_legacy / main_legacy 是 v1 实现,
# 存在"样本级随机切分导致 family 跨 split"的问题,仅保留作对照,入口已不再调用。
# 现行实现见文件末尾的 v2 管道(带样本血缘 + group 级切分)。
# ---------------------------------------------------------------------------


def expand_seeds_legacy():
    """(legacy) 种子 + 模板扩展,返回 (样本列表, 来源统计)。"""
    samples = []
    source = Counter()

    # 1. 种子原样保留
    for s in SEEDS:
        samples.append(s)
        source["seed"] += 1

    # 2. 价格模板 × 全部产品
    for p in PRODUCTS:
        for tpl in PRICE_TEMPLATES:
            samples.append(
                {
                    "intent": "price",
                    "q": tpl.format(name=p["name"]),
                    "a": f"{p['name']} {p['price']} 元一杯。",
                    "keywords": [str(p["price"])],
                    "ood": False,
                }
            )
            source["price_template"] += 1

    # 3. 推荐模板 × 产品标签
    for p in PRODUCTS:
        for tag in p["tags"][:2]:
            for tpl in RECOMMEND_TEMPLATES:
                samples.append(
                    {
                        "intent": "recommend",
                        "q": tpl.format(tag=tag),
                        "a": f"推荐{p['name']},{p['price']} 元一杯。",
                        "keywords": [p["name"]],
                        "ood": False,
                    }
                )
                source["recommend_template"] += 1

    # 4. 定制模板 × 产品(选前 4 个热门款)
    hot = PRODUCTS[:4]
    for p in hot:
        for tpl in CUSTOMIZE_SUGAR_TEMPLATES:
            samples.append(
                {
                    "intent": "customize",
                    "q": tpl.format(name=p["name"]),
                    "a": f"可以,{p['name']} 可以选择糖度,包括无糖和半糖。",
                    "keywords": ["无糖", "半糖"],
                    "ood": False,
                }
            )
            source["customize_template"] += 1
        for tpl in CUSTOMIZE_ICE_TEMPLATES:
            samples.append(
                {
                    "intent": "customize",
                    "q": tpl.format(name=p["name"]),
                    "a": f"可以,{p['name']} 支持去冰;部分饮品可做热饮。",
                    "keywords": ["去冰"],
                    "ood": False,
                }
            )
            source["customize_template"] += 1

    # 5. 小料价格模板
    for topping, price in TOPPINGS:
        samples.append(
            {
                "intent": "price",
                "q": f"加{topping}多少钱?",
                "a": f"加{topping}加 {price} 元。",
                "keywords": [str(price)],
                "ood": False,
            }
        )
        source["topping_template"] += 1

    # 6. 知识库外问题(拒答样本,随机抽部分主题)
    rng = random.Random(RANDOM_SEED)
    for topic in rng.sample(OOD_TOPICS, k=16):
        tpl = rng.choice(OOD_QUESTION_TEMPLATES)
        q = tpl.format(thing=topic)
        a = rng.choice(REFUSAL_TEMPLATES)
        samples.append(
            {"intent": "ood", "q": q, "a": a, "keywords": ["抱歉", "无法回答", "服务范围"], "ood": True}
        )
        source["ood_template"] += 1

    # 7. 前缀改写增强:每条问题生成多个口语化变体
    augmented = []
    for s in samples:
        for tpl in PARAPHRASE_TEMPLATES:
            augmented.append({**s, "q": tpl.format(q=s["q"])})
    source["paraphrase"] = len(augmented) - len(samples)
    samples = augmented

    return samples, source


def clean_and_split_legacy(samples):
    """(legacy) 去重 + 过滤 + 按意图分层随机切分。"""
    seen = set()
    unique = []
    dropped = 0
    for s in samples:
        qn = normalize(s["q"])
        an = normalize(s["a"])
        key = (qn, an)  # 按(问题,答案)去重:同一问题的不同答案保留,onpolicy 变体才有效
        if key in seen:
            dropped += 1
            continue
        if not (4 <= len(s["q"]) <= 80) or not (5 <= len(s["a"]) <= 200):
            dropped += 1
            continue
        seen.add(key)
        unique.append(s)

    # 按意图分层:每个意图内随机 80/10/10
    rng = random.Random(RANDOM_SEED)
    by_intent = {}
    for s in unique:
        by_intent.setdefault(s["intent"], []).append(s)

    train, val, test = [], [], []
    for group in by_intent.values():
        rng.shuffle(group)
        n = len(group)
        train += group[: int(n * 0.8)]
        val += group[int(n * 0.8) : int(n * 0.9)]
        test += group[int(n * 0.9) :]
    return unique, dropped, train, val, test


def main_legacy():
    parser = argparse.ArgumentParser(description="构建领域后训练数据")
    parser.add_argument("--extra-file", action="append", default=[],
                        help="额外的数据文件(可多次指定,如 evolve/onpolicy 产物),合并后统一清洗切分")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    samples, source = expand_seeds()
    for extra_path in args.extra_file:
        with open(extra_path, encoding="utf-8") as f:
            extra = [json.loads(line) for line in f if line.strip()]
        samples.extend(extra)
        source[os.path.splitext(os.path.basename(extra_path))[0]] = len(extra)
        print(f"合并额外数据 {len(extra)} 条: {extra_path}")
    unique, dropped, train, val, test = clean_and_split(samples)

    for name, rows in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "raw_count": len(samples),
        "dedup_dropped": dropped,
        "final_count": len(unique),
        "expansion_ratio": round(len(unique) / len(SEEDS), 1),
        "source": dict(source),
        "intent_dist": dict(Counter(s["intent"] for s in unique)),
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "ood_ratio": round(sum(1 for s in unique if s["ood"]) / len(unique), 3),
    }
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"原始样本: {len(samples)} | 去重清洗后: {len(unique)}(丢弃 {dropped})")
    print(f"扩展倍数(相对种子 {len(SEEDS)} 条): {stats['expansion_ratio']}x")
    print("意图分布:", dict(stats["intent_dist"]))
    print(f"切分: train={len(train)} val={len(val)} test={len(test)} | 拒答占比 {stats['ood_ratio']:.1%}")
    print(f"输出目录: {OUT_DIR}")


# ===========================================================================
# v2 管道:样本血缘(sample lineage)+ family / group 级切分
# ===========================================================================

import difflib


def _infer_entity_key(row):
    """从文本推断样本所属"实体"(产品 / 小料 / OOD 主题)。

    实体是 group 切分的第二把锁:同一产品的不同问法(不同模板 + 口语化改写 +
    蒸馏改写 + 自采样)必须落在同一个 split。
    """
    q = str(row.get("q", ""))
    text = f"{q} {row.get('a', '')}"
    for p in PRODUCTS:
        if p["name"] in text:
            return f"product::{p['name']}"
    for topic in OOD_TOPICS:
        if topic in q:
            return f"ood_topic::{topic}"
    for topping, _price in TOPPINGS:
        if f"加{topping}" in text or (topping in q and "加" in q):
            return f"topping::{topping}"
    return None


def _seed_entity_key(seed, seed_id):
    return _infer_entity_key(seed) or f"seed::{seed_id}"


def _base_samples():
    """第 1~6 步:种子 + 模板扩展(尚未前缀改写),每条都带 lineage 元数据。

    注意:模板循环的顺序、模板内容、OOD 主题抽样方式都与 v1 完全一致,
    以保证生成的问题字符串与历史产物(evolved / onpolicy)逐字可比。
    """
    rows = []
    stats = Counter()

    # 1. 种子原样保留(seed_id 按 seeds.py 顺序编号,稳定且可读)
    for i, s in enumerate(SEEDS, 1):
        sid = f"seed_{i:03d}"
        rows.append(
            {
                "intent": s["intent"],
                "q": s["q"],
                "a": s["a"],
                "keywords": list(s["keywords"]),
                "ood": bool(s["ood"]),
                "source": "seed",
                "variant_type": "seed",
                "template_name": "seed",
                "seed_id": sid,
                "entity_key": _seed_entity_key(s, sid),
            }
        )
        stats["seed"] += 1

    # 2. 价格模板 × 全部产品
    for p in PRODUCTS:
        for tpl in PRICE_TEMPLATES:
            rows.append(
                {
                    "intent": "price",
                    "q": tpl.format(name=p["name"]),
                    "a": f"{p['name']} {p['price']} 元一杯。",
                    "keywords": [str(p["price"])],
                    "ood": False,
                    "source": "template",
                    "variant_type": "template",
                    "template_name": "price_template",
                    "seed_id": None,
                    "entity_key": f"product::{p['name']}",
                }
            )
            stats["price_template"] += 1

    # 3. 推荐模板 × 产品标签
    for p in PRODUCTS:
        for tag in p["tags"][:2]:
            for tpl in RECOMMEND_TEMPLATES:
                rows.append(
                    {
                        "intent": "recommend",
                        "q": tpl.format(tag=tag),
                        "a": f"推荐{p['name']},{p['price']} 元一杯。",
                        "keywords": [p["name"]],
                        "ood": False,
                        "source": "template",
                        "variant_type": "template",
                        "template_name": "recommend_template",
                        "seed_id": None,
                        "entity_key": f"product::{p['name']}",
                    }
                )
                stats["recommend_template"] += 1

    # 4. 定制模板 × 产品(选前 4 个热门款)
    hot = PRODUCTS[:4]
    for p in hot:
        for tpl in CUSTOMIZE_SUGAR_TEMPLATES:
            rows.append(
                {
                    "intent": "customize",
                    "q": tpl.format(name=p["name"]),
                    "a": f"可以,{p['name']} 可以选择糖度,包括无糖和半糖。",
                    "keywords": ["无糖", "半糖"],
                    "ood": False,
                    "source": "template",
                    "variant_type": "template",
                    "template_name": "customize_template",
                    "seed_id": None,
                    "entity_key": f"product::{p['name']}",
                }
            )
            stats["customize_template"] += 1
        for tpl in CUSTOMIZE_ICE_TEMPLATES:
            rows.append(
                {
                    "intent": "customize",
                    "q": tpl.format(name=p["name"]),
                    "a": f"可以,{p['name']} 支持去冰;部分饮品可做热饮。",
                    "keywords": ["去冰"],
                    "ood": False,
                    "source": "template",
                    "variant_type": "template",
                    "template_name": "customize_template",
                    "seed_id": None,
                    "entity_key": f"product::{p['name']}",
                }
            )
            stats["customize_template"] += 1

    # 5. 小料价格模板
    for topping, price in TOPPINGS:
        rows.append(
            {
                "intent": "price",
                "q": f"加{topping}多少钱?",
                "a": f"加{topping}加 {price} 元。",
                "keywords": [str(price)],
                "ood": False,
                "source": "template",
                "variant_type": "template",
                "template_name": "topping_template",
                "seed_id": None,
                "entity_key": f"topping::{topping}",
            }
        )
        stats["topping_template"] += 1

    # 6. 知识库外问题(拒答样本,随机抽部分主题)
    rng = random.Random(RANDOM_SEED)
    for topic in rng.sample(OOD_TOPICS, k=16):
        tpl = rng.choice(OOD_QUESTION_TEMPLATES)
        q = tpl.format(thing=topic)
        a = rng.choice(REFUSAL_TEMPLATES)
        rows.append(
            {
                "intent": "ood",
                "q": q,
                "a": a,
                "keywords": ["抱歉", "无法回答", "服务范围"],
                "ood": True,
                "source": "template",
                "variant_type": "template",
                "template_name": "ood_template",
                "seed_id": None,
                "entity_key": f"ood_topic::{topic}",
            }
        )
        stats["ood_template"] += 1

    for r in rows:
        r["lineage_confidence"] = "gold"
        r["origin_file"] = "generated"
    return rows, stats


def expand_seeds():
    """种子 + 模板扩展 + 前缀改写;返回带完整 lineage 元数据的样本列表。"""
    base, stats = _base_samples()
    rows = []
    for s in base:
        for k, tpl in enumerate(PARAPHRASE_TEMPLATES):
            variant = dict(s)
            variant["q"] = tpl.format(q=s["q"])
            variant["paraphrase_id"] = k
            variant["parent_q"] = s["q"]
            if k > 0:  # k=0 是原样,保留 seed / template 身份
                variant["source"] = "paraphrase"
                variant["variant_type"] = "paraphrase"
            rows.append(variant)
    stats["paraphrase_added"] = len(rows) - len(base)
    return rows, stats


def build_gold_eval_rows():
    """人工 Gold Evaluation Seed → 评测专用样本。

    与普通 gold 样本的区别:
      * source="gold_eval",eval_only=True
      * 不参与 prefix paraphrase / 模板扩展,一条题目就是一条样本
      * 只会被放进 test,绝不进 train/val(见 build_groups / split_groups)
    """
    rows = []
    for item in GOLD_EVAL_SEEDS:
        row = {
            "intent": item["intent"],
            "q": item["q"],
            "a": item["a"],
            "keywords": list(item["keywords"]),
            "ood": bool(item["ood"]),
            "source": "gold_eval",
            "variant_type": "gold_eval",
            "template_name": "gold_eval",
            "seed_id": item["id"],
            "eval_id": item["id"],
            "eval_only": True,
            "lineage_confidence": "gold_eval",
            "origin_file": "data_eng/gold_eval_seeds.py",
            "paraphrase_id": 0,
            "parent_q": item["q"],
            "candidate_seed_ids": [],
            "knowledge_source": item["knowledge_source"],
            "novelty_reason": item["novelty_reason"],
            "entity_key": None,
        }
        row["entity_key"] = _infer_entity_key(row)
        rows.append(row)
    return rows


def freeze_previous_split(prev_dir):
    """读取上一轮 output_clean 的 train/val/test,返回 sample_id -> split。"""
    if not prev_dir or not os.path.isdir(prev_dir):
        return None
    mapping = {}
    for name in ("train", "val", "test"):
        path = os.path.join(prev_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            return None
        for r in load_jsonl(path):
            if r.get("sample_id"):
                mapping[r["sample_id"]] = name
    return mapping or None


def _apply_ids(row):
    """补齐 sample_id / family_id / base_question 等稳定字段。"""
    row["base_question"] = base_question(row["q"])
    if row.get("lineage_confidence") == "unknown":
        row["family_id"] = "unknown_family"
    else:
        row["family_id"] = family_id_of(row["intent"], row["base_question"])
    row["sample_id"] = sample_id_of(row["intent"], row["q"], row["a"])
    row.setdefault("parent_q", row["q"])
    row.setdefault("paraphrase_id", 0)
    row.setdefault("entity_key", None)
    row.setdefault("origin_file", "generated")
    row.setdefault("candidate_seed_ids", [])
    return row


def _similarity(a, b):
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def match_evolved_to_seed(row, margin=0.06):
    """evolved 数据没有 lineage 字段,用 (intent, ood, keywords) + 文本相似度反推父 seed。

    返回 (seed_id | None, candidate_seed_ids, detail):
      * 唯一候选 / 相似度明显领先 → 返回 seed_id
      * 多候选无法区分            → 返回 (None, 全部候选),由上层把这些候选
                                    保守地一起锁进同一个 split(绝不进 val/test)
    """
    seeds = [(f"seed_{i:03d}", s) for i, s in enumerate(SEEDS, 1)]
    same = [
        (sid, s)
        for sid, s in seeds
        if s["intent"] == row["intent"] and bool(s["ood"]) == bool(row["ood"])
    ]
    exact = [(sid, s) for sid, s in same if set(s["keywords"]) == set(row["keywords"])]
    pool = exact or [(sid, s) for sid, s in same if set(s["keywords"]) & set(row["keywords"])]
    if not pool:
        return None, [], {"reason": "no_candidate"}
    if len(pool) == 1:
        sid, s = pool[0]
        return sid, [sid], {
            "reason": "unique_candidate",
            "a_sim": round(_similarity(row["a"], s["a"]), 3),
            "q_sim": round(_similarity(row["q"], s["q"]), 3),
        }
    scored = []
    for sid, s in pool:
        score = 0.55 * _similarity(row["a"], s["a"]) + 0.45 * _similarity(row["q"], s["q"])
        scored.append((score, sid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    detail = {
        "reason": "scored",
        "scores": [[sid, round(sc, 3)] for sc, sid in scored],
        "margin": round(scored[0][0] - scored[1][0], 3),
    }
    if scored[0][0] - scored[1][0] >= margin:
        return scored[0][1], [x[1] for x in scored], detail
    return None, [x[1] for x in scored], detail


def attach_extra_rows(extra_path, gold_rows, q_index, base_index):
    """把一份额外数据(evolved / onpolicy)挂到已有血缘上。

    匹配优先级:同一问题精确匹配 > 同一 base question > evolved 的 seed 反推 >
    实体推断。匹配不上 → lineage_confidence="unknown",family_id="unknown_family"
    (只进 train)。
    """
    raw_rows = load_jsonl(extra_path)
    default_source = os.path.splitext(os.path.basename(extra_path))[0]
    if default_source not in MODEL_GENERATED_SOURCES:
        default_source = "external"

    added, invalid, lineage_log = [], 0, []
    for raw in raw_rows:
        missing = [f for f in ("q", "a", "intent", "keywords", "ood") if raw.get(f) in (None, "")]
        if missing:
            invalid += 1
            lineage_log.append(
                {"origin_file": extra_path.replace("\\", "/"), "status": "invalid", "missing_fields": missing}
            )
            continue

        source = raw.get("source")
        if source not in MODEL_GENERATED_SOURCES:
            source = default_source

        row = {
            "intent": raw["intent"],
            "q": str(raw["q"]),
            "a": str(raw["a"]),
            "keywords": list(raw["keywords"]),
            "ood": bool(raw["ood"]),
            "source": source,
            "variant_type": source,
            "template_name": f"extra:{default_source}",
            "origin_file": extra_path.replace("\\", "/"),
            "seed_id": None,
            "entity_key": None,
            "paraphrase_id": 0,
            "parent_q": str(raw["q"]),
            "candidate_seed_ids": [],
        }
        info = {"origin_file": row["origin_file"], "source": source, "q": row["q"]}

        idx = q_index.get(normalize_text(row["q"]))
        matched_by = "exact_question" if idx is not None else None
        if idx is None:
            idx = base_index.get(normalize_text(base_question(row["q"])))
            matched_by = "base_question" if idx is not None else None

        if idx is not None:
            parent = gold_rows[idx]
            row["entity_key"] = parent.get("entity_key")
            row["seed_id"] = parent.get("seed_id")
            row["parent_q"] = parent["q"]
            row["lineage_confidence"] = matched_by
            info.update(
                status="matched",
                matched_by=matched_by,
                parent_sample_id=parent["sample_id"],
                parent_q=parent["q"],
                seed_id=row["seed_id"],
            )
        elif raw.get("seed_id"):
            # 新版 evolve.py / rft.py 产出的数据自带血缘,直接采信(无需反推)
            row["seed_id"] = raw["seed_id"]
            row["lineage_confidence"] = "self_reported"
            info.update(status="matched", matched_by="self_reported", seed_id=raw["seed_id"])
        elif source == "evolved":
            seed_id, cands, detail = match_evolved_to_seed(row)
            info.update(candidates=cands, detail=detail)
            if seed_id:
                row["seed_id"] = seed_id
                row["lineage_confidence"] = "seed_match"
                info.update(status="matched", matched_by="seed_match", seed_id=seed_id)
            elif cands:
                row["candidate_seed_ids"] = cands
                row["lineage_confidence"] = "ambiguous_union"
                info.update(status="ambiguous", matched_by="ambiguous_union")
            else:
                row["lineage_confidence"] = "unknown"
                info.update(status="unknown", matched_by="none")
        else:
            row["lineage_confidence"] = "entity_only" if _infer_entity_key(row) else "unknown"
            info.update(
                status="matched" if row["lineage_confidence"] == "entity_only" else "unknown",
                matched_by=row["lineage_confidence"],
            )

        if row["entity_key"] is None:
            row["entity_key"] = _infer_entity_key(row)
        _apply_ids(row)
        info["family_id"] = row["family_id"]
        info["entity_key"] = row["entity_key"]
        info["lineage_confidence"] = row["lineage_confidence"]
        added.append(row)
        lineage_log.append(info)
    return added, invalid, lineage_log


def dedup_and_filter(rows):
    """按 (normalized question, normalized answer) 去重 + 长度过滤。

    保留 (问题,答案) 对语义:同一问题的不同答案(如 onpolicy 变体)不会被误删。
    """
    seen = set()
    unique = []
    dropped = 0
    for r in rows:
        key = (normalize_text(r["q"]), normalize_text(r["a"]))
        if key in seen:
            dropped += 1
            continue
        if not (4 <= len(r["q"]) <= 80) or not (5 <= len(r["a"]) <= 200):
            dropped += 1
            continue
        seen.add(key)
        unique.append(r)
    return unique, dropped


class UnionFind:
    """字符串节点的并查集,用于把 family 合并成血缘 group。"""

    def __init__(self):
        self.parent = {}

    def add(self, node):
        self.parent.setdefault(node, node)

    def find(self, node):
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:  # 路径压缩
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # 取字典序小者作根,保证确定性


def build_groups(rows):
    """聚成血缘 group,合并规则(均为同 intent 内合并,避免把不同任务的问法混在一起):
      * 同一 family_id        (同一基础问题的所有改写)
      * 同一 (实体, intent)   (同一产品的同类问题,如"杨枝甘露 价格"的所有问法)
      * 同一 (seed, intent)   (seed 及其所有派生样本)
      * 上述 + 歧义 evolved 的全部候选 seed

    例外:gold_eval(人工评测种子)永远单独成组 —— 绝不允许和训练家族合并,
    否则评测题会被一起拖进 train。
    """
    uf = UnionFind()
    for r in rows:
        fam = ("fam", r["family_id"])
        uf.add(fam)
        if r.get("eval_only"):
            continue
        intent = r["intent"]
        if r.get("entity_key"):
            ent = ("ent", r["entity_key"], intent)
            uf.add(ent)
            uf.union(fam, ent)
        if r.get("seed_id"):
            seed = ("seed", r["seed_id"], intent)
            uf.add(seed)
            uf.union(fam, seed)
        for cand in r.get("candidate_seed_ids") or []:
            seed = ("seed", cand, intent)
            uf.add(seed)
            uf.union(fam, seed)

    components = defaultdict(list)
    for i, r in enumerate(rows):
        components[uf.find(("fam", r["family_id"]))].append(i)

    group_of = {}
    group_meta = {}
    for idxs in components.values():
        families = sorted({rows[i]["family_id"] for i in idxs})
        gid = stable_id("grp_", *families)
        entities = Counter(rows[i]["entity_key"] for i in idxs if rows[i].get("entity_key"))
        intents = Counter(rows[i]["intent"] for i in idxs)
        clean = all(
            rows[i]["source"] in GOLD_SOURCES and rows[i].get("lineage_confidence") != "unknown"
            for i in idxs
        )
        group_meta[gid] = {
            "size": len(idxs),
            "ood_size": sum(1 for i in idxs if rows[i].get("ood")),
            "intents": set(intents),
            "intent_main": intents.most_common(1)[0][0],
            "entity_main": entities.most_common(1)[0][0] if entities else None,
            "n_families": len(families),
            "clean": clean,
        }
        for i in idxs:
            group_of[i] = gid
    return group_of, group_meta


CLEAN_TEST_RATIO = 0.6  # clean group 在 val/test 之间的分配比例(test 占 60%)


def split_groups(
    rows,
    group_of,
    group_meta,
    seed=RANDOM_SEED,
    clean_test_ratio=CLEAN_TEST_RATIO,
    frozen=None,
):
    """group 级切分:同一 group 只能进一个 split;只有 clean group 能进 val/test。

    clean group(内部不含任何 evolved / onpolicy / unknown 样本)是唯一可以进
    val/test 的数据,属于稀缺资源,因此 **全部分配给 val/test**,不再分给 train;
    train 由 non-clean group(以及它们内部仍然合法的 gold 样本)构成。
    这是本数据集无法做到 80/10/10 的根本原因,已在审计报告中如实说明。

    frozen 不为 None 时进入"冻结模式":上一轮已经确定的 group 保持原 split
    (不重新随机打乱),只有 gold_eval 这种新增的评测 group 会被放进 test。

    固定种子打乱 group 顺序 → 按"已填比例"贪心分配 → 再做一次 intent 覆盖修复。
    """
    total = len(rows)
    gids = sorted(group_meta)
    random.Random(seed).shuffle(gids)  # 固定种子 → 完全可复现

    assign = {}
    current = Counter()

    # ---- Phase 0:冻结上一轮 split;gold_eval 强制进 test ----
    by_group = defaultdict(list)
    for i in range(len(rows)):
        by_group[group_of[i]].append(i)

    unfrozen = []
    for gid in sorted(group_meta):
        idxs = by_group[gid]
        if any(rows[i].get("eval_only") for i in idxs):
            assign[gid] = "test"  # 人工 Gold Evaluation Seed:只允许进 test
            current["test"] += group_meta[gid]["size"]
            continue
        if frozen:
            seen = {frozen[rows[i]["sample_id"]] for i in idxs if rows[i]["sample_id"] in frozen}
            if len(seen) == 1:
                assign[gid] = seen.pop()
                current[assign[gid]] += group_meta[gid]["size"]
            elif len(seen) > 1:
                raise SystemExit(
                    f"[冻结 split 冲突] group {gid} 的样本在上一轮 split 中分布在 {seen},拒绝继续"
                )
            else:
                unfrozen.append(gid)
    if frozen and unfrozen:
        raise SystemExit(
            f"[冻结 split 失败] 有 {len(unfrozen)} 个非 gold_eval 的 group 不在上一轮 split 中"
            f"(例如 {unfrozen[:3]});请确认输入数据没有变化,或去掉 --reuse-split-from 重新切分"
        )

    clean_total = sum(m["size"] for m in group_meta.values() if m["clean"])
    # 目标容量按比例计算。修复点:旧实现把 train 容量写死为"非 clean group 总量",
    # 当数据集里一条模型生成数据都没有(纯 gold 数据集)时会算出 train=0;
    # 现在 clean group 三个 split 都可进,非 clean group 仍然只能进 train,
    # 依靠"已填比例"贪心让 val/test 优先拿 clean 数据。
    caps = {s: max(SPLIT_TARGETS[s] * total, 1) for s in ("train", "val", "test")}

    def fill_ratio(split, extra=0):
        """已填比例(含本 group):让 val/test 按 clean_test_ratio 同步增长。"""
        return (current[split] + extra) / caps[split]

    if not frozen:
        # Phase A:先保证 test 的 intent 覆盖 —— 每个"有 clean group 可用"的 intent,
        # 取最小的那个 group 进 test(绝不会取到含 evolved/onpolicy 的 group)。
        available_intents = sorted({i for m in group_meta.values() if m["clean"] for i in m["intents"]})
        for intent in available_intents:
            candidates = [
                g
                for g, m in group_meta.items()
                if m["clean"] and intent in m["intents"] and g not in assign
            ]
            if not candidates:
                continue
            g_cov = min(candidates, key=lambda g: (group_meta[g]["size"], g))
            assign[g_cov] = "test"
            current["test"] += group_meta[g_cov]["size"]

        # Phase B:其余 group 按"已填比例"分配(clean → val/test;non-clean → train)
        for gid in gids:
            if gid in assign:
                continue
            allowed = ["train", "val", "test"] if group_meta[gid]["clean"] else ["train"]
            best = min(
                allowed,
                key=lambda s: (
                    fill_ratio(s, group_meta[gid]["size"]),
                    0 if s == "test" else 1,
                ),
            )
            assign[gid] = best
            current[best] += group_meta[gid]["size"]

        # OOD 平衡:test 中 OOD 占比若超过 50%,把最小的"纯 OOD clean group"挪到 val,
        # 保证 clean test 同时能测"领域正确率"和"拒答率"(至少保留 1 个 OOD group)。
        while True:
            test_groups = [g for g in group_meta if assign[g] == "test"]
            test_size = sum(group_meta[g]["size"] for g in test_groups)
            test_ood = sum(group_meta[g]["ood_size"] for g in test_groups)
            if test_size == 0 or test_ood / test_size <= 0.5:
                break
            ood_only = [
                g for g in test_groups if group_meta[g]["clean"] and group_meta[g]["intents"] <= {"ood"}
            ]
            if len(ood_only) <= 1:
                break
            g_move = min(ood_only, key=lambda g: (group_meta[g]["size"], g))
            assign[g_move] = "val"

    splits = {"train": [], "val": [], "test": []}
    for i, r in enumerate(rows):
        splits[assign[group_of[i]]].append(r)
    for name in splits:
        splits[name].sort(key=lambda r: (r["intent"], r["sample_id"]))
    return splits, assign


def legacy_baseline():
    """对旧 data_eng/output/ 复算泄漏指标,作为"修复前"对照(只读)。"""
    if not os.path.isdir(LEGACY_OUT_DIR):
        return None
    try:
        old = {n: load_jsonl(os.path.join(LEGACY_OUT_DIR, f"{n}.jsonl")) for n in ("train", "val", "test")}
    except FileNotFoundError:
        return None
    qsets = {n: {normalize_text(r["q"]) for r in v} for n, v in old.items()}
    bsets = {n: {normalize_text(base_question(r["q"])) for r in v} for n, v in old.items()}
    obsets = {
        n: {normalize_text(base_question(r["q"])) for r in v if r.get("ood")} for n, v in old.items()
    }
    return {
        "path": LEGACY_OUT_DIR,
        "sizes": {n: len(v) for n, v in old.items()},
        "exact_question_overlap": {
            "train-val": len(qsets["train"] & qsets["val"]),
            "train-test": len(qsets["train"] & qsets["test"]),
            "val-test": len(qsets["val"] & qsets["test"]),
        },
        "base_question_overlap": {
            "train-val": len(bsets["train"] & bsets["val"]),
            "train-test": len(bsets["train"] & bsets["test"]),
            "val-test": len(bsets["val"] & bsets["test"]),
        },
        "test_rows_with_exact_q_in_train": sum(
            1 for r in old["test"] if normalize_text(r["q"]) in qsets["train"]
        ),
        "test_rows_with_base_q_in_train": sum(
            1 for r in old["test"] if normalize_text(base_question(r["q"])) in bsets["train"]
        ),
        "ood_base_question_overlap": {"train-test": len(obsets["train"] & obsets["test"])},
        "test_ood_total": sum(1 for r in old["test"] if r.get("ood")),
        "test_source_distribution": dict(Counter(r.get("source", "<none>") for r in old["test"])),
    }


def audit_gold_eval(unique, splits, previous_test_rows):
    """gold_eval 专项审计:纯度、与 train/val/旧 test 的多层去重、test 覆盖度。"""
    eval_rows = [r for r in unique if r.get("source") == "gold_eval"]
    eval_in_test = [r for r in splits["test"] if r.get("source") == "gold_eval"]
    non_eval_test = [r for r in splits["test"] if r.get("source") != "gold_eval"]
    by_intent = Counter(r["intent"] for r in eval_rows)
    test_intent_counts = Counter(r["intent"] for r in splits["test"])

    def qset(rows):
        return {normalize_text(r["q"]) for r in rows}

    def bset(rows):
        return {normalize_text(r["base_question"]) for r in rows}

    def fset(rows):
        return {r["family_id"] for r in rows}

    others = {
        "train": splits["train"],
        "val": splits["val"],
        "old_test": non_eval_test,
    }
    overlap = {}
    for name, other in others.items():
        overlap[name] = {
            "exact": len(qset(eval_rows) & qset(other)),
            "base": len(bset(eval_rows) & bset(other)),
            "family": len(fset(eval_rows) & fset(other)),
        }
    eval_ood = [r for r in eval_rows if r.get("ood")]
    overlap["ood_family"] = {
        name: len(fset(eval_ood) & fset([r for r in rows if r.get("ood")]))
        for name, rows in others.items()
    }

    # 逐题泄漏检查(用于 GOLD_EVAL_AUDIT.md)
    ref = {
        name: (qset(rows), bset(rows), fset(rows))
        for name, rows in others.items()
    }
    per_row = []
    for r in eval_rows:
        leak = {}
        for name, (qs, bs, fs) in ref.items():
            leak[f"{name}_exact"] = normalize_text(r["q"]) in qs
            leak[f"{name}_base"] = normalize_text(r["base_question"]) in bs
            leak[f"{name}_family"] = r["family_id"] in fs
        per_row.append(
            {
                "eval_id": r["eval_id"],
                "intent": r["intent"],
                "question": r["q"],
                "gold_answer": r["a"],
                "ood": bool(r["ood"]),
                "keywords": r["keywords"],
                "knowledge_source": r.get("knowledge_source", ""),
                "novelty_reason": r.get("novelty_reason", ""),
                "family_id": r["family_id"],
                "in_test": r.get("split") == "test",
                "leak": leak,
                "passed": (r.get("split") == "test") and not any(leak.values()),
            }
        )

    # 上一轮 clean test 是否被原样保留(previous_test_rows 必须在本轮写盘之前读取)
    preserved = None
    if previous_test_rows is not None:
        new_by_id = {r["sample_id"]: r for r in splits["test"]}
        missing = [r["sample_id"] for r in previous_test_rows if r["sample_id"] not in new_by_id]
        changed = [
            r["sample_id"]
            for r in previous_test_rows
            if r["sample_id"] in new_by_id
            and (new_by_id[r["sample_id"]]["q"], new_by_id[r["sample_id"]]["a"]) != (r["q"], r["a"])
        ]
        preserved = {
            "previous_test_rows": len(previous_test_rows),
            "missing": len(missing),
            "content_changed": len(changed),
        }

    all_intents = sorted({r["intent"] for r in unique})
    test_intents = sorted({r["intent"] for r in splits["test"]})
    missing_intents = sorted(set(all_intents) - set(test_intents))
    required = ("price", "recommend", "after_sale", "allergen")

    assertions = [
        {
            "name": "gold_eval samples all present and flagged eval_only",
            "passed": len(eval_rows) == len(GOLD_EVAL_SEEDS)
            and all(r.get("eval_only") for r in eval_rows),
            "detail": f"expected={len(GOLD_EVAL_SEEDS)} actual={len(eval_rows)}",
        },
        {
            "name": "gold_eval only in test (train/val gold_eval = 0)",
            "passed": len(eval_in_test) == len(eval_rows)
            and not any(r.get("source") == "gold_eval" for n in ("train", "val") for r in splits[n]),
            "detail": f"test={len(eval_in_test)} "
            f"train={sum(1 for r in splits['train'] if r.get('source') == 'gold_eval')} "
            f"val={sum(1 for r in splits['val'] if r.get('source') == 'gold_eval')}",
        },
        {
            "name": "gold_eval zero exact/base/family overlap with train/val/old_test",
            "passed": all(
                v == 0 for k, d in overlap.items() if k != "ood_family" for v in d.values()
            ),
            "detail": str({k: v for k, v in overlap.items() if k != "ood_family"}),
        },
        {
            "name": "gold_eval OOD families disjoint from train/val/old_test OOD families",
            "passed": all(v == 0 for v in overlap["ood_family"].values()),
            "detail": str(overlap["ood_family"]),
        },
        {
            "name": "test covers every intent present in the dataset",
            "passed": not missing_intents,
            "detail": f"test_intents={test_intents} missing={missing_intents}",
        },
        {
            "name": "test has >=10 questions for price/recommend/after_sale/allergen",
            "passed": all(test_intent_counts.get(i, 0) >= 10 for i in required),
            "detail": {i: test_intent_counts.get(i, 0) for i in required},
        },
    ]
    if preserved is not None:
        assertions.append(
            {
                "name": "previous clean test rows preserved verbatim",
                "passed": preserved["missing"] == 0 and preserved["content_changed"] == 0,
                "detail": str(preserved),
            }
        )

    return {
        "count": len(eval_rows),
        "expected_count": len(GOLD_EVAL_SEEDS),
        "by_intent": dict(sorted(by_intent.items())),
        "in_test": len(eval_in_test),
        "in_test_by_intent": dict(
            sorted(Counter(r["intent"] for r in eval_in_test).items())
        ),
        "overlap": overlap,
        "test_intent_counts": dict(sorted(test_intent_counts.items())),
        "test_missing_intents": missing_intents,
        "previous_test_preserved": preserved,
        "unsupported_scenarios": UNSUPPORTED_GOLD_SCENARIOS,
        "rows": per_row,
        "assertions": assertions,
    }


def write_gold_eval_audit(out_dir, ge):
    """生成 data_eng/output_clean/GOLD_EVAL_AUDIT.md(逐题人工可核查表)。"""
    lines = []
    lines.append("# 人工 Gold Evaluation Seed 审计表 (GOLD_EVAL_AUDIT)")
    lines.append("")
    lines.append(f"- gold_eval 总数: **{ge['count']}**(期望 {ge['expected_count']})")
    lines.append(f"- 全部进入 test: {ge['in_test'] == ge['count']}")
    lines.append(f"- 按 intent: {ge['by_intent']}")
    lines.append("")
    lines.append("## 0. 与 train / val / 旧 test 的多层去重结果(必须全为 0)")
    lines.append("")
    lines.append("```text")
    for name, d in ge["overlap"].items():
        if name == "ood_family":
            continue
        lines.append(f"gold_eval ∩ {name}: exact={d['exact']} base={d['base']} family={d['family']}")
    lines.append(f"gold_eval OOD family ∩ train/val/old_test: {ge['overlap']['ood_family']}")
    lines.append("```")
    lines.append("")
    lines.append("## 1. 最终 test 的 intent 覆盖")
    lines.append("")
    lines.append("| intent | test 题数 | 其中 gold_eval |")
    lines.append("|---|---|---|")
    for intent, cnt in ge["test_intent_counts"].items():
        lines.append(f"| {intent} | {cnt} | {ge['in_test_by_intent'].get(intent, 0)} |")
    lines.append("")
    lines.append(f"- test 未覆盖的 intent:{ge['test_missing_intents'] if ge['test_missing_intents'] else '无'}")
    lines.append(f"- 上一轮 clean test 保留情况:{ge['previous_test_preserved']}")
    lines.append(f"- 上一轮 test.jsonl 的 sha256:{ge.get('previous_test_sha256')}")
    lines.append("")
    lines.append("## 2. 逐题清单")
    lines.append("")
    for r in ge["rows"]:
        lines.append(f"### {r['eval_id']} | intent = `{r['intent']}`{' | OOD' if r['ood'] else ''}")
        lines.append("")
        lines.append(f"- **Question**:{r['question']}")
        lines.append(f"- **Gold answer**:{r['gold_answer']}")
        lines.append(f"- **Keywords**:{r['keywords']}")
        lines.append(f"- **Knowledge source**:{r['knowledge_source']}")
        lines.append(f"- **Novelty reason**:{r['novelty_reason']}")
        lines.append(f"- **family_id**:`{r['family_id']}`")
        lines.append(
            "- **Leakage check**:"
            + ("PASS" if r["passed"] else f"**FAIL** {r['leak']}")
            + "(exact / base / family 与 train、val、旧 test 均无交集)"
        )
        lines.append("")
    lines.append("## 3. 知识库无法支持、因此刻意没有造题的场景")
    lines.append("")
    lines.append("| intent | 场景 | 缺失的事实 |")
    lines.append("|---|---|---|")
    for s in ge["unsupported_scenarios"]:
        lines.append(f"| {s['intent']} | {s['scenario']} | {s['missing_fact']} |")
    lines.append("")
    lines.append("## 4. Assertions")
    lines.append("")
    lines.append("| assertion | result | detail |")
    lines.append("|---|---|---|")
    for a in ge["assertions"]:
        lines.append(f"| {a['name']} | {'PASS' if a['passed'] else '**FAIL**'} | {str(a['detail']).replace('|', '/')} |")
    lines.append("")
    path = os.path.join(out_dir, "GOLD_EVAL_AUDIT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description="构建带样本血缘的领域后训练数据(group 级切分)")
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="额外的数据文件(可多次指定,如 evolve/onpolicy 产物)",
    )
    parser.add_argument("--out-dir", type=str, default=OUT_DIR_DEFAULT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--reuse-split-from",
        type=str,
        default=OUT_DIR_DEFAULT,
        help="冻结该目录上一轮的 train/val/test 分配,只把 gold_eval 追加进 test;传空字符串则重新切分",
    )
    parser.add_argument(
        "--allow-overwrite-legacy",
        action="store_true",
        help="允许写回旧目录 data_eng/output/(默认禁止,保护审计用旧数据)",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    if out_dir == os.path.abspath(LEGACY_OUT_DIR) and not args.allow_overwrite_legacy:
        raise SystemExit(
            f"拒绝写回旧目录 {LEGACY_OUT_DIR}(它仍有审计价值,必须保留)。\n"
            f"请使用默认输出目录 {OUT_DIR_DEFAULT};确需覆盖请显式加 --allow-overwrite-legacy"
        )

    print("[1/6] 种子 + 模板扩展 + 前缀改写 ...")
    rows, gold_stats = expand_seeds()
    for r in rows:  # 先给 gold 样本补齐稳定 ID,供 extra 数据挂血缘时引用
        _apply_ids(r)
    print(f"      gold 样本 {len(rows)} 条 | 唯一问题 {len({normalize_text(r['q']) for r in rows})} 个")

    q_index, base_index = {}, {}
    for i, r in enumerate(rows):
        q_index.setdefault(normalize_text(r["q"]), i)
        base_index.setdefault(normalize_text(base_question(r["q"])), i)

    lineage_log = []
    extra_counts = {}
    invalid_total = 0
    for path in args.extra_file:
        print(f"[2/6] 合并额外数据并恢复血缘: {path}")
        added, invalid, log = attach_extra_rows(path, rows, q_index, base_index)
        rows.extend(added)  # q_index/base_index 只指向 gold 行,保证 extra 之间不会互相匹配
        extra_counts[path.replace("\\", "/")] = {"valid_rows": len(added), "invalid_dropped": invalid}
        invalid_total += invalid
        lineage_log.extend(log)
        print(f"      有效 {len(added)} 条 | 字段不完整丢弃 {invalid} 条")

    print("[2.5/6] 载入人工 Gold Evaluation Seed(只进 test)...")
    validate_gold_eval_seeds()
    gold_eval_rows = build_gold_eval_rows()
    rows.extend(gold_eval_rows)
    print(f"      gold_eval {len(gold_eval_rows)} 条(覆盖 intent:{dict(Counter(r['intent'] for r in gold_eval_rows))})")

    print("[3/6] 去重 + 长度过滤 ...")
    unique, dropped = dedup_and_filter(rows)
    for r in unique:
        _apply_ids(r)
    by_q = {}
    for r in unique:
        by_q.setdefault(normalize_text(r["q"]), r["sample_id"])
    for r in unique:
        r["parent_sample_id"] = by_q.get(normalize_text(r.get("parent_q") or r["q"]))
    print(f"      raw {len(rows)} -> {len(unique)} 条(丢弃 {dropped})")

    survived_eval = [r for r in unique if r.get("source") == "gold_eval"]
    if len(survived_eval) != len(gold_eval_rows):
        missing = {r["eval_id"] for r in gold_eval_rows} - {r["eval_id"] for r in survived_eval}
        raise SystemExit(
            f"[gold_eval 冲突] 以下评测题与已有数据 (q,a) 完全重复而被去重,必须重新设计:{sorted(missing)}"
        )

    print("[4/6] 构建血缘 group(family / 实体 / seed 连通分量)...")
    group_of, group_meta = build_groups(unique)
    clean_groups = sum(1 for m in group_meta.values() if m["clean"])
    print(f"      {len(group_meta)} 个 group(其中可进 val/test 的 clean group {clean_groups} 个)")

    frozen = None if args.reuse_split_from == "" else freeze_previous_split(args.reuse_split_from)
    # 注意:必须在写盘之前读取上一轮 test,否则读到的是本轮刚写出的文件
    previous_test_rows = None
    previous_test_sha256 = None
    if args.reuse_split_from:
        prev_test_path = os.path.join(args.reuse_split_from, "test.jsonl")
        if os.path.exists(prev_test_path):
            previous_test_rows = load_jsonl(prev_test_path)
            previous_test_sha256 = file_sha256(prev_test_path)
    if frozen:
        print(
            f"[5/6] 冻结上一轮 split({args.reuse_split_from} 的 {len(frozen)} 条样本),"
            "仅把 gold_eval 追加进 test ..."
        )
    else:
        print("[5/6] 全新 group 级切分(未找到可复用的上一轮 split)...")
    splits, assign = split_groups(unique, group_of, group_meta, seed=args.seed, frozen=frozen)
    for i, r in enumerate(unique):
        r["split_group_id"] = group_of[i]
        r["split"] = assign[group_of[i]]

    print("[6/6] 写盘 + 审计 ...")
    os.makedirs(out_dir, exist_ok=True)
    for name in ("train", "val", "test"):
        audit_lib.write_jsonl(os.path.join(out_dir, f"{name}.jsonl"), splits[name])

    here = os.path.dirname(os.path.abspath(__file__))
    inputs = {
        "seeds.py": f"sha256:{file_sha256(os.path.join(here, 'seeds.py'))[:16]}",
        "kb.py": f"sha256:{file_sha256(os.path.join(here, 'kb.py'))[:16]}",
        "generator": f"build_dataset.py sha256:{file_sha256(os.path.abspath(__file__))[:16]}",
        "extra_files": extra_counts,
        "split_seed": str(args.seed),
        "build_command": "python data_eng/build_dataset.py " + " ".join(sys.argv[1:]),
    }
    for path in args.extra_file:
        if os.path.exists(path):
            inputs[f"extra_file:{path.replace(chr(92), '/')}"] = (
                f"sha256:{file_sha256(path)[:16]}"
            )

    legacy = legacy_baseline()
    unknown_count = sum(1 for r in unique if r.get("lineage_confidence") == "unknown")
    ambiguous_count = sum(1 for r in unique if r.get("lineage_confidence") == "ambiguous_union")
    lineage_summary = {
        "gold_samples": len(rows) - sum(v["valid_rows"] for v in extra_counts.values()),
        "extra_samples": sum(v["valid_rows"] for v in extra_counts.values()),
        "invalid_extra_rows_dropped": invalid_total,
        "dedup_and_length_dropped": dropped,
        "unique_samples": len(unique),
        "groups_total": len(group_meta),
        "groups_clean": clean_groups,
        "groups_non_clean": len(group_meta) - clean_groups,
        "unknown_lineage_samples": unknown_count,
        "ambiguous_lineage_samples": ambiguous_count,
        "lineage_confidence": dict(Counter(r.get("lineage_confidence") for r in unique)),
        "gold_template_breakdown": dict(gold_stats),
    }

    missing_in_test = sorted(
        {r.get("intent") for r in unique} - {r.get("intent") for r in splits["test"]}
    )
    notes = [
        "旧目录 data_eng/output/ 与全部旧 checkpoint 未被修改;本脚本只写 --out-dir。",
        "val/test 只接受 gold 样本(source ∈ {seed, template, paraphrase})且血缘可追溯。",
        "evolved / onpolicy 只能进 train:它们会把所在血缘 group 标记为 non-clean,从而自动被排除出 val/test。",
        f"血缘不可追溯(unknown)样本 {unknown_count} 条,全部锁进 train(family_id=unknown_family),不计入任何 val/test 指标。",
        f"血缘歧义(ambiguous_union)样本 {ambiguous_count} 条:无法唯一确定父 seed,已把所有候选 seed 的 group 一起锁进 train。",
        f"test 未能覆盖的 intent(不做任何数据修补,仅如实报告):{missing_in_test}",
        "下游训练/评测脚本默认数据路径未改动;重训/评测时必须显式指定 --data-file data_eng/output_clean/*.jsonl。",
    ]
    notes.append(
        f"本轮新增人工 Gold Evaluation Seed {len(gold_eval_rows)} 条(source=gold_eval, eval_only=True):"
        "只允许进 test,不参与 paraphrase / 模板扩展 / 蒸馏 / 自采样,也不会出现在 train/val。"
    )
    notes.append(
        "gold_eval 与训练数据在 exact question / base question / family 三个层面完全隔离;"
        "但同一产品、同一主题下的**其他业务场景**仍可能与 train 共享实体"
        "(例如 train 有\"杨枝甘露什么价位\"、test 有\"杨枝甘露和茉莉绿茶哪个更贵\"),"
        "这是有意设计的\"新场景评测\",逐题理由见 GOLD_EVAL_AUDIT.md。"
    )
    if legacy:
        notes.append(
            "修复前对照(旧 data_eng/output/,样本级随机切分):test "
            f"{legacy['sizes']['test']} 条中,{legacy['test_rows_with_exact_q_in_train']} 条与 train 精确同题、"
            f"{legacy['test_rows_with_base_q_in_train']} 条去掉前缀后与 train 同源;"
            f"OOD base question train∩test = {legacy['ood_base_question_overlap']['train-test']}/"
            f"{legacy['test_ood_total']};test 含模型生成数据 {legacy['test_source_distribution']}。"
        )

    audit = audit_lib.audit_splits(
        splits,
        dataset_name=os.path.basename(out_dir),
        inputs=inputs,
        raw_rows=unique,
        lineage=lineage_summary,
        notes=notes,
    )
    if legacy:
        audit["legacy_baseline"] = legacy

    gold_eval_audit = audit_gold_eval(unique, splits, previous_test_rows)
    gold_eval_audit["previous_test_sha256"] = previous_test_sha256
    audit["gold_eval"] = gold_eval_audit
    audit["assertions"].extend(gold_eval_audit["assertions"])
    audit["all_passed"] = all(a["passed"] for a in audit["assertions"])

    json_path, md_path = audit_lib.write_reports(audit, out_dir)
    gold_md_path = write_gold_eval_audit(out_dir, gold_eval_audit)

    with open(os.path.join(out_dir, "lineage_report.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": lineage_summary, "extra_rows": lineage_log}, f, ensure_ascii=False, indent=2)

    group_rows = [
        {
            "group_id": gid,
            "size": m["size"],
            "clean": m["clean"],
            "split": assign[gid],
            "intent_main": m["intent_main"],
            "entity_main": m["entity_main"],
            "n_families": m["n_families"],
        }
        for gid, m in group_meta.items()
    ]
    group_rows.sort(key=lambda x: (-x["size"], x["group_id"]))
    with open(os.path.join(out_dir, "split_groups.json"), "w", encoding="utf-8") as f:
        json.dump(group_rows, f, ensure_ascii=False, indent=2)

    stats = {
        "dataset": os.path.basename(out_dir),
        "out_dir": out_dir.replace("\\", "/"),
        "split_seed": args.seed,
        "raw_count": len(rows),
        "dedup_dropped": dropped,
        "final_count": len(unique),
        "expanded_from_seeds": round(len(unique) / len(SEEDS), 2),
        "source_distribution": {n: audit["source_distribution"][n] for n in ("train", "val", "test")},
        "intent_distribution": {n: audit["intent_distribution"][n] for n in ("train", "val", "test")},
        "split": {n: len(splits[n]) for n in ("train", "val", "test")},
        "families": {n: audit["split"][n]["unique_families"] for n in ("train", "val", "test")},
        "groups": {n: audit["split"][n]["unique_groups"] for n in ("train", "val", "test")},
        "ood": {
            "samples": {n: audit["ood"]["counts"][n] for n in ("train", "val", "test")},
            "families": {n: audit["ood"]["families"][n] for n in ("train", "val", "test")},
        },
        "ood_ratio": round(sum(1 for r in unique if r.get("ood")) / len(unique), 4),
        "test_purity": audit["test_purity"],
        "gold_eval": {
            "count": gold_eval_audit["count"],
            "by_intent": gold_eval_audit["by_intent"],
            "test_gold_eval_count": gold_eval_audit["in_test"],
            "test_gold_eval_by_intent": gold_eval_audit["in_test_by_intent"],
            "overlap_with_train_val_old_test": gold_eval_audit["overlap"],
            "test_intent_counts": gold_eval_audit["test_intent_counts"],
            "previous_test_preserved": gold_eval_audit["previous_test_preserved"],
            "previous_test_sha256": gold_eval_audit["previous_test_sha256"],
            "unsupported_scenarios": gold_eval_audit["unsupported_scenarios"],
        },
        "unknown_lineage": audit["unknown_lineage"],
        "lineage_summary": lineage_summary,
        "missing_intents_per_split": audit["missing_intents_per_split"],
        "leakage": audit["leakage"],
        "assertions": audit["assertions"],
        "all_assertions_passed": audit["all_passed"],
        "legacy_baseline": legacy,
        "inputs": inputs,
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print(f"raw {len(rows)} -> unique {len(unique)}(dropped {dropped})")
    print(f"split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print("source distribution:")
    for name in ("train", "val", "test"):
        print(f"  {name}: {audit['source_distribution'][name]}")
    print("intent distribution:")
    for name in ("train", "val", "test"):
        print(f"  {name}: {audit['intent_distribution'][name]}")
    print(f"OOD: {audit['ood']['counts']} | OOD families: {audit['ood']['families']}")
    print(f"leakage: exact={audit['leakage']['exact_question_overlap']} "
          f"family={audit['leakage']['family_overlap']} "
          f"base={audit['leakage']['base_question_overlap']}")
    print(f"test purity: {audit['test_purity']}")
    print(f"gold_eval: {gold_eval_audit['count']} 条,进 test {gold_eval_audit['in_test']} 条 | "
          f"{gold_eval_audit['in_test_by_intent']}")
    print(f"test intent counts: {gold_eval_audit['test_intent_counts']}")
    print(f"gold_eval overlap (train/val/old_test): "
          f"{ {k: v for k, v in gold_eval_audit['overlap'].items() if k != 'ood_family'} }")
    print(f"unknown lineage: {audit['unknown_lineage']['unknown_family_count']} | "
          f"ambiguous: {audit['unknown_lineage']['ambiguous_lineage_count']}")
    print(f"missing intents: {audit['missing_intents_per_split']}")
    print(f"ALL ASSERTIONS PASSED: {audit['all_passed']}")
    for a in audit["assertions"]:
        if not a["passed"]:
            print(f"  FAIL: {a['name']} -> {a['detail']}")
    print(f"outputs: {out_dir}")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    print(f"  - {gold_md_path}")
    print("  - stats.json / lineage_report.json / split_groups.json")
    print("=" * 78)
    return 0 if audit["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
````

### `data_eng/evolve.py`

```python
"""Self-Instruct / Evol-Instruct 风格数据蒸馏扩写(智谱 GLM API)。

在模板扩展之外,用更强的 LLM 把种子问答"改写/进化"成风格多样的新数据:
  - Self-Instruct:让 GLM 对种子问答做口语化改写(不同问法,事实一致)
  - Evol-Instruct:可选 --evolve 模式,让 GLM 增加约束/复杂度生成更难的问题

流程:
  1. 从 .env 或环境变量读取 ZHIPU_API_KEY(密钥绝不入库)
  2. 挑选覆盖各意图的种子样本
  3. 对每个种子调用 API 生成多个新问答
  4. 质量过滤:新回答必须命中知识库关键实体/数字,否则丢弃
  5. 输出 evolved.jsonl,由 build_dataset.py --extra-file 合并进正式管道
"""

import argparse
import json
import os
import random
import re
import time

import httpx

from build_dataset import normalize
from kb import PRODUCTS
from leakage_audit import family_id_of
from seeds import SEEDS

# seed_id 与 build_dataset.py 一致(按 seeds.py 中的顺序编号),保证蒸馏数据的血缘可追溯
SEED_IDS = {id(s): f"seed_{i:03d}" for i, s in enumerate(SEEDS, 1)}

BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4-flash"  # 便宜快;可选 glm-4-air / glm-4-plus


def load_api_key():
    key = os.getenv("ZHIPU_API_KEY")
    if key:
        return key.strip()
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("ZHIPU_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        "未找到 ZHIPU_API_KEY:请在项目根目录创建 .env 文件(内容:ZHIPU_API_KEY=你的key),"
        "或设置环境变量"
    )


def chat(prompt, model, temperature=0.8, max_tokens=1024):
    key = load_api_key()
    resp = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_json_array(text):
    """从模型输出里解析 JSON 数组(容忍 markdown 代码块等包裹)。"""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def quality_pass(q, a, seed):
    """质量门:新回答必须包含种子答案的关键实体,或知识库产品名/数字。"""
    if not q or not a or len(q) < 4 or len(a) < 5:
        return False
    if any(k in a for k in seed["keywords"]):
        return True
    if any(p["name"] in a for p in PRODUCTS):
        return True
    return bool(re.search(r"\d", a))


PROMPT_TEMPLATE = (
    "你是数据标注专家。请基于下面的问答示例,生成 {n} 个'问法不同、但答案事实一致'的新问答对。\n"
    "要求:\n"
    "1. 用户问题要用不同的口语化表达(换称呼、换语气、换问法)\n"
    "2. 答案必须基于示例答案的事实,不能编造任何新事实、新价格、新政策\n"
    "3. 答案保持简洁(不超过 50 字)\n"
    "4. 只输出 JSON 数组,不要任何其他文字,格式:\n"
    '[{{"question": "问题", "answer": "答案"}}]\n\n'
    "示例问答:\n问题:{q}\n答案:{a}\n\n请生成 {n} 个新问答:"
)

EVOLVE_PROMPT_TEMPLATE = (
    "你是数据标注专家。请基于下面的问答示例,生成 {n} 个'更难、更复杂、场景更具体'的进化版问答。\n"
    "要求:\n"
    "1. 在原问题基础上增加约束/场景/细节(例如具体到某产品、某时间、某政策组合)\n"
    "2. 答案必须基于示例答案的事实,可以在示例事实范围内补充说明,但不能编造\n"
    "3. 答案保持简洁(不超过 60 字)\n"
    "4. 只输出 JSON 数组,不要任何其他文字,格式:\n"
    '[{{"question": "问题", "answer": "答案"}}]\n\n'
    "示例问答:\n问题:{q}\n答案:{a}\n\n请生成 {n} 个进化版问答:"
)


def main():
    parser = argparse.ArgumentParser(description="用智谱 GLM 蒸馏扩写领域数据")
    parser.add_argument("--seeds", type=int, default=30, help="挑多少条种子(控制 API 调用量)")
    parser.add_argument("--variants", type=int, default=3, help="每条种子生成几个变体")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--evolve", action="store_true", help="启用 Evol-Instruct 进化模式")
    parser.add_argument("--out", type=str, default="data_eng/output/evolved.jsonl")
    args = parser.parse_args()

    load_api_key()  # 提前校验 key 存在
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rng = random.Random(42)
    by_intent = {}
    for s in SEEDS:
        by_intent.setdefault(s["intent"], []).append(s)
    # 尽量覆盖所有意图
    sampled = []
    for group in by_intent.values():
        sampled.extend(rng.sample(group, min(len(group), max(1, args.seeds // len(by_intent)))))
    sampled = sampled[: args.seeds]
    print(f"选中 {len(sampled)} 条种子,意图覆盖 {len(by_intent)} 类")

    template = EVOLVE_PROMPT_TEMPLATE if args.evolve else PROMPT_TEMPLATE
    rows, failed = [], 0
    for i, seed in enumerate(sampled, 1):
        prompt = template.format(q=seed["q"], a=seed["a"], n=args.variants)
        try:
            text = chat(prompt, args.model)
            data = extract_json_array(text)
        except Exception as e:
            print(f"  [{i}/{len(sampled)}] API 调用失败:{e}")
            failed += 1
            continue
        if not data:
            failed += 1
            continue
        for item in data:
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if quality_pass(q, a, seed):
                seed_id = SEED_IDS.get(id(seed))
                rows.append(
                    {
                        "intent": seed["intent"],
                        "q": q,
                        "a": a,
                        "keywords": seed["keywords"],
                        "ood": seed["ood"],
                        "source": "evolved",
                        # ---- 样本血缘:让蒸馏数据能追溯到父 seed ----
                        "seed_id": seed_id,
                        "family_id": family_id_of(seed["intent"], seed["q"]),
                        "base_question": seed["q"],
                        "parent_q": seed["q"],
                        "variant_type": "evolved",
                        "lineage_confidence": "self_reported",
                    }
                )
        time.sleep(0.5)
        if i % 5 == 0:
            print(f"  已处理 {i}/{len(sampled)},累计合格 {len(rows)} 条")

    # 去重
    seen = set()
    unique = []
    for r in rows:
        key = normalize(r["q"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成:API 失败 {failed} 次,生成合格 {len(unique)} 条 -> {args.out}")
    print("下一步:python data_eng/build_dataset.py --extra-file " + args.out)


if __name__ == "__main__":
    main()
```

### `data_eng/gold_eval_seeds.py`

```python
"""人工 Gold Evaluation Seeds(仅用于评测,绝不进入训练)。

硬性约束(与本轮任务要求一致):
  1. 全部由人工依据 data_eng/kb.py(以及已在 seeds.py 中确认的事实)编写;
     **没有调用任何 LLM / API**(GLM、Qwen、Self-Instruct、evolve.py、RFT 均未运行)。
  2. 每条都带 knowledge_source(事实出处)与 novelty_reason(为什么它是一个新的业务场景,
     而不是已有问题的同义改写)。
  3. eval_only = True → build_dataset.py 只允许它们进 test,永不进 train/val,
     也永远不会被 prefix paraphrase / 模板扩展 / evolved / onpolicy 复制。
  4. family_id 由 (intent, base_question) 派生,与训练数据的 family 互斥
     (由 leakage audit 逐条验证)。
  5. 知识库无法可靠回答的场景不在这里造题,统一记录在 GOLD_EVAL_AUDIT.md 的
     "无法构造的题目 / 缺失事实" 一节。

字段说明:
  id               评测样本唯一 id(同时作为 seed_id)
  q / a            问题 / 人工金标准答案
  intent           10 类业务意图之一
  keywords         评测用关键事实(命中即视为答对,与 sft/eval_domain.py 的定义一致)
  ood              是否知识库外(应为拒答)
  knowledge_source 答案的事实出处(kb.py 字段 / seeds.py 中已确认的事实)
  novelty_reason   为什么这条是新场景,而不是已有问题的改写
"""

GOLD_EVAL_SEEDS = [
    # ================= price(12)=================
    # 说明:已有的 price 覆盖"单款问价"和"加料问价";下面 12 条覆盖比价、区间、
    #      组合计价、优惠后实付等**新的信息需求**。
    {
        "id": "eval_price_001",
        "q": "杨枝甘露和茉莉绿茶,哪个更贵?",
        "a": "杨枝甘露 18 元一杯,比茉莉绿茶(8 元一杯)贵 10 元。",
        "intent": "price",
        "keywords": ["18", "8"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(杨枝甘露 18 / 茉莉绿茶 8)",
        "novelty_reason": "跨产品比价,已有 price 只有单款问价与\"最便宜的是什么\"。",
    },
    {
        "id": "eval_price_002",
        "q": "菜单上最贵的是哪一款?",
        "a": "最贵的是杨枝甘露,18 元一杯。",
        "intent": "price",
        "keywords": ["杨枝甘露", "18"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(8 款价格 8~18 元,最高为杨枝甘露 18)",
        "novelty_reason": "全场最高价查询,与已有的\"最便宜的是什么\"方向相反。",
    },
    {
        "id": "eval_price_003",
        "q": "10 元以内能买到哪几款饮品?",
        "a": "10 元以内的有茉莉绿茶 8 元和柠檬气泡水 10 元。",
        "intent": "price",
        "keywords": ["茉莉绿茶", "柠檬气泡水"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(茉莉绿茶 8、柠檬气泡水 10)",
        "novelty_reason": "价格区间筛选,不是单款问价。",
    },
    {
        "id": "eval_price_004",
        "q": "珍珠和椰果的加价是一样的吗?",
        "a": "一样,珍珠和椰果都是加 2 元。",
        "intent": "price",
        "keywords": ["2"],
        "ood": False,
        "knowledge_source": "kb.py TOPPINGS(珍珠 2、椰果 2)",
        "novelty_reason": "两种小料之间的比价,已有的是单种小料问价。",
    },
    {
        "id": "eval_price_005",
        "q": "五种小料每种各加一份,一共要多花多少钱?",
        "a": "珍珠 2 元、椰果 2 元、芋泥 3 元、奶盖 4 元、布丁 3 元,一共 14 元。",
        "intent": "price",
        "keywords": ["14"],
        "ood": False,
        "knowledge_source": "kb.py TOPPINGS(2+2+3+4+3=14)",
        "novelty_reason": "全部小料的合计加价,是新场景(不是单种小料价格)。",
    },
    {
        "id": "eval_price_006",
        "q": "经典珍珠奶茶加椰果和芋泥,一共多少钱?",
        "a": "经典珍珠奶茶 12 元,加椰果 2 元、芋泥 3 元,一共 17 元。",
        "intent": "price",
        "keywords": ["17"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(经典珍珠奶茶 12)+ TOPPINGS(椰果 2、芋泥 3)",
        "novelty_reason": "单品 + 多种小料的组合计价,KB 支持但此前没有这种问法。",
    },
    {
        "id": "eval_price_007",
        "q": "来两杯茉莉绿茶,一共多少钱?",
        "a": "茉莉绿茶 8 元一杯,两杯一共 16 元。",
        "intent": "price",
        "keywords": ["16"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(茉莉绿茶 8,8×2=16)",
        "novelty_reason": "多杯总价,已有问法都是单杯价格。",
    },
    {
        "id": "eval_price_008",
        "q": "周三买两杯经典珍珠奶茶,一共要付多少钱?",
        "a": "经典珍珠奶茶 12 元一杯,周三第二杯半价,两杯一共 18 元。",
        "intent": "price",
        "keywords": ["18"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(12)+ COUPON_WED(第二杯半价),12+6=18",
        "novelty_reason": "把会员日优惠换算成实付金额,是 price×member 的新场景。",
    },
    {
        "id": "eval_price_009",
        "q": "新客第一次买一杯杨枝甘露,实际要付多少钱?",
        "a": "杨枝甘露 18 元一杯,新客注册立减 3 元,实际 15 元。",
        "intent": "price",
        "keywords": ["15"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(18)+ COUPON_NEW(立减 3),18-3=15",
        "novelty_reason": "新客优惠后的实际支付价,已有的是优惠本身而非算价。",
    },
    {
        "id": "eval_price_010",
        "q": "我有 100 积分,买一杯茉莉绿茶能抵多少?",
        "a": "100 积分可抵 5 元,茉莉绿茶 8 元一杯,抵扣后是 3 元。",
        "intent": "price",
        "keywords": ["5", "3"],
        "ood": False,
        "knowledge_source": "kb.py MEMBER_RULE(100 积分抵 5 元)+ PRODUCTS(茉莉绿茶 8)",
        "novelty_reason": "积分抵扣后的实付价,不重复已有的\"积分有什么用\"。",
    },
    {
        "id": "eval_price_011",
        "q": "想凑到免配送费,最少要买多少钱?",
        "a": "满 25 元免配送费,所以最少要买到 25 元。",
        "intent": "price",
        "keywords": ["25"],
        "ood": False,
        "knowledge_source": "kb.py DELIVERY_FREE_OVER=25",
        "novelty_reason": "凑单门槛金额,是 price×delivery 的新场景(不是问配送费多少)。",
    },
    {
        "id": "eval_price_012",
        "q": "一杯芝士奶盖四季春加上 3 元配送费,一共多少钱?",
        "a": "芝士奶盖四季春 16 元一杯,加 3 元配送费一共 19 元。",
        "intent": "price",
        "keywords": ["19"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(16)+ DELIVERY_FEE(3),16+3=19",
        "novelty_reason": "商品加配送费的总额计算,已有价格题都不含配送费。",
    },
    # ================= recommend(12)=================
    # 说明:已有 recommend 已被 16 个 tag × 6 个模板覆盖,下面 12 条全部走
    #      "预算 / 成分 / 过敏 / 组合 / 小料替代"等新维度,不是换词。
    {
        "id": "eval_recommend_001",
        "q": "想喝清淡一点的,推荐哪款?",
        "a": "推荐茉莉绿茶,8 元一杯,标签就是清爽清淡。",
        "intent": "recommend",
        "keywords": ["茉莉绿茶"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS 标签(茉莉绿茶 tags 含\"清淡\")",
        "novelty_reason": "\"清淡\"不在已有模板使用的 16 个 tag 里,是新的筛选维度。",
    },
    {
        "id": "eval_recommend_002",
        "q": "有带西米的饮品吗?",
        "a": "杨枝甘露里含西米,18 元一杯,喜欢颗粒口感可以试试。",
        "intent": "recommend",
        "keywords": ["杨枝甘露"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS 标签(杨枝甘露 tags 含\"西米\")",
        "novelty_reason": "按配料\"西米\"推荐,不在已有模板的 tag 集合中。",
    },
    {
        "id": "eval_recommend_003",
        "q": "第一次来不知道怎么点,帮我搭一个两杯的入门组合?",
        "a": "可以点一杯杨枝甘露 18 元配一杯经典珍珠奶茶 12 元,一杯果味一杯奶茶,一共 30 元。",
        "intent": "recommend",
        "keywords": ["杨枝甘露", "经典珍珠奶茶"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(两款均为 8 款在售产品之一);seeds.py seed_015 已确认这两款是招牌",
        "novelty_reason": "首次购买的组合点单,已有都是单品推荐。",
    },
    {
        "id": "eval_recommend_004",
        "q": "我乳糖不耐受,推荐一款不含牛奶的饮品?",
        "a": "过敏原提示里不含牛奶、乳制品的饮品有茉莉绿茶 8 元和柠檬气泡水 10 元,这两款更稳妥。",
        "intent": "recommend",
        "keywords": ["茉莉绿茶", "柠檬气泡水"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(仅黑糖珍珠鲜奶/芋泥波波鲜奶含牛奶、芝士奶盖四季春含乳制品)",
        "novelty_reason": "过敏约束下的推荐,区别于已有的\"哪些饮品含牛奶\"(纯查询)。",
    },
    {
        "id": "eval_recommend_005",
        "q": "两个人喝,想一杯果味一杯奶茶,怎么点比较合适?",
        "a": "可以点杨枝甘露 18 元(芒果、西柚风味)配经典珍珠奶茶 12 元,一共 30 元。",
        "intent": "recommend",
        "keywords": ["杨枝甘露", "经典珍珠奶茶"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS 标签(杨枝甘露含芒果/西柚;经典珍珠奶茶含奶茶/珍珠)",
        "novelty_reason": "双人双口味搭配,不是单款推荐。",
    },
    {
        "id": "eval_recommend_006",
        "q": "我想一次喝到两种水果风味,推荐哪款?",
        "a": "推荐杨枝甘露,含芒果和西柚两种水果风味,18 元一杯。",
        "intent": "recommend",
        "keywords": ["杨枝甘露"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS 标签(杨枝甘露 tags: 芒果、西柚)",
        "novelty_reason": "按\"双水果风味\"这一组合条件推荐,不同于单一 tag(芒果/西柚)的已有问法。",
    },
    {
        "id": "eval_recommend_007",
        "q": "想给同事带 4 杯不重样的饮品,有什么搭配建议?",
        "a": "可以点经典珍珠奶茶 12 元、焦糖布丁奶茶 14 元、茉莉绿茶 8 元、杨枝甘露 18 元,口味覆盖奶茶、茶和果味,一共 52 元。",
        "intent": "recommend",
        "keywords": ["52"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(12+14+8+18=52)",
        "novelty_reason": "多人多杯不重样搭配,是新的点单场景。",
    },
    {
        "id": "eval_recommend_008",
        "q": "你们家有哪些茶类饮品?",
        "a": "茶类的有茉莉绿茶 8 元(标签含绿茶)和芝士奶盖四季春 16 元(标签含茶)。",
        "intent": "recommend",
        "keywords": ["茉莉绿茶", "芝士奶盖四季春"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS 标签(茉莉绿茶含\"绿茶\"、芝士奶盖四季春含\"茶\")",
        "novelty_reason": "\"茶类\"是跨产品的归纳,已有模板 tag 只有绿茶/茉莉/芝士/奶盖等具体词。",
    },
    {
        "id": "eval_recommend_009",
        "q": "想要芋泥但不想喝鲜奶类,有别的办法吗?",
        "a": "可以加芋泥小料,加 3 元,比如加在茉莉绿茶上就是一杯带芋泥的茶饮。",
        "intent": "recommend",
        "keywords": ["芋泥", "3"],
        "ood": False,
        "knowledge_source": "kb.py TOPPINGS(芋泥 3 元)+ PRODUCTS(茉莉绿茶 8)",
        "novelty_reason": "用\"小料\"绕开产品限制的推荐思路,KB 支持但此前完全没有。",
    },
    {
        "id": "eval_recommend_010",
        "q": "一杯奶茶配一杯清爽的茶,怎么组合?",
        "a": "可以经典珍珠奶茶 12 元配茉莉绿茶 8 元,一浓一淡,一共 20 元。",
        "intent": "recommend",
        "keywords": ["经典珍珠奶茶", "茉莉绿茶"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(12 与 8,合计 20)",
        "novelty_reason": "浓淡搭配的组合推荐,区别于单款推荐和双人场景。",
    },
    {
        "id": "eval_recommend_011",
        "q": "想加点椰果,推荐怎么搭配?",
        "a": "椰果加 2 元,可以加在经典珍珠奶茶(12 元)上,一共 14 元。",
        "intent": "recommend",
        "keywords": ["椰果"],
        "ood": False,
        "knowledge_source": "kb.py TOPPINGS(椰果 2)+ PRODUCTS(经典珍珠奶茶 12)",
        "novelty_reason": "小料搭配推荐,已有的是\"加椰果多少钱\"(纯问价)。",
    },
    {
        "id": "eval_recommend_012",
        "q": "四个人一人一杯,预算 60 元够吗?",
        "a": "够。最便宜的四款是茉莉绿茶 8 元、柠檬气泡水 10 元、经典珍珠奶茶 12 元、焦糖布丁奶茶 14 元,一共 44 元,在 60 元预算内。",
        "intent": "recommend",
        "keywords": ["44"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(8+10+12+14=44,全部 ≤60)",
        "novelty_reason": "按人数与预算做可行性判断,是新的决策场景。",
    },
    # ================= after_sale(10)=================
    {
        "id": "eval_after_sale_001",
        "q": "饮品做错口味了,可以怎么处理?",
        "a": "出品 30 分钟内口味不满意,可以免费重做或退款。",
        "intent": "after_sale",
        "keywords": ["30", "重做", "退款"],
        "ood": False,
        "knowledge_source": "kb.py REFUND_RULE",
        "novelty_reason": "做错单场景,已有是\"不好喝/味道不对\"这类主观口味问题。",
    },
    {
        "id": "eval_after_sale_002",
        "q": "冰量做错了,能重新做一杯吗?",
        "a": "可以的,出品 30 分钟内口味不满意可以免费重做或退款,冰量做错可以直接申请重做。",
        "intent": "after_sale",
        "keywords": ["30", "重做"],
        "ood": False,
        "knowledge_source": "kb.py REFUND_RULE",
        "novelty_reason": "规格(冰量)出错,不是口味评价类问题。",
    },
    {
        "id": "eval_after_sale_003",
        "q": "我想申请退款,需要满足什么条件?",
        "a": "两种情形:饮品出品 30 分钟内口味不满意可免费重做或退款;外卖洒漏或破损,拍照联系客服可退款。",
        "intent": "after_sale",
        "keywords": ["30", "退款", "拍照"],
        "ood": False,
        "knowledge_source": "kb.py REFUND_RULE + DELIVERY_REFUND",
        "novelty_reason": "退款条件汇总,已有的是单纯问\"能退款吗\"。",
    },
    {
        "id": "eval_after_sale_004",
        "q": "外卖杯盖没盖好洒出来了,怎么退?",
        "a": "外卖洒漏或破损,拍照联系客服可退款。",
        "intent": "after_sale",
        "keywords": ["拍照", "退款"],
        "ood": False,
        "knowledge_source": "kb.py DELIVERY_REFUND",
        "novelty_reason": "具体破损原因(杯盖)下的退款规定,已有只问\"外卖洒了怎么办\"。",
    },
    {
        "id": "eval_after_sale_005",
        "q": "超过 30 分钟才觉得不好喝,还能退吗?",
        "a": "免费重做或退款对应的是\"出品 30 分钟内口味不满意\",超过 30 分钟不在这个规则范围内。",
        "intent": "after_sale",
        "keywords": ["30"],
        "ood": False,
        "knowledge_source": "kb.py REFUND_RULE(30 分钟时限边界)",
        "novelty_reason": "政策时限的边界询问,已有问题都默认在时限内。",
    },
    {
        "id": "eval_after_sale_006",
        "q": "我是自取的,也能免费重做吗?",
        "a": "可以,规则是饮品出品 30 分钟内口味不满意可免费重做或退款。",
        "intent": "after_sale",
        "keywords": ["30", "重做"],
        "ood": False,
        "knowledge_source": "kb.py REFUND_RULE(以\"出品\"为口径,不区分自取/堂食)",
        "novelty_reason": "购买渠道(自取)场景,已有问题都没提渠道。",
    },
    {
        "id": "eval_after_sale_007",
        "q": "我可以既要重做又要退款吗?",
        "a": "规则是二选一:出品 30 分钟内口味不满意,可以免费重做或退款。",
        "intent": "after_sale",
        "keywords": ["重做", "退款"],
        "ood": False,
        "knowledge_source": "kb.py REFUND_RULE(\"免费重做或退款\"为二选一)",
        "novelty_reason": "权益边界(能否同时主张),是新的政策解读场景。",
    },
    {
        "id": "eval_after_sale_008",
        "q": "外卖洒了,是找骑手还是找客服?",
        "a": "外卖洒漏或破损,拍照联系客服可退款,走客服渠道处理。",
        "intent": "after_sale",
        "keywords": ["拍照", "客服"],
        "ood": False,
        "knowledge_source": "kb.py DELIVERY_REFUND(联系客服)",
        "novelty_reason": "责任/渠道询问,已有问题只问结果不问找谁。",
    },
    {
        "id": "eval_after_sale_009",
        "q": "能退款的情况有哪些?",
        "a": "两种:饮品出品 30 分钟内口味不满意可免费重做或退款;外卖洒漏或破损,拍照联系客服可退款。",
        "intent": "after_sale",
        "keywords": ["30", "退款"],
        "ood": False,
        "knowledge_source": "kb.py REFUND_RULE + DELIVERY_REFUND",
        "novelty_reason": "把两条售后政策合并枚举,已有问题都是单一场景。",
    },
    {
        "id": "eval_after_sale_010",
        "q": "重做或退款的申请要在多久之内提出?",
        "a": "要在出品 30 分钟内提出,30 分钟内口味不满意可以免费重做或退款。",
        "intent": "after_sale",
        "keywords": ["30"],
        "ood": False,
        "knowledge_source": "kb.py REFUND_RULE",
        "novelty_reason": "直接问时限本身,与已有\"不好喝能换吗\"这类结果导向问题不同。",
    },
    # ================= allergen(10)=================
    {
        "id": "eval_allergen_001",
        "q": "芝士奶盖四季春含乳制品吗?",
        "a": "含的,芝士奶盖四季春的过敏原提示是含乳制品。",
        "intent": "allergen",
        "keywords": ["乳制品"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(芝士奶盖四季春: 含乳制品)",
        "novelty_reason": "单款产品的乳制品确认,已有是\"哪些饮品是奶制品\"这类全表枚举。",
    },
    {
        "id": "eval_allergen_002",
        "q": "黑糖珍珠鲜奶的过敏原是什么?",
        "a": "黑糖珍珠鲜奶含牛奶。",
        "intent": "allergen",
        "keywords": ["牛奶"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(黑糖珍珠鲜奶: 含牛奶)",
        "novelty_reason": "按产品查过敏原条目,已有只查过杨枝甘露和\"哪款不含奶\"。",
    },
    {
        "id": "eval_allergen_003",
        "q": "芋泥波波鲜奶里含哪两种过敏原?",
        "a": "芋泥波波鲜奶含牛奶和芋泥。",
        "intent": "allergen",
        "keywords": ["牛奶", "芋泥"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(芋泥波波鲜奶: 含牛奶和芋泥)",
        "novelty_reason": "双过敏原产品查询,是新的产品×过敏原组合。",
    },
    {
        "id": "eval_allergen_004",
        "q": "我对芒果过敏,能喝杨枝甘露吗?",
        "a": "不建议,杨枝甘露含芒果,对芒果过敏的话不建议饮用。",
        "intent": "allergen",
        "keywords": ["芒果"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(杨枝甘露: 含芒果)",
        "novelty_reason": "过敏人群的决策询问,已有是\"杨枝甘露里有芒果吗\"这种事实查询。",
    },
    {
        "id": "eval_allergen_005",
        "q": "我乳糖不耐受,能喝芝士奶盖四季春吗?",
        "a": "不建议,芝士奶盖四季春含乳制品,乳糖不耐受的话要避开。",
        "intent": "allergen",
        "keywords": ["乳制品"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(芝士奶盖四季春: 含乳制品)",
        "novelty_reason": "特定人群(乳糖不耐受)+ 单款的判断,是新的组合场景。",
    },
    {
        "id": "eval_allergen_006",
        "q": "同时对牛奶和芒果过敏,能喝哪几款?",
        "a": "要同时避开含牛奶的黑糖珍珠鲜奶、芋泥波波鲜奶和含乳制品的芝士奶盖四季春,以及含芒果的杨枝甘露;剩下的经典珍珠奶茶、焦糖布丁奶茶、茉莉绿茶、柠檬气泡水不在这些过敏原提示里。",
        "intent": "allergen",
        "keywords": ["经典珍珠奶茶", "茉莉绿茶"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(4 条产品过敏原提示)",
        "novelty_reason": "双过敏原求交集,需要综合全部 4 条过敏原记录,信息需求最复杂。",
    },
    {
        "id": "eval_allergen_007",
        "q": "哪款饮品同时含牛奶和芋泥?",
        "a": "芋泥波波鲜奶,它的过敏原提示是含牛奶和芋泥。",
        "intent": "allergen",
        "keywords": ["芋泥波波鲜奶"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(芋泥波波鲜奶: 含牛奶和芋泥)",
        "novelty_reason": "反向检索(按过敏原组合找产品),与\"查某产品含什么\"方向相反。",
    },
    {
        "id": "eval_allergen_008",
        "q": "奶盖类的饮品含乳制品吗?",
        "a": "含的,芝士奶盖四季春含乳制品,奶盖类要注意乳制品过敏。",
        "intent": "allergen",
        "keywords": ["乳制品"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(芝士奶盖四季春: 含乳制品)",
        "novelty_reason": "按产品标签类别(奶盖)推断过敏原,是新的归纳式问法。",
    },
    {
        "id": "eval_allergen_009",
        "q": "我牛奶过敏,黑糖珍珠鲜奶和芋泥波波鲜奶都不能喝吗?",
        "a": "是的,黑糖珍珠鲜奶和芋泥波波鲜奶都含牛奶,牛奶过敏的话两款都要避开。",
        "intent": "allergen",
        "keywords": ["黑糖珍珠鲜奶", "芋泥波波鲜奶"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(两款均含牛奶)",
        "novelty_reason": "多产品同时确认,已有只问过\"有含牛奶的饮品吗\"这类存在性问题。",
    },
    {
        "id": "eval_allergen_010",
        "q": "对乳制品过敏的人,菜单上要避开哪一款?",
        "a": "要避开芝士奶盖四季春,它的过敏原提示是含乳制品。",
        "intent": "allergen",
        "keywords": ["芝士奶盖四季春"],
        "ood": False,
        "knowledge_source": "kb.py ALLERGENS(芝士奶盖四季春: 含乳制品)",
        "novelty_reason": "从过敏人群出发做单品定位,与已有\"哪些是奶制品\"的枚举方向不同。",
    },
    # ================= hours(8;补足 test 到 13)=================
    {
        "id": "eval_hours_001",
        "q": "你们一天营业多少个小时?",
        "a": "营业时间是 10:00-22:00,一天共 12 个小时。",
        "intent": "hours",
        "keywords": ["12"],
        "ood": False,
        "knowledge_source": "kb.py OPEN_TIME/CLOSE_TIME(10:00-22:00 = 12 小时)",
        "novelty_reason": "营业时长换算,已有只问开门/关门时间点。",
    },
    {
        "id": "eval_hours_002",
        "q": "节假日你们营业吗?",
        "a": "营业,节假日正常营业,营业时间 10:00-22:00。",
        "intent": "hours",
        "keywords": ["10:00", "22:00"],
        "ood": False,
        "knowledge_source": "kb.py OPEN_TIME/CLOSE_TIME;seeds.py seed_028 确认\"周末和节假日正常营业\"",
        "novelty_reason": "节假日场景,已有只问过周末。",
    },
    {
        "id": "eval_hours_003",
        "q": "早上九点能买到吗?",
        "a": "买不到,门店 10:00 才开门营业。",
        "intent": "hours",
        "keywords": ["10:00"],
        "ood": False,
        "knowledge_source": "kb.py OPEN_TIME=10:00",
        "novelty_reason": "以\"某个具体时间点能不能买到\"提问,是新的判定场景。",
    },
    {
        "id": "eval_hours_004",
        "q": "晚上十点半还能到店买吗?",
        "a": "不能,门店 22:00 关门,十点半已经打烊了。",
        "intent": "hours",
        "keywords": ["22:00"],
        "ood": False,
        "knowledge_source": "kb.py CLOSE_TIME=22:00",
        "novelty_reason": "打烊后的边界时间判断,与已有\"营业到几点\"不同。",
    },
    {
        "id": "eval_hours_005",
        "q": "每天的营业时间都一样吗?",
        "a": "一样,都是 10:00-22:00。",
        "intent": "hours",
        "keywords": ["10:00", "22:00"],
        "ood": False,
        "knowledge_source": "kb.py OPEN_TIME/CLOSE_TIME;seeds.py seed_028 确认周末/节假日同时间",
        "novelty_reason": "作息一致性确认,是新的政策澄清类问题。",
    },
    {
        "id": "eval_hours_006",
        "q": "外卖的结束时间和门店关门时间一样吗?",
        "a": "不一样,门店 22:00 关门,外卖 21:30 截止下单。",
        "intent": "hours",
        "keywords": ["22:00", "21:30"],
        "ood": False,
        "knowledge_source": "kb.py CLOSE_TIME=22:00 与 DELIVERY_CUTOFF=21:30",
        "novelty_reason": "两个时间的对比,需要同时检索门店与外送政策。",
    },
    {
        "id": "eval_hours_007",
        "q": "中午去店里,你们在营业吗?",
        "a": "在的,营业时间是 10:00-22:00,中午也在营业时间内。",
        "intent": "hours",
        "keywords": ["10:00", "22:00"],
        "ood": False,
        "knowledge_source": "kb.py OPEN_TIME/CLOSE_TIME",
        "novelty_reason": "特定时段(中午)是否营业,已有问的是\"现在这个点\"。",
    },
    {
        "id": "eval_hours_008",
        "q": "最晚几点还能进店点单?",
        "a": "门店营业到 22:00,最晚在 22:00 关门前到店点单。",
        "intent": "hours",
        "keywords": ["22:00"],
        "ood": False,
        "knowledge_source": "kb.py CLOSE_TIME=22:00",
        "novelty_reason": "以\"最晚点单时间\"提问,是新的可用性判断角度。",
    },
    # ================= delivery(6;补足 test 到 11)=================
    {
        "id": "eval_delivery_001",
        "q": "3.5 公里能送吗?",
        "a": "不能,门店只送 3 公里以内。",
        "intent": "delivery",
        "keywords": ["3"],
        "ood": False,
        "knowledge_source": "kb.py DELIVERY_RADIUS=3 公里",
        "novelty_reason": "具体距离的可行性判断,已有只问\"配送范围多远\"。",
    },
    {
        "id": "eval_delivery_002",
        "q": "买满多少钱可以免配送费?",
        "a": "满 25 元免配送费。",
        "intent": "delivery",
        "keywords": ["25"],
        "ood": False,
        "knowledge_source": "kb.py DELIVERY_FREE_OVER=25",
        "novelty_reason": "免配送门槛,已有问的是\"配送费多少钱\"。",
    },
    {
        "id": "eval_delivery_003",
        "q": "只点一杯茉莉绿茶,要付配送费吗?",
        "a": "要付,茉莉绿茶 8 元不满 25 元,配送费 3 元。",
        "intent": "delivery",
        "keywords": ["3", "25"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(茉莉绿茶 8)与 DELIVERY_FEE=3/DELIVERY_FREE_OVER=25",
        "novelty_reason": "单品金额与免配送门槛的对照判断,是新的组合计算。",
    },
    {
        "id": "eval_delivery_004",
        "q": "点两杯杨枝甘露够免配送费吗?",
        "a": "够,两杯杨枝甘露 36 元,超过 25 元可以免配送费。",
        "intent": "delivery",
        "keywords": ["36", "25"],
        "ood": False,
        "knowledge_source": "kb.py PRODUCTS(杨枝甘露 18×2=36)与 DELIVERY_FREE_OVER=25",
        "novelty_reason": "用具体订单金额判断能否免配送费。",
    },
    {
        "id": "eval_delivery_005",
        "q": "晚上九点四十还能点外卖吗?",
        "a": "不能,外卖 21:30 截止下单,九点四十已经过了截止时间。",
        "intent": "delivery",
        "keywords": ["21:30"],
        "ood": False,
        "knowledge_source": "kb.py DELIVERY_CUTOFF=21:30",
        "novelty_reason": "具体时间点的下单可行性判断,已有只问截止时间。",
    },
    {
        "id": "eval_delivery_006",
        "q": "配送范围是以门店为中心算的吗?",
        "a": "是的,门店 3 公里内配送,以门店为中心。",
        "intent": "delivery",
        "keywords": ["3"],
        "ood": False,
        "knowledge_source": "kb.py DELIVERY_RADIUS=3 公里",
        "novelty_reason": "配送范围的计量口径(以门店为中心),是新的政策澄清问题。",
    },
    # ================= member(5;补足 test 到 15)=================
    {
        "id": "eval_member_001",
        "q": "消费 30 元能积多少分?",
        "a": "消费 1 元积 1 分,消费 30 元可以积 30 分。",
        "intent": "member",
        "keywords": ["30"],
        "ood": False,
        "knowledge_source": "kb.py MEMBER_RULE(1 元 1 分)",
        "novelty_reason": "具体消费金额换算积分,已有只问\"怎么积分\"。",
    },
    {
        "id": "eval_member_002",
        "q": "500 积分能抵多少钱?",
        "a": "100 积分抵 5 元,500 积分可以抵 25 元。",
        "intent": "member",
        "keywords": ["25"],
        "ood": False,
        "knowledge_source": "kb.py MEMBER_RULE(100 积分抵 5 元,500 分 = 25 元)",
        "novelty_reason": "具体积分数的抵扣换算,已有只问\"积分有什么用\"。",
    },
    {
        "id": "eval_member_003",
        "q": "消费满 100 元能积多少分?",
        "a": "消费 1 元积 1 分,100 元可以积 100 分。",
        "intent": "member",
        "keywords": ["100"],
        "ood": False,
        "knowledge_source": "kb.py MEMBER_RULE",
        "novelty_reason": "换一个消费档位做积分换算,与 30 元档位是不同计算实例。",
    },
    {
        "id": "eval_member_004",
        "q": "消费一元能积几分?",
        "a": "消费 1 元积 1 分。",
        "intent": "member",
        "keywords": ["1"],
        "ood": False,
        "knowledge_source": "kb.py MEMBER_RULE",
        "novelty_reason": "直接问积分单位比例,是把既有规则拆到最小颗粒度来考。",
    },
    {
        "id": "eval_member_005",
        "q": "我是第一次来,有什么优惠吗?",
        "a": "有,新客注册立减 3 元。",
        "intent": "member",
        "keywords": ["3"],
        "ood": False,
        "knowledge_source": "kb.py COUPON_NEW(新客注册立减 3 元)",
        "novelty_reason": "首单新客视角,已有问的是\"有优惠券吗/怎么成为会员\"。",
    },
    # ================= store(5;补足 test 到 10)=================
    {
        "id": "eval_store_001",
        "q": "茶语时光主打什么?",
        "a": "茶语时光 2019 年创立,主打新鲜现做的茶饮。",
        "intent": "store",
        "keywords": ["新鲜现做"],
        "ood": False,
        "knowledge_source": "kb.py BRAND_STORY",
        "novelty_reason": "品牌定位询问,已有的是\"你们是什么品牌\"这种身份问题。",
    },
    {
        "id": "eval_store_002",
        "q": "你们官方小程序能做什么?",
        "a": "可以在小程序里查看离你最近的门店。",
        "intent": "store",
        "keywords": ["小程序"],
        "ood": False,
        "knowledge_source": "kb.py STORE_NOTE(可通过小程序查看最近门店)",
        "novelty_reason": "问小程序的功能,已有的是\"怎么找最近门店\"这一步操作。",
    },
    {
        "id": "eval_store_003",
        "q": "你们有官方小程序吗?",
        "a": "有的,可以通过茶语时光小程序查看最近的门店。",
        "intent": "store",
        "keywords": ["小程序"],
        "ood": False,
        "knowledge_source": "kb.py STORE_NOTE",
        "novelty_reason": "存在性确认(有没有小程序),与\"怎么用小程序找店\"不同。",
    },
    {
        "id": "eval_store_004",
        "q": "你们的门店多吗?",
        "a": "门店不少,全国有 200 多家。",
        "intent": "store",
        "keywords": ["200"],
        "ood": False,
        "knowledge_source": "kb.py STORE_NOTE(全国 200+ 家门店)",
        "novelty_reason": "规模化描述询问,不同于\"附近有门店吗\"这种定位问题。",
    },
    {
        "id": "eval_store_005",
        "q": "门店提供哪些额外服务?",
        "a": "门店提供免费 WiFi。",
        "intent": "store",
        "keywords": ["WiFi"],
        "ood": False,
        "knowledge_source": "kb.py WIFI_NOTE",
        "novelty_reason": "服务清单询问,已有的是\"有 WiFi 吗\"这种单点确认。",
    },
    # ================= ood(10)=================
    # 说明:现有 OOD 已覆盖修手机/系统/出行/票务/保险/装修/驾照/培训/签证/发票/二手车/
    #      违章/疫苗/水电/空调/钥匙/快递/开锁/搬家/家教/股票/天气/租房/招聘/论文/话费。
    #      下面补充**完全不同领域**的 OOD,答案统一为拒答话术。
    {
        "id": "eval_ood_001",
        "q": "帮我写一段 Python 爬虫代码",
        "a": "抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。",
        "intent": "ood",
        "keywords": ["抱歉", "无法回答"],
        "ood": True,
        "knowledge_source": "kb.py REFUSAL_TEMPLATES(知识库外统一拒答)",
        "novelty_reason": "编程类 OOD,现有 OOD 主题里没有编程。",
    },
    {
        "id": "eval_ood_002",
        "q": "帮我算一下 128 乘以 37 等于多少",
        "a": "不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。",
        "intent": "ood",
        "keywords": ["不好意思", "服务范围"],
        "ood": True,
        "knowledge_source": "kb.py REFUSAL_TEMPLATES",
        "novelty_reason": "数学计算类 OOD,现有 OOD 没有纯计算题。",
    },
    {
        "id": "eval_ood_003",
        "q": "帮我规划一下去三亚的五天行程",
        "a": "抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。",
        "intent": "ood",
        "keywords": ["抱歉", "无法回答"],
        "ood": True,
        "knowledge_source": "kb.py REFUSAL_TEMPLATES",
        "novelty_reason": "旅游规划类 OOD(已有只有\"退机票/订酒店\"这类票务操作)。",
    },
    {
        "id": "eval_ood_004",
        "q": "隔壁那家奶茶店有什么新品?",
        "a": "不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。",
        "intent": "ood",
        "keywords": ["不好意思", "服务范围"],
        "ood": True,
        "knowledge_source": "kb.py REFUSAL_TEMPLATES",
        "novelty_reason": "其他品牌类 OOD,测试模型是否会为竞品编造信息(现有 OOD 没有竞品问题)。",
    },
    {
        "id": "eval_ood_005",
        "q": "我租房合同有纠纷,能帮我看看吗?",
        "a": "抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。",
        "intent": "ood",
        "keywords": ["抱歉", "无法回答"],
        "ood": True,
        "knowledge_source": "kb.py REFUSAL_TEMPLATES",
        "novelty_reason": "法律咨询类 OOD(已有\"附近哪里可以租房\"是生活服务类,主题不同)。",
    },
    {
        "id": "eval_ood_006",
        "q": "我家猫最近不吃东西怎么办?",
        "a": "不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。",
        "intent": "ood",
        "keywords": ["不好意思", "服务范围"],
        "ood": True,
        "knowledge_source": "kb.py REFUSAL_TEMPLATES",
        "novelty_reason": "宠物咨询类 OOD,现有 OOD 主题完全没有宠物。",
    },
    {
        "id": "eval_ood_007",
        "q": "昨晚那场球赛谁赢了?",
        "a": "抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。",
        "intent": "ood",
        "keywords": ["抱歉", "无法回答"],
        "ood": True,
        "knowledge_source": "kb.py REFUSAL_TEMPLATES",
        "novelty_reason": "体育赛事类 OOD,与已有的生活服务主题不同。",
    },
    {
        "id": "eval_ood_008",
        "q": "推荐几部好看的电影",
        "a": "不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。",
        "intent": "ood",
        "keywords": ["不好意思", "服务范围"],
        "ood": True,
        "knowledge_source": "kb.py REFUSAL_TEMPLATES",
        "novelty_reason": "影视推荐类 OOD,测试模型是否会把\"推荐\"能力错误地用到领域外。",
    },
    {
        "id": "eval_ood_009",
        "q": "帮我把这句话翻译成英文",
        "a": "抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。",
        "intent": "ood",
        "keywords": ["抱歉", "无法回答"],
        "ood": True,
        "knowledge_source": "kb.py REFUSAL_TEMPLATES",
        "novelty_reason": "翻译类 OOD,属于通用语言能力请求,不在客服职责内。",
    },
    {
        "id": "eval_ood_010",
        "q": "番茄炒蛋怎么做才好吃?",
        "a": "不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。",
        "intent": "ood",
        "keywords": ["不好意思", "服务范围"],
        "ood": True,
        "knowledge_source": "kb.py REFUSAL_TEMPLATES",
        "novelty_reason": "菜谱类 OOD,与饮品制作不同,属于领域外生活问题。",
    },
]


# 知识库当前无法可靠回答、因此**刻意没有造题**的场景(记录在审计报告中)
UNSUPPORTED_GOLD_SCENARIOS = [
    {
        "intent": "price",
        "scenario": "杯型/规格价格(中杯/大杯/升杯加价)",
        "missing_fact": "kb.py PRODUCTS 只有单一价格,没有杯型/容量字段",
    },
    {
        "intent": "price",
        "scenario": "套餐价格、季节限定价格",
        "missing_fact": "kb.py 没有套餐与季节限定商品",
    },
    {
        "intent": "allergen",
        "scenario": "坚果/花生/芝麻过敏原,小料的过敏原",
        "missing_fact": "kb.py ALLERGENS 只覆盖 4 款产品的 4 类提示,没有小料与其他过敏原",
    },
    {
        "intent": "after_sale",
        "scenario": "漏发/错发/少送、异物、超时未送达的处理规则",
        "missing_fact": "kb.py 只有 REFUND_RULE 与 DELIVERY_REFUND 两条政策,未规定漏发与超时",
    },
    {
        "intent": "member",
        "scenario": "积分有效期、积分与优惠券叠加、生日福利",
        "missing_fact": "kb.py MEMBER_RULE / COUPON_* 未涉及有效期与叠加规则",
    },
    {
        "intent": "store",
        "scenario": "加盟合作、WiFi 密码、座位/环境、门店地址清单",
        "missing_fact": "kb.py STORE_NOTE 只有门店数量与小程序查店,没有上述信息",
    },
    {
        "intent": "delivery",
        "scenario": "超出 3 公里的处理、配送时长承诺、配送费退还",
        "missing_fact": "kb.py 只定义 3 公里/3 元/满 25 元免,未定义超范围与时效承诺",
    },
    {
        "intent": "customize",
        "scenario": "常温/加冰量以外的规格定制(如加浓、换奶)",
        "missing_fact": "kb.py 未定义糖度/冰量以外的定制选项",
    },
]

REQUIRED_FIELDS = ("id", "q", "a", "intent", "keywords", "ood", "knowledge_source", "novelty_reason")


def validate_gold_eval_seeds():
    """结构自检:字段齐全、id 唯一、答案非空。"""
    ids = set()
    for row in GOLD_EVAL_SEEDS:
        for field in REQUIRED_FIELDS:
            if row.get(field) in (None, "", []):
                raise ValueError(f"gold eval seed {row.get('id')} 缺少字段: {field}")
        if row["id"] in ids:
            raise ValueError(f"gold eval seed id 重复: {row['id']}")
        ids.add(row["id"])
        if row["ood"] not in (True, False):
            raise ValueError(f"gold eval seed {row['id']} 的 ood 必须是 bool")
    return len(GOLD_EVAL_SEEDS)
```

### `data_eng/kb.py`

```python
"""知识库:虚构奶茶品牌"茶语时光"的完整业务资料。

这是整个后训练数据的事实来源:
  1. 所有"答案"都从这份知识库生成,保证事实一致
  2. 知识库外的问题(不在这里面)应被拒答 —— 这是对齐的核心场景
"""

# 菜单:产品、价格、卖点标签
PRODUCTS = [
    {"name": "经典珍珠奶茶", "price": 12, "tags": ["经典", "奶茶", "珍珠"]},
    {"name": "黑糖珍珠鲜奶", "price": 15, "tags": ["黑糖", "鲜奶"]},
    {"name": "芝士奶盖四季春", "price": 16, "tags": ["芝士", "奶盖", "茶"]},
    {"name": "芋泥波波鲜奶", "price": 17, "tags": ["芋泥", "鲜奶"]},
    {"name": "杨枝甘露", "price": 18, "tags": ["芒果", "西柚", "西米"]},
    {"name": "焦糖布丁奶茶", "price": 14, "tags": ["焦糖", "布丁"]},
    {"name": "茉莉绿茶", "price": 8, "tags": ["绿茶", "茉莉", "清淡"]},
    {"name": "柠檬气泡水", "price": 10, "tags": ["柠檬", "气泡水"]},
]

# 小料及加价(元)
TOPPINGS = [("珍珠", 2), ("椰果", 2), ("芋泥", 3), ("奶盖", 4), ("布丁", 3)]

# 营业信息
OPEN_TIME = "10:00"
CLOSE_TIME = "22:00"
DELIVERY_CUTOFF = "21:30"
DELIVERY_RADIUS = "3 公里"
DELIVERY_FEE = "3 元"
DELIVERY_FREE_OVER = 25

# 会员与优惠
MEMBER_RULE = "1 元消费积 1 分,100 积分可抵 5 元"
COUPON_NEW = "新客注册立减 3 元"
COUPON_WED = "每周三第二杯半价"

# 售后政策
REFUND_RULE = "饮品出品 30 分钟内口味不满意可免费重做或退款"
DELIVERY_REFUND = "外卖洒漏或破损,拍照联系客服可退款"

# 过敏原提示
ALLERGENS = {
    "杨枝甘露": "含芒果",
    "芝士奶盖四季春": "含乳制品",
    "芋泥波波鲜奶": "含牛奶和芋泥",
    "黑糖珍珠鲜奶": "含牛奶",
}

# 门店信息
STORE_NOTE = "全国 200+ 家门店,可通过小程序查看最近门店"
BRAND_STORY = "茶语时光 2019 年创立,主打新鲜现做的茶饮"
WIFI_NOTE = "门店提供免费 WiFi"

# 拒答话术(知识库外问题统一使用)
REFUSAL_TEMPLATES = [
    "抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。",
    "不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。",
]
```

### `data_eng/leakage_audit.py`

````python
"""数据血缘(leakage)审计工具:文本归一化、稳定 ID、多层泄漏检查、报告生成。

设计原则:
  1. 只读:本模块不修改任何数据文件,只做计算与报告输出。
  2. 可复现:所有 ID 都由内容 hash 派生,相同输入必得相同 ID / 相同报告。
  3. 可审计:每一条泄漏指标都同时给出"值集合交集大小"和"行级命中数",避免口径歧义。

三层泄漏口径(由细到粗):
  * exact  : normalized question(小写 + 去空白 + 统一中英文标点)完全相同
  * family : family_id 相同(= intent + base_question 相同,prefix 已剥离)
  * group  : split_group_id 相同(= 同一实体 / 同一 seed 的血缘连通分量,最严格)

可单独执行对任意数据目录做审计:
  python data_eng/leakage_audit.py --data-dir data_eng/output_clean --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------- 文本归一化

# 中英文标点统一:全部映射到半角
PUNCT_TABLE = str.maketrans(
    {
        "？": "?",
        "，": ",",
        "。": ".",
        "！": "!",
        "：": ":",
        "；": ";",
        "、": ",",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "～": "~",
        "％": "%",
        "　": " ",
    }
)

# prefix paraphrase 模板产生的前缀(长的放前面,避免"请问"先吃掉"请问哪里")
PARAPHRASE_PREFIXES = (
    "麻烦问一下,",
    "麻烦问一下，",
    "你好,",
    "你好，",
    "请问",
    "亲,",
    "亲，",
)

SPLIT_NAMES = ("train", "val", "test")
MODEL_GENERATED_SOURCES = ("evolved", "onpolicy")

AUDIT_FIELDS = (
    "sample_id",
    "source",
    "variant_type",
    "intent",
    "family_id",
    "split_group_id",
    "base_question",
)


def normalize_text(text) -> str:
    """小写 + 统一中英文标点 + 去掉所有空白。"""
    t = str(text).lower().translate(PUNCT_TABLE)
    t = re.sub(r"\s+", "", t)
    return t.strip()


def normalize(text) -> str:
    """兼容旧接口(build_dataset.normalize / evolve.py 曾使用)。"""
    return normalize_text(text)


def strip_paraphrase_prefix(text, prefixes=PARAPHRASE_PREFIXES) -> str:
    """反复剥离 prefix paraphrase 前缀,直到不再变化。

    必须反复剥离:模板本身可能已含"请问"(例如 "请问{name}的价格是?"),
    paraphrase 之后会出现 "请问请问..."。
    """
    cur = str(text).strip()
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if cur.startswith(p) and len(cur) > len(p) + 1:
                cur = cur[len(p):].strip()
                changed = True
                break
    return cur


def base_question(text) -> str:
    """去掉口语化前缀后的"基础问题",是 family 的判定依据。"""
    return strip_paraphrase_prefix(text)


# ---------------------------------------------------------------- 稳定 ID


def stable_id(prefix: str, *parts, width: int = 12) -> str:
    payload = "\u241f".join(str(p) for p in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:width]}"


def family_id_of(intent: str, base_q: str) -> str:
    """family = (intent, 基础问题);同一基础问题的所有改写共享同一 family。"""
    return stable_id("fam_", intent, normalize_text(base_q))


def sample_id_of(intent: str, q: str, a: str) -> str:
    """sample_id 由内容派生:同一 (intent, q, a) 在任意一次重建中 ID 都不变。"""
    return stable_id("s_", intent, normalize_text(q), normalize_text(a))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- IO


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 审计核心


def _q_set(rows):
    return {normalize_text(r["q"]) for r in rows}


def _base_set(rows):
    return {normalize_text(r.get("base_question") or base_question(r["q"])) for r in rows}


def _family_set(rows):
    return {r["family_id"] for r in rows if r.get("family_id")}


def _group_set(rows):
    return {r["split_group_id"] for r in rows if r.get("split_group_id")}


def _ood_rows(rows):
    return [r for r in rows if r.get("ood")]


def _ood_family_set(rows):
    return {r["family_id"] for r in _ood_rows(rows) if r.get("family_id")}


def _ood_entity_set(rows):
    return {r.get("entity_key") for r in _ood_rows(rows) if r.get("entity_key")}


def _counter(values):
    return dict(sorted(Counter(values).items(), key=lambda kv: (-kv[1], str(kv[0]))))


def _pairwise_overlap(set_fn, splits):
    value_sets = {name: set_fn(rows) for name, rows in splits.items()}
    overlap = {}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            overlap[f"{a}-{b}"] = len(value_sets[a] & value_sets[b])
    return overlap


def _sample_stats(rows):
    ood = _ood_rows(rows)
    return {
        "samples": len(rows),
        "unique_questions": len(_q_set(rows)),
        "unique_base_questions": len(_base_set(rows)),
        "unique_families": len(_family_set(rows)),
        "unique_groups": len(_group_set(rows)),
        "unique_intents": len({r.get("intent") for r in rows}),
        "ood_samples": len(ood),
        "ood_families": len(_ood_family_set(rows)),
        "ood_entities": sorted(x for x in _ood_entity_set(rows) if x),
    }


def audit_splits(
    splits: dict,
    *,
    dataset_name: str = "clean",
    inputs: dict | None = None,
    raw_rows: list | None = None,
    lineage: dict | None = None,
    notes: list | None = None,
) -> dict:
    """对 {train,val,test} 做多层泄漏审计;splits 的值是行 list。"""
    splits = {name: list(splits.get(name, [])) for name in SPLIT_NAMES}

    # ---- 字段完整性
    missing = defaultdict(int)
    for name, rows in splits.items():
        for r in rows:
            for field in AUDIT_FIELDS:
                if r.get(field) in (None, ""):
                    missing[f"{name}.{field}"] += 1

    exact_overlap = _pairwise_overlap(_q_set, splits)
    base_overlap = _pairwise_overlap(_base_set, splits)
    family_overlap = _pairwise_overlap(_family_set, splits)
    group_overlap = _pairwise_overlap(_group_set, splits)
    ood_family_overlap = _pairwise_overlap(_ood_family_set, splits)
    ood_entity_overlap = _pairwise_overlap(_ood_entity_set, splits)

    # ---- 行级污染面:另一个 split 有多少行"撞上"了本 split 的值
    exact_row_hits, base_row_hits, family_row_hits = {}, {}, {}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a == b:
                continue
            other_q = _q_set(splits[b])
            other_b = _base_set(splits[b])
            other_f = _family_set(splits[b])
            exact_row_hits[f"{a}<-{b}"] = sum(1 for r in splits[a] if normalize_text(r["q"]) in other_q)
            base_row_hits[f"{a}<-{b}"] = sum(
                1 for r in splits[a]
                if normalize_text(r.get("base_question") or base_question(r["q"])) in other_b
            )
            family_row_hits[f"{a}<-{b}"] = sum(1 for r in splits[a] if r.get("family_id") in other_f)

    # ---- split 内部重复(信息项,不参与断言)
    dup_within = {}
    for name, rows in splits.items():
        qa = Counter((normalize_text(r["q"]), normalize_text(r["a"])) for r in rows)
        qq = Counter(normalize_text(r["q"]) for r in rows)
        dup_within[name] = {
            "duplicate_qa_pairs": sum(c - 1 for c in qa.values() if c > 1),
            "duplicate_questions": sum(c - 1 for c in qq.values() if c > 1),
        }

    source_dist = {name: _counter(r.get("source", "<missing>") for r in rows) for name, rows in splits.items()}
    intent_dist = {name: _counter(r.get("intent", "<missing>") for r in rows) for name, rows in splits.items()}
    variant_dist = {name: _counter(r.get("variant_type", "<missing>") for r in rows) for name, rows in splits.items()}

    test_model_generated = {src: source_dist["test"].get(src, 0) for src in MODEL_GENERATED_SOURCES}
    val_model_generated = {src: source_dist["val"].get(src, 0) for src in MODEL_GENERATED_SOURCES}

    unknown_rows = [r for rows in splits.values() for r in rows if r.get("lineage_confidence") == "unknown"]
    ambiguous_rows = [
        r for rows in splits.values() for r in rows if r.get("lineage_confidence") == "ambiguous_union"
    ]

    raw_stats = None
    if raw_rows is not None:
        raw_stats = _sample_stats(raw_rows)
        raw_stats["source_distribution"] = _counter(r.get("source", "<missing>") for r in raw_rows)
        raw_stats["intent_distribution"] = _counter(r.get("intent", "<missing>") for r in raw_rows)

    assertions = []

    def _assert(name, passed, detail):
        assertions.append({"name": name, "passed": bool(passed), "detail": detail})

    _assert(
        "exact normalized question never crosses split",
        all(v == 0 for v in exact_overlap.values()),
        f"overlap={exact_overlap}",
    )
    _assert(
        "family never crosses split",
        all(v == 0 for v in family_overlap.values()),
        f"overlap={family_overlap}",
    )
    _assert(
        "base question never crosses split",
        all(v == 0 for v in base_overlap.values()),
        f"overlap={base_overlap}",
    )
    _assert(
        "split group never crosses split",
        all(v == 0 for v in group_overlap.values()),
        f"overlap={group_overlap}",
    )
    _assert(
        "OOD family never crosses split",
        all(v == 0 for v in ood_family_overlap.values()),
        f"overlap={ood_family_overlap} (train={len(_ood_family_set(splits['train']))}, "
        f"val={len(_ood_family_set(splits['val']))}, test={len(_ood_family_set(splits['test']))})",
    )
    _assert(
        "OOD entity/topic never crosses split",
        all(v == 0 for v in ood_entity_overlap.values()),
        f"overlap={ood_entity_overlap}",
    )
    _assert(
        "test contains 0 model-generated samples (onpolicy/evolved)",
        all(v == 0 for v in test_model_generated.values()),
        f"test model-generated = {test_model_generated}",
    )
    _assert(
        "val contains 0 model-generated samples (onpolicy/evolved)",
        all(v == 0 for v in val_model_generated.values()),
        f"val model-generated = {val_model_generated}",
    )
    _assert(
        "all samples carry complete lineage metadata",
        not missing,
        "no missing fields" if not missing else f"missing_fields={dict(missing)}",
    )
    _assert(
        "train/val/test all non-empty",
        all(len(splits[name]) > 0 for name in SPLIT_NAMES),
        f"sizes={{k: len(v) for k, v in splits.items()}}".replace("{k: len(v) for k, v in splits.items()}", str({n: len(r) for n, r in splits.items()})),
    )
    _assert(
        "no unknown-lineage sample in val/test",
        all(r.get("lineage_confidence") != "unknown" for name in ("val", "test") for r in splits[name]),
        f"unknown_lineage_samples_total={len(unknown_rows)}",
    )

    all_intents = sorted({r.get("intent") for rows in splits.values() for r in rows})
    intent_coverage = {name: sorted({r.get("intent") for r in rows}) for name, rows in splits.items()}
    missing_intents = {name: sorted(set(all_intents) - set(intent_coverage[name])) for name in SPLIT_NAMES}

    audit = {
        "dataset": dataset_name,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": inputs or {},
        "raw_dataset": raw_stats,
        "split": {name: _sample_stats(splits[name]) for name in SPLIT_NAMES},
        "source_distribution": source_dist,
        "variant_distribution": variant_dist,
        "intent_distribution": intent_dist,
        "intent_coverage": intent_coverage,
        "missing_intents_per_split": missing_intents,
        "ood": {
            "counts": {name: _sample_stats(splits[name])["ood_samples"] for name in SPLIT_NAMES},
            "families": {name: _sample_stats(splits[name])["ood_families"] for name in SPLIT_NAMES},
            "entities": {name: _sample_stats(splits[name])["ood_entities"] for name in SPLIT_NAMES},
            "family_overlap": ood_family_overlap,
            "entity_overlap": ood_entity_overlap,
        },
        "leakage": {
            "exact_question_overlap": exact_overlap,
            "base_question_overlap": base_overlap,
            "family_overlap": family_overlap,
            "split_group_overlap": group_overlap,
            "row_level": {
                "exact_question_rows": exact_row_hits,
                "base_question_rows": base_row_hits,
                "family_rows": family_row_hits,
            },
            "duplicate_within_split": dup_within,
        },
        "test_purity": {
            "model_generated_sources": test_model_generated,
            "test_onpolicy": source_dist["test"].get("onpolicy", 0),
            "test_evolved": source_dist["test"].get("evolved", 0),
        },
        "unknown_lineage": {
            "unknown_family_count": len(unknown_rows),
            "ambiguous_lineage_count": len(ambiguous_rows),
            "unknown_by_source": _counter(r.get("source", "<missing>") for r in unknown_rows),
            "ambiguous_by_source": _counter(r.get("source", "<missing>") for r in ambiguous_rows),
        },
        "lineage": lineage or {},
        "assertions": assertions,
        "all_passed": all(a["passed"] for a in assertions),
        "notes": notes or [],
    }
    return audit


# ---------------------------------------------------------------- 报告


def _table(counter_dict, key_header="key"):
    lines = [f"| {key_header} | count |", "|---|---|"]
    if not counter_dict:
        lines.append("| (empty) | 0 |")
    for k, v in counter_dict.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def render_markdown(audit: dict) -> str:
    o = []
    o.append("# 数据集划分与泄漏审计报告 (DATASET LEAKAGE AUDIT)")
    o.append("")
    o.append(f"- dataset: `{audit['dataset']}`")
    o.append(f"- generated_at: {audit['generated_at']}")
    for k, v in (audit.get("inputs") or {}).items():
        o.append(f"- input `{k}`: {v}")
    o.append(f"- **all assertions passed: `{audit['all_passed']}`**")
    if audit.get("reused_build_sections"):
        o.append(
            "- 注:本次为独立重跑审计,`raw dataset / lineage / notes / legacy_baseline` "
            "段落沿用最近一次完整构建(`build_dataset.py`)的输出。"
        )
    o.append("")

    if audit.get("notes"):
        o.append("## 0. 说明与已知残余风险")
        o.append("")
        for n in audit["notes"]:
            o.append(f"- {n}")
        o.append("")

    raw = audit.get("raw_dataset")
    if raw:
        o.append("## 1. Raw dataset(去重清洗后、切分前)")
        o.append("")
        o.append("```text")
        o.append(f"total samples           : {raw['samples']}")
        o.append(f"unique questions        : {raw['unique_questions']}")
        o.append(f"unique base questions   : {raw['unique_base_questions']}")
        o.append(f"unique families         : {raw['unique_families']}")
        o.append(f"unique split groups     : {raw['unique_groups']}")
        o.append(f"unique intents          : {raw['unique_intents']}")
        o.append(f"OOD samples / families  : {raw['ood_samples']} / {raw['ood_families']}")
        o.append("```")
        o.append("")
        o.append("来源分布(切分前):")
        o.append("")
        o.append(_table(raw["source_distribution"], "source"))
        o.append("")

    o.append("## 2. Split")
    o.append("")
    o.append("| split | samples | questions | base questions | families | groups | intents | OOD samples | OOD families |")
    o.append("|---|---|---|---|---|---|---|---|---|")
    for name in SPLIT_NAMES:
        s = audit["split"][name]
        o.append(
            f"| {name} | {s['samples']} | {s['unique_questions']} | {s['unique_base_questions']} | "
            f"{s['unique_families']} | {s['unique_groups']} | {s['unique_intents']} | "
            f"{s['ood_samples']} | {s['ood_families']} |"
        )
    o.append("")

    o.append("## 3. Source distribution")
    o.append("")
    for name in SPLIT_NAMES:
        o.append(f"### {name}")
        o.append("")
        o.append(_table(audit["source_distribution"][name], "source"))
        o.append("")

    o.append("## 4. Intent distribution")
    o.append("")
    for name in SPLIT_NAMES:
        o.append(f"### {name}")
        o.append("")
        o.append(_table(audit["intent_distribution"][name], "intent"))
        o.append("")
    o.append("intent 覆盖(缺失项会被显式列出,不做任何数据修补):")
    o.append("")
    for name in SPLIT_NAMES:
        miss = audit["missing_intents_per_split"][name]
        o.append(f"- {name}: covered={audit['intent_coverage'][name]} | **missing={miss}**")
    o.append("")

    o.append("## 5. OOD distribution")
    o.append("")
    o.append("| split | OOD samples | OOD families | OOD entities |")
    o.append("|---|---|---|---|")
    for name in SPLIT_NAMES:
        o.append(
            f"| {name} | {audit['ood']['counts'][name]} | {audit['ood']['families'][name]} | "
            f"{len(audit['ood']['entities'][name])} |"
        )
    o.append("")
    o.append("交集(必须全为 0):")
    o.append("")
    o.append("```text")
    o.append(f"OOD family overlap : {audit['ood']['family_overlap']}")
    o.append(f"OOD entity overlap : {audit['ood']['entity_overlap']}")
    o.append("```")
    o.append("")
    for name in SPLIT_NAMES:
        ents = audit["ood"]["entities"][name]
        o.append(f"- {name} OOD entities({len(ents)}): {', '.join(ents) if ents else '(none)'}")
    o.append("")

    leak = audit["leakage"]
    o.append("## 6. Leakage")
    o.append("")
    o.append("### 6.1 Exact duplicate (normalized question)")
    o.append("")
    o.append("```text")
    for k, v in leak["exact_question_overlap"].items():
        o.append(f"{k}: {v}")
    o.append("```")
    o.append("")
    o.append("### 6.2 Same-family leakage")
    o.append("")
    o.append("```text")
    for k, v in leak["family_overlap"].items():
        o.append(f"{k}: {v}")
    o.append("```")
    o.append("")
    o.append("### 6.3 Base-question leakage (prefix paraphrase 已剥离)")
    o.append("")
    o.append("```text")
    for k, v in leak["base_question_overlap"].items():
        o.append(f"{k}: {v}")
    o.append("```")
    o.append("")
    o.append("### 6.4 Split-group leakage (实体级 / seed 级连通分量)")
    o.append("")
    o.append("```text")
    for k, v in leak["split_group_overlap"].items():
        o.append(f"{k}: {v}")
    o.append("```")
    o.append("")
    o.append("### 6.5 行级污染面")
    o.append("")
    o.append("```text")
    for label, payload in (
        ("exact question rows", leak["row_level"]["exact_question_rows"]),
        ("base question rows", leak["row_level"]["base_question_rows"]),
        ("family rows", leak["row_level"]["family_rows"]),
    ):
        o.append(f"{label}: {payload}")
    o.append("```")
    o.append("")
    o.append("### 6.6 split 内部重复(信息项)")
    o.append("")
    o.append(
        _table(
            {
                f"{k}.{kk}": vv
                for k, v in leak["duplicate_within_split"].items()
                for kk, vv in v.items()
            },
            "metric",
        )
    )
    o.append("")

    o.append("## 7. Test purity")
    o.append("")
    o.append("```text")
    o.append(f"test onpolicy = {audit['test_purity']['test_onpolicy']}")
    o.append(f"test evolved  = {audit['test_purity']['test_evolved']}")
    o.append(f"test model-generated total = {audit['test_purity']['model_generated_sources']}")
    o.append("```")
    o.append("")

    o.append("## 8. Unknown / ambiguous lineage")
    o.append("")
    u = audit["unknown_lineage"]
    o.append("```text")
    o.append(f"unknown_family_count    = {u['unknown_family_count']}")
    o.append(f"ambiguous_lineage_count = {u['ambiguous_lineage_count']}")
    o.append(f"unknown by source       = {u['unknown_by_source']}")
    o.append(f"ambiguous by source     = {u['ambiguous_by_source']}")
    o.append("```")
    o.append("")
    lin = audit.get("lineage") or {}
    if lin:
        o.append("### lineage 恢复统计")
        o.append("")
        o.append("```text")
        for k, v in lin.items():
            if not isinstance(v, (dict, list)):
                o.append(f"{k} = {v}")
        o.append("```")
        o.append("")
        for k, v in lin.items():
            if isinstance(v, dict):
                o.append(f"**{k}**")
                o.append("")
                o.append(_table({str(kk): vv for kk, vv in v.items()}, "key"))
                o.append("")
            elif isinstance(v, list):
                o.append(f"- {k}: {v}")
        o.append("")

    ge = audit.get("gold_eval")
    if ge:
        o.append("## 9. Gold evaluation set(人工 Gold Evaluation Seed)")
        o.append("")
        o.append("```text")
        o.append(f"gold_eval count        = {ge.get('count')} (expected {ge.get('expected_count')})")
        o.append(f"gold_eval in test      = {ge.get('in_test')}")
        o.append(f"by intent              = {ge.get('by_intent')}")
        o.append(f"test intent counts     = {ge.get('test_intent_counts')}")
        o.append(f"test missing intents   = {ge.get('test_missing_intents')}")
        o.append(f"previous test preserved= {ge.get('previous_test_preserved')}")
        o.append(f"previous test sha256    = {ge.get('previous_test_sha256')}")
        o.append("```")
        o.append("")
        o.append("gold_eval 与 train / val / 旧 test 的多层去重(必须全为 0):")
        o.append("")
        o.append("```text")
        for name, d in (ge.get("overlap") or {}).items():
            o.append(f"gold_eval ∩ {name}: {d}")
        o.append("```")
        o.append("")
        o.append("逐题清单见 `GOLD_EVAL_AUDIT.md`(含 knowledge source / novelty reason / leakage check)。")
        o.append("")
    o.append("## 10. Assertions")
    o.append("")
    o.append("| assertion | result | detail |")
    o.append("|---|---|---|")
    for a in audit["assertions"]:
        detail = str(a["detail"]).replace("|", "/")
        o.append(f"| {a['name']} | {'PASS' if a['passed'] else '**FAIL**'} | {detail} |")
    o.append("")
    return "\n".join(o)


def write_reports(audit: dict, out_dir: str) -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "leakage_audit.json")
    md_path = os.path.join(out_dir, "DATASET_LEAKAGE_AUDIT.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(audit))
    return json_path, md_path


def audit_directory(data_dir: str, dataset_name: str | None = None) -> dict:
    splits = {name: load_jsonl(os.path.join(data_dir, f"{name}.jsonl")) for name in SPLIT_NAMES}
    audit = audit_splits(splits, dataset_name=dataset_name or os.path.basename(os.path.abspath(data_dir)))

    # 单独重跑审计时不丢信息:沿用最近一次完整构建写入的 raw / lineage / notes / legacy 段落
    previous_path = os.path.join(data_dir, "leakage_audit.json")
    if os.path.exists(previous_path):
        try:
            with open(previous_path, encoding="utf-8") as f:
                previous = json.load(f)
        except (json.JSONDecodeError, OSError):
            previous = {}
        carried = [k for k in ("raw_dataset", "lineage", "notes", "legacy_baseline") if previous.get(k)]
        for key in carried:
            audit[key] = previous[key]
        if carried:
            audit["reused_build_sections"] = carried
    return audit


def main():
    parser = argparse.ArgumentParser(description="对 train/val/test 做多层泄漏审计")
    parser.add_argument("--data-dir", type=str, default="data_eng/output_clean")
    parser.add_argument("--write", action="store_true", help="把审计结果写回该目录")
    args = parser.parse_args()

    audit = audit_directory(args.data_dir)
    if args.write:
        json_path, md_path = write_reports(audit, args.data_dir)
        print(f"审计报告已写入: {json_path} / {md_path}")
    print(render_markdown(audit))
    print(f"\nALL ASSERTIONS PASSED: {audit['all_passed']}")
    return 0 if audit["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
````

### `data_eng/run_llm_judge.py`

```python
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

    # ⚠️ 修复点:原实现把"183 条"硬编码进标题,换评测集后会写出与事实不符的样本量。
    # 改为从实际预测文件读取条数,标题不再绑定某个历史数据集。
    n_rows = len(load_jsonl(pred_files[0])) if pred_files else 0
    lines = [f"# LLM-as-Judge 评测结果(每模型 {n_rows} 条,复用项目原有 judge prompt)", ""]
    lines.append(f"- Judge model:`{args.judge_model}`(与 `sft/eval_domain.py --judge-model` 默认一致)")
    lines.append(f"- 数据来源:`{args.predictions_dir}/*.jsonl`(**直接复用已有回答,未重新推理**)")
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
```

### `data_eng/run_pipeline.py`

```python
"""按项目真实依赖顺序重跑全流程;所有产物写入 --root,**不覆盖任何现有文件**。

为什么是这个顺序(SFT → RFT → 合并 → SFT → DPO/RL):
  1. `build_dataset`(不带额外数据)产出"纯 gold"数据 v1;
  2. 用 v1 训一个 SFT(sft_v1),它代表"当前 policy";
  3. `rft.py` 从 sft_v1 对 v1 的 train 做 on-policy 采样 + 规则验证器筛选(拒绝采样);
  4. `build_dataset` 把 evolved(Self-Instruct)与 onpolicy(RFT)合并进 gold,做血缘/group 切分
     → clean 数据集(test 仍只含 gold,零泄漏);
  5. 用 clean train 训最终 SFT;
  6. DPO 与 RL(DAPO)都从**最终 SFT** 出发(两个并列分支,不是串行链)。

每步的命令 / 起止时间 / 退出码 / 耗时 / 关键产物都写入 <root>/manifest.jsonl 与 MANIFEST.md。

用法:
  python data_eng/run_pipeline.py --root runs/2026-10-12 --only evolve
  python data_eng/run_pipeline.py --root runs/2026-10-12 --from sft_v1
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_stages(root, input_root=None):
    """返回 {stage_name: [ [cmd...], ... ]}。路径全部指向 root,不触碰既有产物。

    input_root 只给末尾的 GSPO 消融阶段用:那三个阶段的**输入**(最终 SFT 适配器 +
    clean 数据集)来自上一轮已经跑完的 run,产物仍然写进 root,互不覆盖。
    """
    py = sys.executable
    inp = os.path.abspath(input_root) if input_root else root
    inp_models = os.path.join(inp, "models")
    inp_clean = os.path.join(inp, "data", "clean")
    data = os.path.join(root, "data")
    v1 = os.path.join(data, "v1")
    clean = os.path.join(data, "clean")
    models = os.path.join(root, "models")
    evald = os.path.join(root, "eval")
    preds = os.path.join(evald, "predictions")
    judge = os.path.join(root, "judge")
    evolved = os.path.join(data, "evolved.jsonl")
    onpolicy = os.path.join(data, "onpolicy.jsonl")
    rejected = os.path.join(data, "onpolicy_rejected.jsonl")
    pairs = os.path.join(data, "dpo_pairs.jsonl")

    return {
        # 1) 纯 gold 数据集(v1)
        "data_v1": [[
            py, "data_eng/build_dataset.py",
            "--out-dir", v1,
            "--reuse-split-from", "",          # 强制全新 group 切分(不冻结旧 split)
        ]],
        # 2) Self-Instruct 蒸馏(调智谱 GLM API)
        "evolve": [[
            py, "data_eng/evolve.py",
            "--seeds", "30", "--variants", "3", "--model", "glm-4-flash",
            "--out", evolved,
        ]],
        # 3) SFT v1(用于 RFT 的 on-policy 采样)
        "sft_v1": [[
            py, "sft/train_sft.py",
            "--data-file", os.path.join(v1, "train.jsonl"),
            "--max-steps", "600", "--max-examples", "2000", "--seed", "42",
            "--output-dir", os.path.join(models, "sft_v1"),
        ]],
        # 4) RFT:从 sft_v1 采样 + 验证器筛选(含 rejected 与通过率落盘)
        "rft": [[
            py, "rl/rft.py",
            "--adapter", os.path.join(models, "sft_v1"),
            "--data-file", os.path.join(v1, "train.jsonl"),
            "--max-rows", "400", "--num-samples", "4", "--min-score", "1.0",
            "--batch-size", "8", "--seed", "42",
            "--out", onpolicy, "--rejected-out", rejected,
        ]],
        # 5) 合并 evolved + onpolicy → clean 数据集(血缘 + group 切分 + leakage audit)
        "data_clean": [[
            py, "data_eng/build_dataset.py",
            "--extra-file", evolved,
            "--extra-file", onpolicy,
            "--out-dir", clean,
            "--reuse-split-from", "",
        ]],
        # 6) 最终 SFT(用 clean train)
        "sft": [[
            py, "sft/train_sft.py",
            "--data-file", os.path.join(clean, "train.jsonl"),
            "--max-steps", "600", "--max-examples", "2000", "--seed", "42",
            "--output-dir", os.path.join(models, "sft"),
        ]],
        # 7) DPO(从最终 SFT 出发;偏好对落盘)
        "dpo": [[
            py, "rl/train_dpo.py",
            "--adapter", os.path.join(models, "sft"),
            "--data-file", os.path.join(clean, "train.jsonl"),
            "--max-pairs", "400", "--num-samples", "4", "--pairs-batch-size", "8",
            "--max-steps", "200", "--seed", "42", "--pairs-out", pairs,
            "--output-dir", os.path.join(models, "dpo"),
        ]],
        # 8) RL / DAPO(从最终 SFT 出发)
        "rl": [[
            py, "rl/train_grpo.py",
            "--adapter", os.path.join(models, "sft"),
            "--data-file", os.path.join(clean, "train.jsonl"),
            "--mode", "domain", "--max-steps", "300", "--seed", "42",
            "--output-dir", os.path.join(models, "rl"),
        ]],
        # 9) 评测:SFT/base 用 chat(NFT 训练格式),DPO/RL 用 raw(它们自己的训练格式)
        "eval_chat": [[
            py, "sft/eval_domain.py",
            "--data-file", os.path.join(clean, "test.jsonl"),
            "--adapters", os.path.join(models, "sft"),
            "--prompt-format", "chat",
            "--predictions-dir", preds,
            "--report", os.path.join(evald, "eval_chat.md"),
        ]],
        "eval_raw": [[
            py, "sft/eval_domain.py",
            "--data-file", os.path.join(clean, "test.jsonl"),
            # checkpooint-100=DPO 早停点(对应旧矩阵里的"DPO 10 步");checkpoint-200=DPO 最终点。
            # 不再单列 models/dpo —— 已用 sha256 验证它与 checkpoint-200 的 adapter 完全一致,重复评测无信息量。
            "--adapters", os.path.join(models, "dpo", "checkpoint-100"),
            os.path.join(models, "dpo", "checkpoint-200"),
            os.path.join(models, "rl"),
            "--prompt-format", "raw",
            "--predictions-dir", preds,
            "--report", os.path.join(evald, "eval_raw.md"),
        ]],
        # 10) LLM-as-Judge(复用已有 predictions)
        "judge": [[
            py, "data_eng/run_llm_judge.py",
            "--predictions-dir", preds, "--out-dir", judge,
            "--judge-model", "glm-4-flash",
        ]],
        # ---------- GSPO 消融(唯一变量 = --seq-level) ----------
        # 与 DAPO 那轮相比,只有 loss 配置不同:
        #   DAPO : loss_type=dapo + importance_sampling_level=token
        #   GSPO : loss_type=grpo + importance_sampling_level=sequence(TRL 官方文档指定的 GSPO 复现配置)
        # 其余(起始适配器 / 数据 / 步数 / seed / batch / lr / temperature / beta)完全一致。
        "gspo": [[
            py, "rl/train_grpo.py",
            "--adapter", os.path.join(inp_models, "sft"),
            "--data-file", os.path.join(inp_clean, "train.jsonl"),
            "--mode", "domain", "--max-steps", "300", "--seed", "42",
            "--seq-level",
            "--output-dir", os.path.join(models, "grpo_seqlevel"),
        ]],
        # 与 DAPO 用同一份 clean test、同一 evaluator、同一 raw prompt 口径
        "eval_gspo": [[
            py, "sft/eval_domain.py",
            "--data-file", os.path.join(inp_clean, "test.jsonl"),
            "--adapters", os.path.join(models, "grpo_seqlevel"),
            "--prompt-format", "raw",
            "--predictions-dir", preds,
            "--report", os.path.join(evald, "eval_gspo.md"),
        ]],
        "judge_gspo": [[
            py, "data_eng/run_llm_judge.py",
            "--predictions-dir", preds, "--out-dir", judge,
            "--judge-model", "glm-4-flash",
        ]],
    }


ORDER = ["data_v1", "evolve", "sft_v1", "rft", "data_clean", "sft", "dpo", "rl",
         "eval_chat", "eval_raw", "judge",
         "gspo", "eval_gspo", "judge_gspo"]


def sha256_head(path, n=16):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def snapshot_outputs(root, stage):
    """记录该阶段关键产物的 hash(用于审计"是否真的产出了东西")。"""
    out = {}
    for sub in ("data", "models", "eval", "judge"):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            if stage not in dirpath and not dirpath.startswith(os.path.join(root, sub)):
                continue
            for fn in files:
                if fn.endswith((".jsonl", ".json", ".md", ".pt", ".safetensors")) and "manifest" not in fn:
                    p = os.path.join(dirpath, fn)
                    rel = os.path.relpath(p, root)
                    out[rel] = sha256_head(p)
    return out


def main():
    ap = argparse.ArgumentParser(description="按项目顺序重跑全流程(产物写入独立 root)")
    ap.add_argument("--root", type=str, default="runs/2026-10-12")
    ap.add_argument("--input-root", type=str, default=None,
                    help="GSPO 消融阶段的输入来源 run(默认与 --root 相同)")
    ap.add_argument("--only", action="append", default=[], help="只跑指定阶段(可多次)")
    ap.add_argument("--from", dest="from_stage", type=str, default=None, help="从该阶段开始跑")
    ap.add_argument("--list", action="store_true", help="只列出阶段")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    stages = build_stages(root, args.input_root)
    if args.list:
        for i, s in enumerate(ORDER, 1):
            print(f"{i:2d}. {s}")
        return 0

    todo = list(ORDER)
    if args.from_stage:
        todo = todo[todo.index(args.from_stage):]
    if args.only:
        todo = [s for s in todo if s in args.only]

    for sub in ("logs", "data", "models", "eval", "judge"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    manifest_path = os.path.join(root, "manifest.jsonl")

    print(f"[pipeline] root = {root}")
    print(f"[pipeline] 计划执行:{todo}")
    for stage in todo:
        log_path = os.path.join(root, "logs", f"{stage}.log")
        print(f"\n[pipeline] === {stage} === -> {log_path}")
        t0 = time.time()
        rc = 0
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n\n===== {stage} @ {datetime.now().isoformat(timespec='seconds')} =====\n")
            lf.flush()
            for cmd in stages[stage]:
                lf.write("CMD: " + " ".join(cmd) + "\n")
                lf.flush()
                rc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    break
        entry = {
            "stage": stage,
            "start": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            "duration_sec": round(time.time() - t0, 1),
            "returncode": rc,
            "log": os.path.relpath(log_path, root),
            "outputs": snapshot_outputs(root, stage),
        }
        with open(manifest_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[pipeline] {stage} rc={rc} 用时 {entry['duration_sec']}s")
        if rc != 0:
            print(f"[pipeline] 阶段 {stage} 失败,已停止;日志:{log_path}")
            return rc

    # 生成可读清单
    lines = ["# 全流程重跑 MANIFEST", "", f"- root:`{root}`", ""]
    with open(manifest_path, encoding="utf-8") as mf:
        for line in mf:
            e = json.loads(line)
            lines.append(f"## {e['stage']}")
            lines.append(f"- 时间:{e['start']} | 用时 {e['duration_sec']}s | 退出码 {e['returncode']}")
            lines.append(f"- 日志:`{e['log']}`")
            if e["outputs"]:
                lines.append("- 产物:")
                for k, v in list(e["outputs"].items())[:12]:
                    lines.append(f"  - `{k}` (sha256:{v})")
            lines.append("")
    with open(os.path.join(root, "MANIFEST.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[pipeline] 全部完成,清单:{os.path.join(root, 'MANIFEST.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `data_eng/seeds.py`

```python
"""种子数据:人工编写的领域问答(覆盖主要意图)。

种子是数据管道的起点 —— 面试官关心的"数据从哪来"的第一步:
人工基于知识库逐条编写,保证质量;数量少没关系,后面用模板扩展。

每个样本:
  intent    意图标签(统计/分层用)
  q         用户问题
  a         标准答案(来自知识库,简洁)
  keywords  判定答案是否正确的关键实体(评测/奖励用)
  ood       是否为知识库外问题(应拒答)
"""

SEEDS = [
    # ---------- 价格查询 ----------
    {"intent": "price", "q": "经典珍珠奶茶多少钱?", "a": "经典珍珠奶茶 12 元一杯。", "keywords": ["12"], "ood": False},
    {"intent": "price", "q": "杨枝甘露什么价位?", "a": "杨枝甘露 18 元一杯。", "keywords": ["18"], "ood": False},
    {"intent": "price", "q": "你们最便宜的是什么?", "a": "茉莉绿茶最便宜,8 元一杯。", "keywords": ["茉莉绿茶", "8"], "ood": False},
    {"intent": "price", "q": "芝士奶盖四季春贵不贵?", "a": "芝士奶盖四季春 16 元一杯。", "keywords": ["16"], "ood": False},
    {"intent": "price", "q": "加珍珠要加钱吗?", "a": "加珍珠加 2 元,其他小料椰果 2 元、芋泥 3 元、奶盖 4 元、布丁 3 元。", "keywords": ["2"], "ood": False},
    {"intent": "price", "q": "芋泥波波鲜奶卖多少?", "a": "芋泥波波鲜奶 17 元一杯。", "keywords": ["17"], "ood": False},
    {"intent": "price", "q": "黑糖珍珠鲜奶的价格?", "a": "黑糖珍珠鲜奶 15 元一杯。", "keywords": ["15"], "ood": False},
    {"intent": "price", "q": "柠檬气泡水多少钱?", "a": "柠檬气泡水 10 元一杯。", "keywords": ["10"], "ood": False},
    {"intent": "price", "q": "焦糖布丁奶茶呢?", "a": "焦糖布丁奶茶 14 元一杯。", "keywords": ["14"], "ood": False},
    {"intent": "price", "q": "茉莉绿茶多少钱?", "a": "茉莉绿茶 8 元一杯。", "keywords": ["8"], "ood": False},

    # ---------- 产品介绍/推荐 ----------
    {"intent": "recommend", "q": "有什么好喝的吗?", "a": "推荐杨枝甘露,芒果西柚配西米,酸甜清爽,是招牌。", "keywords": ["杨枝甘露"], "ood": False},
    {"intent": "recommend", "q": "想喝奶茶,推荐什么?", "a": "经典珍珠奶茶最受欢迎,12 元一杯;喜欢浓郁可以试黑糖珍珠鲜奶。", "keywords": ["经典珍珠奶茶", "黑糖珍珠鲜奶"], "ood": False},
    {"intent": "recommend", "q": "有没有不含茶的饮品?", "a": "柠檬气泡水和黑糖珍珠鲜奶都不含茶,可以放心选。", "keywords": ["柠檬气泡水", "黑糖珍珠鲜奶"], "ood": False},
    {"intent": "recommend", "q": "我想喝带芝士的,推荐?", "a": "芝士奶盖四季春 16 元,奶盖咸香,茶底清爽。", "keywords": ["芝士奶盖四季春"], "ood": False},
    {"intent": "recommend", "q": "哪个是招牌?", "a": "招牌是杨枝甘露和经典珍珠奶茶,点单最多。", "keywords": ["杨枝甘露", "经典珍珠奶茶"], "ood": False},
    {"intent": "recommend", "q": "有什么适合夏天的?", "a": "夏天推荐柠檬气泡水和杨枝甘露,清爽解暑。", "keywords": ["柠檬气泡水", "杨枝甘露"], "ood": False},
    {"intent": "recommend", "q": "喜欢芋泥,有什么?", "a": "芋泥波波鲜奶 17 元,芋泥绵密配鲜奶,很受欢迎。", "keywords": ["芋泥波波鲜奶"], "ood": False},
    {"intent": "recommend", "q": "不想喝甜的,有什么?", "a": "茉莉绿茶清爽不甜,还可以选无糖,8 元一杯。", "keywords": ["茉莉绿茶"], "ood": False},
    {"intent": "recommend", "q": "有给小朋友喝的吗?", "a": "柠檬气泡水和茉莉绿茶比较清淡,也可以做去冰少糖。", "keywords": ["柠檬气泡水", "茉莉绿茶"], "ood": False},

    # ---------- 口味定制 ----------
    {"intent": "customize", "q": "可以选糖度吗?", "a": "可以,糖度分正常、少糖、半糖、三分甜和无糖。", "keywords": ["无糖", "半糖"], "ood": False},
    {"intent": "customize", "q": "能去冰吗?", "a": "可以,冰量可选正常、少冰、去冰,部分饮品可以做热的。", "keywords": ["去冰", "少冰"], "ood": False},
    {"intent": "customize", "q": "珍珠奶茶能做无糖吗?", "a": "可以,任何奶茶都可以选无糖,珍珠本身带一点甜味。", "keywords": ["无糖"], "ood": False},
    {"intent": "customize", "q": "可以加奶盖吗?", "a": "可以,加奶盖加 4 元。", "keywords": ["4"], "ood": False},
    {"intent": "customize", "q": "能做成热饮吗?", "a": "部分饮品可以做热的,比如经典珍珠奶茶和芋泥波波鲜奶。", "keywords": ["经典珍珠奶茶", "芋泥波波鲜奶"], "ood": False},

    # ---------- 营业时间 ----------
    {"intent": "hours", "q": "你们几点开门?", "a": "我们早上 10:00 开门营业。", "keywords": ["10:00", "10"], "ood": False},
    {"intent": "hours", "q": "营业到几点?", "a": "营业到晚上 22:00,外卖 21:30 截止。", "keywords": ["22:00", "21:30"], "ood": False},
    {"intent": "hours", "q": "现在这个点还营业吗?", "a": "我们营业时间是 10:00-22:00,您看下当前时间。", "keywords": ["10:00", "22:00"], "ood": False},
    {"intent": "hours", "q": "周末营业吗?", "a": "周末和节假日正常营业,10:00-22:00。", "keywords": ["10:00", "22:00"], "ood": False},

    # ---------- 外卖 ----------
    {"intent": "delivery", "q": "可以点外卖吗?", "a": "可以,3 公里内配送,满 25 元免配送费,不满收 3 元配送费。", "keywords": ["3", "25"], "ood": False},
    {"intent": "delivery", "q": "配送费多少钱?", "a": "配送费 3 元,满 25 元免配送费。", "keywords": ["3", "25"], "ood": False},
    {"intent": "delivery", "q": "外卖几点截止?", "a": "外卖 21:30 截止下单。", "keywords": ["21:30"], "ood": False},
    {"intent": "delivery", "q": "配送范围多远?", "a": "门店 3 公里内配送。", "keywords": ["3"], "ood": False},

    # ---------- 会员/优惠 ----------
    {"intent": "member", "q": "怎么积分?", "a": "消费 1 元积 1 分,100 积分可以抵 5 元。", "keywords": ["1", "100", "5"], "ood": False},
    {"intent": "member", "q": "积分有什么用?", "a": "积分可以在结账时抵扣,100 积分抵 5 元。", "keywords": ["100", "5"], "ood": False},
    {"intent": "member", "q": "有优惠券吗?", "a": "新客注册立减 3 元,每周三第二杯半价。", "keywords": ["3", "半价"], "ood": False},
    {"intent": "member", "q": "怎么成为会员?", "a": "在小程序注册即可成为会员,新客立减 3 元。", "keywords": ["小程序", "3"], "ood": False},
    {"intent": "member", "q": "周三有什么活动?", "a": "每周三第二杯半价。", "keywords": ["半价"], "ood": False},

    # ---------- 售后 ----------
    {"intent": "after_sale", "q": "不好喝可以换吗?", "a": "出品 30 分钟内口味不满意,可以免费重做或退款。", "keywords": ["30", "重做", "退款"], "ood": False},
    {"intent": "after_sale", "q": "外卖洒了怎么办?", "a": "外卖洒漏或破损,拍照联系客服可退款。", "keywords": ["拍照", "退款"], "ood": False},
    {"intent": "after_sale", "q": "能退款吗?", "a": "30 分钟内口味不满意可退,外卖破损可退。", "keywords": ["30", "退款"], "ood": False},

    # ---------- 过敏原 ----------
    {"intent": "allergen", "q": "杨枝甘露里有芒果吗?", "a": "有的,杨枝甘露含芒果,对芒果过敏的话不建议饮用。", "keywords": ["芒果"], "ood": False},
    {"intent": "allergen", "q": "有含牛奶的饮品吗?", "a": "黑糖珍珠鲜奶、芋泥波波鲜奶、芝士奶盖都含乳制品。", "keywords": ["黑糖珍珠鲜奶", "芋泥波波鲜奶"], "ood": False},
    {"intent": "allergen", "q": "哪款不含奶?", "a": "茉莉绿茶、柠檬气泡水、杨枝甘露不含牛奶。", "keywords": ["茉莉绿茶", "柠檬气泡水"], "ood": False},

    # ---------- 门店 ----------
    {"intent": "store", "q": "附近有门店吗?", "a": "全国有 200 多家门店,可以在小程序里查看最近的门店。", "keywords": ["200", "小程序"], "ood": False},
    {"intent": "store", "q": "有 WiFi 吗?", "a": "门店提供免费 WiFi。", "keywords": ["WiFi"], "ood": False},
    {"intent": "store", "q": "你们是什么品牌?", "a": "茶语时光 2019 年创立,主打新鲜现做茶饮,全国 200 多家门店。", "keywords": ["2019", "200"], "ood": False},
    {"intent": "store", "q": "怎么找最近的门店?", "a": "打开茶语时光小程序,可以定位并查看最近门店。", "keywords": ["小程序"], "ood": False},

    # ---------- 知识库外问题(应拒答) ----------
    {"intent": "ood", "q": "你们能修手机吗?", "a": "抱歉,我是茶语时光的客服,只负责店内产品和服务相关问题,这个问题我无法回答。", "keywords": ["抱歉", "无法回答"], "ood": True},
    {"intent": "ood", "q": "推荐一只股票?", "a": "抱歉,这个问题超出了我的服务范围,我是茶语时光客服,只解答店内业务。", "keywords": ["抱歉", "服务范围"], "ood": True},
    {"intent": "ood", "q": "帮我查一下明天北京的天气", "a": "抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。", "keywords": ["抱歉", "无法回答"], "ood": True},
    {"intent": "ood", "q": "附近哪里可以租房?", "a": "不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。", "keywords": ["不好意思", "服务范围"], "ood": True},
    {"intent": "ood", "q": "你们招人吗?", "a": "抱歉,我是茶语时光的客服,招聘问题请联系我们的官方招聘渠道。", "keywords": ["抱歉", "招聘"], "ood": True},
    {"intent": "ood", "q": "帮我写一篇论文", "a": "抱歉,这个问题超出了我的服务范围,我是茶语时光客服,只解答店内业务。", "keywords": ["抱歉", "服务范围"], "ood": True},
    {"intent": "ood", "q": "怎么炒股?", "a": "抱歉,我是茶语时光的客服,只负责门店产品和服务相关问题,这个问题我无法回答。", "keywords": ["抱歉", "无法回答"], "ood": True},
    {"intent": "ood", "q": "你们家能充话费吗?", "a": "抱歉,我是茶语时光的客服,只负责店内产品和服务相关问题,这个问题我无法回答。", "keywords": ["抱歉", "无法回答"], "ood": True},
]
```
