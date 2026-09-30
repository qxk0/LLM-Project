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
    caps = {
        "train": max(total - clean_total, 1),  # train 吸收全部 non-clean group
        "test": max(clean_total * clean_test_ratio, 1),
        "val": max(clean_total * (1 - clean_test_ratio), 1),
    }

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
            allowed = ["val", "test"] if group_meta[gid]["clean"] else ["train"]
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
