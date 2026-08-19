"""领域后训练数据构建管道(面试核心亮点)。

流程:
  1. 种子:人工基于知识库编写高质量问答(seeds.py,含拒答样本)
  2. 扩展:模板 + 实体替换,把种子扩展到几十倍(问题多样性的来源)
  3. 清洗:规范化、精确去重、长度过滤
  4. 切分:按意图分层划分 train/val/test(评测集与训练集严格分离)
  5. 统计:输出数据报告(stats.json),所有数字可直接写进简历

输出:data_eng/output/ 下的 train.jsonl / val.jsonl / test.jsonl
"""

import json
import os
import random
import re
from collections import Counter

from kb import PRODUCTS, REFUSAL_TEMPLATES, TOPPINGS
from seeds import SEEDS

OUT_DIR = "data_eng/output"
RANDOM_SEED = 42

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


def expand_seeds():
    """种子 + 模板扩展,返回 (样本列表, 来源统计)。"""
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


def clean_and_split(samples):
    """去重 + 过滤 + 按意图分层切分。"""
    seen = set()
    unique = []
    dropped = 0
    for s in samples:
        qn = normalize(s["q"])
        if qn in seen:
            dropped += 1
            continue
        if not (4 <= len(s["q"]) <= 80) or not (5 <= len(s["a"]) <= 200):
            dropped += 1
            continue
        seen.add(qn)
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


def main():
    parser = argparse.ArgumentParser(description="构建领域后训练数据")
    parser.add_argument("--extra-file", type=str, default=None,
                        help="额外的数据文件(如 evolve.py 蒸馏产物),合并后统一清洗切分")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    samples, source = expand_seeds()
    if args.extra_file:
        with open(args.extra_file, encoding="utf-8") as f:
            extra = [json.loads(line) for line in f if line.strip()]
        samples.extend(extra)
        source["evolved"] = len(extra)
        print(f"合并额外数据 {len(extra)} 条: {args.extra_file}")
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


if __name__ == "__main__":
    main()
