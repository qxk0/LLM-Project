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
from seeds import SEEDS

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
                rows.append(
                    {
                        "intent": seed["intent"],
                        "q": q,
                        "a": a,
                        "keywords": seed["keywords"],
                        "ood": seed["ood"],
                        "source": "evolved",
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
