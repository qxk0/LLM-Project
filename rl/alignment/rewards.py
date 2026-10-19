"""偏好对构造与 GRPO 共用的规则奖励。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, List

REFUSAL_MARKERS = ["抱歉", "不好意思", "无法回答", "超出", "服务范围", "无法处理"]


def repetition_penalty(text: str) -> float:
    """对明显复读做轻量惩罚,避免 best-of-N 选中复读文本。"""
    text = re.sub(r"\s+", "", text)
    if len(text) < 8:
        return 0.0
    grams = [text[i:i + 4] for i in range(len(text) - 3)]
    if not grams:
        return 0.0
    max_count = max(Counter(grams).values())
    return min(0.8, max(0, max_count - 2) * 0.2)


def score_response(completion: str, keywords: List[str], is_ood: bool) -> float:
    """与评测口径一致的标量奖励。

    非 OOD: 任一关键词命中 +1,否则 -1;
    OOD: 出现拒答词 +1,否则 -1;
    简洁 <=80 字符 +0.1;复读扣分。
    """
    text = (completion or "").strip()
    if not text:
        return -1.2
    if is_ood:
        score = 1.0 if any(m in text for m in REFUSAL_MARKERS) else -1.0
    else:
        score = 1.0 if any(k in text for k in keywords) else -1.0
    if len(text) <= 80:
        score += 0.1
    score -= repetition_penalty(text)
    return score


def domain_reward(
    prompts: List[str],
    completions: List[str],
    answer_keywords: List[List[str]],
    ood: List[bool],
    **kwargs: Any,
) -> List[float]:
    return [
        score_response(comp, keywords, bool(is_ood))
        for comp, keywords, is_ood in zip(completions, answer_keywords, ood)
    ]

