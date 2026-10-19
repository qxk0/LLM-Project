"""Qwen chat template、assistant mask 与多轮 tokenization。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def is_qwen3_tokenizer(tokenizer) -> bool:
    name = str(getattr(tokenizer, "name_or_path", "") or "").lower()
    template = str(getattr(tokenizer, "chat_template", "") or "")
    return "qwen3" in name or "enable_thinking" in template


def render_chat(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    """渲染 chat template。

    Qwen3 非 thinking 模式在 generation prompt 中会插入空的 <think></think>。
    训练完整序列时模板不会自动插入,这里显式补上,保证 SFT 与推理一致。
    """
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if is_qwen3_tokenizer(tokenizer):
        kwargs["enable_thinking"] = enable_thinking
    text = tokenizer.apply_chat_template(messages, **kwargs)
    if is_qwen3_tokenizer(tokenizer) and not enable_thinking and not add_generation_prompt:
        # 只给还没有 <think> 的 assistant 起始标记补空 think 块。
        text = re.sub(
            r"(<\|im_start\|>assistant\n)(?!<think>)",
            r"\1<think>\n\n</think>\n\n",
            text,
        )
    return text


def _find_subsequence(ids: List[int], sub: List[int], start: int = 0) -> int:
    if not sub:
        return -1
    end = len(ids) - len(sub)
    for i in range(start, end + 1):
        if ids[i:i + len(sub)] == sub:
            return i
    return -1


def tokenize_multiturn_example(
    tokenizer,
    messages: List[Dict[str, str]],
    max_len: int = 1024,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Tokenize 完整多轮对话,只对 assistant 内容 + 结束符计算 loss。

    Qwen 的 chat template 不支持 {% generation %} 标记,不能直接用
    return_assistant_tokens_mask;这里用 assistant 内容的 token 子序列定位。
    """
    text = render_chat(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len).input_ids
    mask = [0] * len(ids)

    end_ids = set()
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            end_ids.add(token_id)
    if tokenizer.eos_token_id is not None:
        end_ids.add(tokenizer.eos_token_id)

    search_from = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content_ids = tokenizer(
            msg.get("content", ""), add_special_tokens=False
        ).input_ids
        if not content_ids:
            continue
        pos = _find_subsequence(ids, content_ids, search_from)
        if pos < 0:
            # 序列被 max_len 截断时,后面的 assistant 轮次可能已经不存在。
            break
        end = pos + len(content_ids)
        mask[pos:end] = [1] * len(content_ids)
        if end < len(ids) and ids[end] in end_ids:
            mask[end] = 1
            search_from = end + 1
        else:
            search_from = end

    labels = [token_id if mask[i] else -100 for i, token_id in enumerate(ids)]
    assistant_tokens = sum(mask)
    if assistant_tokens == 0:
        raise ValueError("no assistant tokens found in tokenized conversation")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
        "assistant_tokens": assistant_tokens,
    }


def parse_action(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """兼容训练脚本的轻量导入。"""
    if not text or not text.strip():
        return None, "empty_action"
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        return None, "no_json_object"
    try:
        obj = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    if not isinstance(obj, dict):
        return None, "not_object"
    return obj, None


def format_action(action: Dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))

