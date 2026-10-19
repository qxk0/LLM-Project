"""交互式客服测试。

两种模式:
  single: 单轮 SFT/DPO/RL 模型,直接输入自然语言问题;
  agent : 多轮工具 Agent,模型输出 JSON 动作,脚本自动执行工具和用户 simulator。

示例:
  python tools/chat_customer_service.py --mode single
  python tools/chat_customer_service.py --mode agent
  python tools/chat_customer_service.py --mode agent --question "我的外卖洒了还少了一件,怎么处理?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if os.path.join(REPO, "sft") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "sft"))

from eval_domain import (  # noqa: E402
    build_model as build_single_turn_model,
    build_prompt,
    generate_answer,
    resolve_stop_ids,
)
from rl.multiturn.env import make_episode  # noqa: E402
from rl.multiturn.modeling import load_model_and_tokenizer  # noqa: E402
from rl.multiturn.rollout import rollout_episode  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="茶语时光客服交互测试")
    parser.add_argument("--mode", choices=["single", "agent"], default="single")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--prompt-format", choices=["chat", "raw"], default="chat")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--question", type=str, default=None, help="只问一条;省略则进入交互循环")
    parser.add_argument("--scenario", choices=[
        "multi_issue", "delivery_no_photo", "pickup_remake",
        "faq_coupon", "ood", "late_delivery",
    ], default=None)
    return parser.parse_args()


def pick_scenario(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["修手机", "股票", "天气", "论文", "租房", "招人", "话费"]):
        return "ood"
    if any(k in q for k in ["做错", "口味", "重做"]):
        return "pickup_remake"
    if any(k in q for k in ["优惠", "新客", "券", "会员"]):
        return "faq_coupon"
    if any(k in q for k in ["晚", "超时", "迟到"]):
        return "late_delivery"
    if any(k in q for k in ["洒", "漏", "破损", "退款", "少"]):
        return "multi_issue"
    return "multi_issue"


def scenario_name(scenario: str) -> str:
    return {
        "multi_issue": "test_multi_issue",
        "delivery_no_photo": "test_delivery_no_photo",
        "pickup_remake": "test_pickup_remake",
        "faq_coupon": "test_faq_coupon",
        "ood": "train_ood",
        "late_delivery": "val_late_delivery",
    }[scenario]


def print_agent_transcript(traj, episode, question: str):
    print(f"\n你: {question}")
    for msg in traj.transcript:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            continue
        if role == "assistant":
            try:
                action = json.loads(content)
            except Exception:
                action = None
            if action and action.get("action") in {"answer", "refuse", "escalate"}:
                print(f"客服: {action.get('content', content)}")
            else:
                print(f"Agent: {content}")
        elif role == "tool":
            print(f"工具: {content}")
        elif role == "user":
            if content != question:
                print(f"用户: {content}")
    correct = traj.tool_correct / traj.tool_calls if traj.tool_calls else 1.0
    print(
        f"[结果] success={traj.success} | turns={len(traj.turns)} "
        f"| tool_acc={correct:.1%} | tokens={traj.total_tokens}"
    )


def run_single(args):
    adapter = args.adapter or os.path.join("runs", "2026-10-12", "models", "sft")
    model_name = args.model_name or "Qwen/Qwen2.5-0.5B"
    print(f"加载单轮模型: {model_name} + {adapter}")
    model = build_single_turn_model(model_name, adapter)
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = resolve_stop_ids(tokenizer)
    print("输入问题,:quit 退出。")
    while True:
        question = args.question or input("\n你: ").strip()
        if not question or question in {":quit", "quit", "exit"}:
            break
        result = generate_answer(
            model,
            tokenizer,
            question,
            max_new=args.max_new_tokens,
            prompt_format=args.prompt_format,
            stop_ids=stop_ids,
        )
        print(f"客服: {result['prediction']}")
        print(
            f"[tokens={result['generated_tokens']} eos={result['hit_eos']} "
            f"elapsed={result['elapsed_seconds']}s]"
        )
        if args.question:
            break


def run_agent(args):
    adapter = args.adapter or os.path.join("models", "multiturn_grpo_turn_h8_scaled")
    model_name = args.model_name or "Qwen/Qwen3-0.6B"
    print(f"加载多轮 Agent: {model_name} + {adapter}")
    model, tokenizer, _ = load_model_and_tokenizer(
        model_name,
        adapter=adapter,
        for_training=False,
        device_map="cuda",
    )
    print("输入客服问题,脚本会自动跑工具/用户 simulator,:quit 退出。")
    index = 2000
    while True:
        question = args.question or input("\n你: ").strip()
        if not question or question in {":quit", "quit", "exit"}:
            break
        scenario = args.scenario or pick_scenario(question)
        episode = make_episode(scenario_name(scenario), index)
        episode.initial_message = question
        traj = rollout_episode(
            model,
            tokenizer,
            episode,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            do_sample=False,
            enable_thinking=False,
        )
        print_agent_transcript(traj, episode, question)
        index += 1
        if args.question:
            break


def main():
    args = parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_agent(args)


if __name__ == "__main__":
    main()

