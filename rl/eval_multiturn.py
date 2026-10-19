"""多轮客服评测:teacher baseline / SFT / RL adapter,按 horizon 分桶。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl.multiturn.env import Episode, episode_from_dict  # noqa: E402
from rl.multiturn.modeling import load_model_and_tokenizer  # noqa: E402
from rl.multiturn.rollout import (  # noqa: E402
    Trajectory,
    rollout_episode,
    teacher_trajectory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="多轮客服评测")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapters", nargs="*", default=[])
    parser.add_argument("--teacher", action="store_true", help="评测脚本化 teacher")
    parser.add_argument("--data-file", type=str, default="data/multiturn/episodes_test.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/multiturn_eval")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=640)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-horizon-limit", type=int, default=0,
                        help="每个 horizon 只评测 N 条;0=不限制")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--device-map", choices=["auto", "cuda"], default="cuda")
    return parser.parse_args()


def load_episodes(path: str, limit: int = 0, per_horizon_limit: int = 0) -> List[Episode]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(episode_from_dict(json.loads(line)))
            if limit and not per_horizon_limit and len(rows) >= limit:
                break
    if per_horizon_limit:
        kept = []
        counts: Dict[int, int] = defaultdict(int)
        for episode in rows:
            if counts[episode.horizon] < per_horizon_limit:
                kept.append(episode)
                counts[episode.horizon] += 1
        rows = kept
    elif limit:
        rows = rows[:limit]
    return rows


def early_refusal(traj: Trajectory, episode: Episode) -> bool:
    for turn in traj.turns:
        action = turn.parsed_action or {}
        if action.get("action") in {"answer", "refuse", "escalate"}:
            return action.get("action") == "refuse" and episode.expected_outcome != "refuse"
    return False


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["horizon"]].append(row)
    out = []
    for horizon in sorted(grouped):
        items = grouped[horizon]
        n = len(items)
        tool_calls = sum(i["tool_calls"] for i in items)
        tool_correct = sum(i["tool_correct"] for i in items)
        total_tokens = sum(i["total_tokens"] for i in items)
        successes = sum(1 for i in items if i["success"])
        out.append(
            {
                "horizon": horizon,
                "n": n,
                "success": successes / n if n else 0.0,
                "avg_turns": sum(i["turns"] for i in items) / n if n else 0.0,
                "tool_accuracy": tool_correct / tool_calls if tool_calls else 0.0,
                "avg_tokens": total_tokens / n if n else 0.0,
                "success_per_1k_tokens": successes / (total_tokens / 1000) if total_tokens else 0.0,
                "early_refusal": sum(1 for i in items if i["early_refusal"]) / n if n else 0.0,
            }
        )
    return out


def write_report(path: str, policy_rows: Dict[str, List[Dict[str, Any]]]):
    lines = ["# 多轮客服评测\n"]
    for policy, rows in policy_rows.items():
        lines.append(f"## {policy}\n")
        lines.append("| horizon | n | success | avg turns | tool acc | avg tokens | success/1k tok | early refusal |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in aggregate(rows):
            lines.append(
                f"| {item['horizon']} | {item['n']} | {item['success']:.1%} | "
                f"{item['avg_turns']:.2f} | {item['tool_accuracy']:.1%} | "
                f"{item['avg_tokens']:.1f} | {item['success_per_1k_tokens']:.2f} | "
                f"{item['early_refusal']:.1%} |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def result_row(policy: str, episode: Episode, traj: Trajectory) -> Dict[str, Any]:
    return {
        "policy": policy,
        "episode_id": episode.episode_id,
        "scenario": episode.scenario,
        "family_id": episode.family_id,
        "horizon": episode.horizon,
        "success": traj.success,
        "turns": len(traj.turns),
        "tool_calls": traj.tool_calls,
        "tool_correct": traj.tool_correct,
        "total_tokens": traj.total_tokens,
        "terminal_reward": traj.terminal_reward,
        "early_refusal": early_refusal(traj, episode),
        "transcript": traj.transcript,
    }


def main():
    args = parse_args()
    if args.smoke_test:
        args.limit = args.limit or 4
    if not args.teacher and not args.adapters:
        raise SystemExit("请至少指定 --teacher 或一个 --adapters")
    os.makedirs(args.output_dir, exist_ok=True)
    episodes = load_episodes(args.data_file, args.limit, args.per_horizon_limit)
    print(f"[data] episodes={len(episodes)}")

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    predictions = []
    if args.teacher:
        rows = []
        for episode in episodes:
            traj = teacher_trajectory(episode)
            row = result_row("teacher", episode, traj)
            rows.append(row)
            predictions.append(row)
        all_rows["teacher"] = rows
        print(f"[teacher] success={sum(r['success'] for r in rows) / len(rows):.1%}")

    for adapter in args.adapters:
        policy = os.path.basename(os.path.normpath(adapter)) or adapter
        model, tokenizer, _ = load_model_and_tokenizer(
            args.model_name,
            adapter=adapter,
            for_training=False,
            precision=args.precision,
            device_map=args.device_map,
        )
        rows = []
        for episode in episodes:
            traj = rollout_episode(
                model,
                tokenizer,
                episode,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                max_prompt_tokens=args.max_prompt_tokens,
                max_seq_len=args.max_seq_len,
                enable_thinking=args.enable_thinking,
            )
            row = result_row(policy, episode, traj)
            rows.append(row)
            predictions.append(row)
        all_rows[policy] = rows
        print(f"[{policy}] success={sum(r['success'] for r in rows) / len(rows):.1%}")
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(os.path.join(args.output_dir, "report.md"), all_rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: aggregate(v) for k, v in all_rows.items()}, f, ensure_ascii=False, indent=2)
    print(f"评测报告: {os.path.join(args.output_dir, 'report.md')}")


if __name__ == "__main__":
    main()

