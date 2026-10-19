"""多轮工具客服 GRPO/REINFORCE 原型。

核心对比:
  --reward-mode terminal : 只使用终局 reward,所有 turn 共享同一个 advantage
  --reward-mode turn     : 终端 reward + 每轮 process reward 的 return-to-go

训练方式:
1. 同一 episode 采样 G 条轨迹;
2. 按 turn 做组内归一化,得到 turn-level advantage;
3. 对每个 assistant turn 的 token 计算 CE,乘 advantage 后反传;
4. 每个 episode 只做一次 optimizer step,避免跨 turn 保留计算图。

这是便于 4GB 显存跑通的教学/研究原型,不是完整 DAPO recipe。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl.multiturn.env import Episode, episode_from_dict  # noqa: E402
from rl.multiturn.modeling import load_model_and_tokenizer  # noqa: E402
from rl.multiturn.rollout import Trajectory, rollout_episode  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="多轮工具客服 GRPO")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", type=str, default="models/sft_multiturn")
    parser.add_argument("--data-file", type=str, default="data/multiturn/episodes_train.jsonl")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reward-mode", choices=["terminal", "turn"], default="turn")
    parser.add_argument("--curriculum", type=str, default="",
                        help="例如 1,2,4,8;为空则不启用课程")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=640)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--process-scale", type=float, default=0.2,
                        help="turn 模式下的过程奖励缩放")
    parser.add_argument("--terminal-scale", type=float, default=2.0,
                        help="终局奖励缩放;建议 turn 模式用 2.0 防止过程奖励主导")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--device-map", choices=["auto", "cuda"], default="cuda")
    return parser.parse_args()


def apply_smoke_defaults(args):
    if not args.smoke_test:
        return args
    if args.max_steps == 10:
        args.max_steps = 2
    if args.num_generations == 4:
        args.num_generations = 2
    if args.max_new_tokens == 64:
        args.max_new_tokens = 48
    if args.max_episodes == 0:
        args.max_episodes = 8
    if args.save_steps == 5:
        args.save_steps = 1
    if args.output_dir is None:
        args.output_dir = f"models/multiturn_grpo_{args.reward_mode}_smoke"
    return args


def load_episodes(path: str, limit: int = 0) -> List[Episode]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(episode_from_dict(json.loads(line)))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no episodes in {path}")
    return rows


def compute_advantages(
    trajectories: List[Trajectory],
    mode: str,
    gamma: float,
    process_scale: float = 1.0,
    terminal_scale: float = 1.0,
):
    """turn-level return-to-go + 组内归一化。"""
    max_len = max(len(t.turns) for t in trajectories)
    by_turn: List[List[float]] = [[] for _ in range(max_len)]
    for traj in trajectories:
        t_len = len(traj.turns)
        if t_len == 0:
            traj.returns = []
            continue
        if mode == "terminal":
            returns = [traj.terminal_reward * terminal_scale] * t_len
        else:
            rewards = [turn.process_reward * process_scale for turn in traj.turns]
            rewards[-1] += traj.terminal_reward * terminal_scale
            returns = []
            running = 0.0
            for reward in reversed(rewards):
                running = reward + gamma * running
                returns.append(running)
            returns.reverse()
        traj.returns = returns
        for i, value in enumerate(returns):
            by_turn[i].append(value)

    stats = []
    zero_std_groups = 0
    for values in by_turn:
        mean = sum(values) / len(values) if values else 0.0
        if len(values) > 1:
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        else:
            std = 0.0
        if std < 1e-8:
            zero_std_groups += 1
        stats.append((mean, std))

    for traj in trajectories:
        traj.advantages = [
            (value - stats[i][0]) / (stats[i][1] + 1e-6)
            for i, value in enumerate(traj.returns)
        ]
    return zero_std_groups, max(1, len(by_turn))


def select_episode(
    episodes: List[Episode],
    step: int,
    max_steps: int,
    curriculum: List[int],
    rng: random.Random,
) -> Episode:
    if not curriculum:
        return rng.choice(episodes)
    stage = min(len(curriculum) - 1, int(step / max(1, max_steps) * len(curriculum)))
    horizon = curriculum[stage]
    candidates = [ep for ep in episodes if ep.horizon == horizon]
    return rng.choice(candidates if candidates else episodes)


def main():
    args = apply_smoke_defaults(parse_args())
    if args.output_dir is None:
        args.output_dir = f"models/multiturn_grpo_{args.reward_mode}"
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    episodes = load_episodes(args.data_file, args.max_episodes)
    curriculum = [int(x) for x in args.curriculum.split(",") if x.strip()]
    print(
        f"[data] episodes={len(episodes)} | horizons={dict(sorted(Counter(e.horizon for e in episodes).items()))} "
        f"| reward_mode={args.reward_mode} | curriculum={curriculum or 'off'}"
    )

    model, tokenizer, is_qwen3 = load_model_and_tokenizer(
        args.model_name,
        adapter=args.adapter,
        for_training=True,
        precision=args.precision,
        device_map=args.device_map,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; check adapter / LoRA")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    rng = random.Random(args.seed)
    logs: List[Dict[str, Any]] = []
    for step in range(1, args.max_steps + 1):
        episode = select_episode(episodes, step - 1, args.max_steps, curriculum, rng)
        trajectories = [
            rollout_episode(
                model,
                tokenizer,
                episode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                max_prompt_tokens=args.max_prompt_tokens,
                max_seq_len=args.max_seq_len,
                enable_thinking=args.enable_thinking,
            )
            for _ in range(args.num_generations)
        ]
        zero_std_groups, turn_groups = compute_advantages(
            trajectories,
            args.reward_mode,
            args.gamma,
            process_scale=args.process_scale,
            terminal_scale=args.terminal_scale,
        )

        model.train()
        model.config.use_cache = False
        optimizer.zero_grad()
        total_turns = sum(len(t.turns) for t in trajectories)
        total_loss = 0.0
        used_turns = 0
        for traj in trajectories:
            for t_idx, turn in enumerate(traj.turns):
                advantage = traj.advantages[t_idx] if t_idx < len(traj.advantages) else 0.0
                if abs(advantage) < 1e-8:
                    continue
                input_ids = torch.tensor(
                    [turn.prompt_ids + turn.action_ids],
                    dtype=torch.long,
                    device=model.device,
                )
                labels = torch.tensor(
                    [[-100] * len(turn.prompt_ids) + turn.action_ids],
                    dtype=torch.long,
                    device=model.device,
                )
                attention_mask = torch.ones_like(input_ids)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss * advantage / max(1, total_turns)
                loss.backward()
                total_loss += float(loss.detach().cpu())
                used_turns += 1

        if used_turns:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        success = sum(1 for t in trajectories if t.success) / len(trajectories)
        tool_calls = sum(t.tool_calls for t in trajectories)
        tool_correct = sum(t.tool_correct for t in trajectories)
        avg_turns = sum(len(t.turns) for t in trajectories) / len(trajectories)
        avg_return = sum(sum(t.returns) for t in trajectories) / len(trajectories)
        record = {
            "step": step,
            "episode_id": episode.episode_id,
            "horizon": episode.horizon,
            "reward_mode": args.reward_mode,
            "process_scale": args.process_scale,
            "terminal_scale": args.terminal_scale,
            "success": success,
            "avg_turns": avg_turns,
            "tool_accuracy": tool_correct / tool_calls if tool_calls else 0.0,
            "zero_std_fraction": zero_std_groups / turn_groups,
            "avg_return": avg_return,
            "loss": total_loss,
            "used_turns": used_turns,
        }
        logs.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if step % args.save_steps == 0 and step < args.max_steps:
            model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{step}"))

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"多轮 RL 完成: {args.output_dir}")


if __name__ == "__main__":
    main()

