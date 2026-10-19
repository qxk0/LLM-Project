"""构建多轮工具客服 episode 与 SFT 轨迹。

产物:
  data/multiturn/episodes_train.jsonl       train horizon 1/2/4
  data/multiturn/episodes_train_all.jsonl   train horizon 1/2/4/8(用于课程)
  data/multiturn/episodes_val.jsonl         val horizon 2/4/6
  data/multiturn/episodes_test.jsonl        test horizon 2/4/6/8
  data/multiturn/sft_*.jsonl                脚本化 teacher 生成的成功轨迹

切分原则:
- test_* / val_* / train_* 是不同的 scenario family,family 不跨 split;
- train 最大 horizon=4,test 含 6/8,用于测 horizon 外推;
- teacher 轨迹全部通过环境终局判定,保证 SFT 数据可解。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl.multiturn.env import (  # noqa: E402
    episode_to_dict,
    make_episode,
    run_teacher,
)


TRAIN_SCENARIOS = [
    "train_faq_coupon",
    "train_pickup_remake",
    "train_delivery_spill",
    "train_ood",
]
TRAIN_ALL_SCENARIOS = TRAIN_SCENARIOS + ["train_multi_issue"]
VAL_SCENARIOS = ["val_late_delivery", "val_pickup_remake"]
TEST_SCENARIOS = [
    "test_faq_coupon",
    "test_pickup_remake",
    "test_delivery_no_photo",
    "test_multi_issue",
]


def parse_args():
    parser = argparse.ArgumentParser(description="构建多轮客服 RL 数据")
    parser.add_argument("--out-dir", type=str, default="data/multiturn")
    parser.add_argument("--per-scenario", type=int, default=30, help="train 每个 scenario 的 episode 数")
    parser.add_argument("--val-per-scenario", type=int, default=10)
    parser.add_argument("--test-per-scenario", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="极小数据,用于本地跑通")
    return parser.parse_args()


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_split(
    scenarios: List[str],
    count: int,
    seed: int,
    index_start: int,
):
    episodes = []
    sft_rows = []
    idx = index_start
    for scenario in scenarios:
        for _ in range(count):
            episode = make_episode(scenario, idx, seed=seed)
            messages, env = run_teacher(episode)
            episodes.append(episode_to_dict(episode))
            sft_rows.append(
                {
                    "episode_id": episode.episode_id,
                    "scenario": episode.scenario,
                    "family_id": episode.family_id,
                    "horizon": episode.horizon,
                    "messages": messages,
                    "success": env.success,
                    "turns": env.turns,
                }
            )
            idx += 1
    return episodes, sft_rows, idx


def summarize(name: str, episodes: List[Dict[str, Any]], sft_rows: List[Dict[str, Any]]):
    return {
        "split": name,
        "episodes": len(episodes),
        "sft_rows": len(sft_rows),
        "horizons": dict(sorted(Counter(e["horizon"] for e in episodes).items())),
        "scenarios": dict(sorted(Counter(e["scenario"] for e in episodes).items())),
        "avg_turns": (
            sum(e["horizon"] for e in episodes) / len(episodes) if episodes else 0.0
        ),
    }


def main():
    args = parse_args()
    if args.smoke_test:
        args.per_scenario = min(args.per_scenario, 2)
        args.val_per_scenario = min(args.val_per_scenario, 1)
        args.test_per_scenario = min(args.test_per_scenario, 1)

    os.makedirs(args.out_dir, exist_ok=True)
    idx = 1000
    train_eps, train_sft, idx = build_split(
        TRAIN_SCENARIOS, args.per_scenario, args.seed, idx
    )
    train_all_eps, train_all_sft, idx = build_split(
        TRAIN_ALL_SCENARIOS, args.per_scenario, args.seed, idx
    )
    val_eps, val_sft, idx = build_split(
        VAL_SCENARIOS, args.val_per_scenario, args.seed, idx
    )
    test_eps, test_sft, idx = build_split(
        TEST_SCENARIOS, args.test_per_scenario, args.seed, idx
    )

    families = {
        "train": {e["family_id"] for e in train_eps},
        "train_all": {e["family_id"] for e in train_all_eps},
        "val": {e["family_id"] for e in val_eps},
        "test": {e["family_id"] for e in test_eps},
    }
    if families["val"] & families["test"] or families["train"] & families["test"]:
        raise AssertionError("scenario family leaked across splits")

    write_jsonl(os.path.join(args.out_dir, "episodes_train.jsonl"), train_eps)
    write_jsonl(os.path.join(args.out_dir, "episodes_train_all.jsonl"), train_all_eps)
    write_jsonl(os.path.join(args.out_dir, "episodes_val.jsonl"), val_eps)
    write_jsonl(os.path.join(args.out_dir, "episodes_test.jsonl"), test_eps)
    write_jsonl(os.path.join(args.out_dir, "sft_train.jsonl"), train_sft)
    write_jsonl(os.path.join(args.out_dir, "sft_train_all.jsonl"), train_all_sft)
    write_jsonl(os.path.join(args.out_dir, "sft_val.jsonl"), val_sft)
    write_jsonl(os.path.join(args.out_dir, "sft_test.jsonl"), test_sft)

    stats = {
        "seed": args.seed,
        "smoke_test": args.smoke_test,
        "splits": {
            "train": summarize("train", train_eps, train_sft),
            "train_all": summarize("train_all", train_all_eps, train_all_sft),
            "val": summarize("val", val_eps, val_sft),
            "test": summarize("test", test_eps, test_sft),
        },
        "family_overlap": {
            "train_test": len(families["train"] & families["test"]),
            "val_test": len(families["val"] & families["test"]),
        },
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    readme = """# 多轮工具客服数据

由 `data_eng/build_multi_turn.py` 生成,所有 SFT 轨迹都由确定性 teacher 跑通环境后保存。

- `episodes_train.jsonl`: 训练 horizon 1/2/4
- `episodes_train_all.jsonl`: 课程训练,horizon 1/2/4/8
- `episodes_val.jsonl`: horizon 2/4/6
- `episodes_test.jsonl`: horizon 2/4/6/8(6/8 为训练未见 horizon)
- `sft_*.jsonl`: 多轮 SFT 轨迹,assistant 只包含 JSON 动作和最终回答

重建:
```powershell
python data_eng/build_multi_turn.py --smoke-test
python data_eng/build_multi_turn.py
```
"""
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

