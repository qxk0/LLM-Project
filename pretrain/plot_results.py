"""把实验日志画成图:训练/验证 loss 曲线、参数缩放曲线。

用法:
  python pretrain/plot_results.py --log models/exp_a/log.jsonl --log models/exp_b/log.jsonl
  python pretrain/plot_results.py --scaling models/*/log.jsonl
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def plot_curves(log_paths, out="models/pretrain/loss_curves.png"):
    plt.figure(figsize=(8, 5))
    for path in log_paths:
        events = load_events(path)
        steps = [e["step"] for e in events if e["type"] == "train"]
        losses = [e["loss"] for e in events if e["type"] == "train"]
        if not steps:
            continue
        label = os.path.basename(os.path.dirname(path))
        plt.plot(steps, losses, label=label, lw=1.5)
        val_steps = [e["step"] for e in events if e["type"] == "eval"]
        val_losses = [e["val_loss"] for e in events if e["type"] == "eval"]
        if val_steps:
            plt.plot(val_steps, val_losses, "o--", alpha=0.6)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.title("训练/验证 loss 曲线")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"曲线图已保存: {out}")


def plot_scaling(log_paths, out="models/pretrain/scaling.png"):
    """横轴参数量、纵轴最终验证 loss,看缩放规律。"""
    points = []
    for path in log_paths:
        events = load_events(path)
        meta = next((e for e in events if e["type"] == "meta"), None)
        evals = [e for e in events if e["type"] == "eval"]
        if not meta or not evals:
            continue
        points.append((meta["n_params"], evals[-1]["val_loss"], path))
    if not points:
        print("没有找到带 meta 的日志,跳过缩放图")
        return
    points.sort()
    plt.figure(figsize=(8, 5))
    plt.plot([p[0] / 1e6 for p in points], [p[1] for p in points], "o-")
    for x, y, p in points:
        label = os.path.basename(os.path.dirname(p))
        plt.annotate(label, (x / 1e6, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
    plt.xlabel("参数量 (M)")
    plt.ylabel("最终验证 loss")
    plt.title("缩放定律(小规模复现)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"缩放曲线已保存: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", default=[], help="日志文件,可多次指定")
    parser.add_argument("--scaling", action="store_true", help="画缩放曲线(需要多个实验)")
    parser.add_argument("--out", type=str, default="models/pretrain/loss_curves.png")
    args = parser.parse_args()

    if args.scaling:
        paths = sorted(glob.glob("models/exp_*/log.jsonl"))
        plot_scaling(paths, out=args.out.replace(".png", "_scaling.png"))
    elif args.log:
        plot_curves(args.log, out=args.out)
    else:
        parser.error("请用 --log 指定日志文件,或用 --scaling 画缩放曲线")


if __name__ == "__main__":
    main()
