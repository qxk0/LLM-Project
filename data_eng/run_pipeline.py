"""按项目真实依赖顺序重跑全流程;所有产物写入 --root,**不覆盖任何现有文件**。

为什么是这个顺序(SFT → RFT → 合并 → SFT → DPO/RL):
  1. `build_dataset`(不带额外数据)产出"纯 gold"数据 v1;
  2. 用 v1 训一个 SFT(sft_v1),它代表"当前 policy";
  3. `rft.py` 从 sft_v1 对 v1 的 train 做 on-policy 采样 + 规则验证器筛选(拒绝采样);
  4. `build_dataset` 把 evolved(Self-Instruct)与 onpolicy(RFT)合并进 gold,做血缘/group 切分
     → clean 数据集(test 仍只含 gold,零泄漏);
  5. 用 clean train 训最终 SFT;
  6. DPO 与 RL(DAPO)都从**最终 SFT** 出发(两个并列分支,不是串行链)。

每步的命令 / 起止时间 / 退出码 / 耗时 / 关键产物都写入 <root>/manifest.jsonl 与 MANIFEST.md。

用法:
  python data_eng/run_pipeline.py --root runs/2026-10-12 --only evolve
  python data_eng/run_pipeline.py --root runs/2026-10-12 --from sft_v1
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_stages(root, input_root=None):
    """返回 {stage_name: [ [cmd...], ... ]}。路径全部指向 root,不触碰既有产物。

    input_root 只给末尾的 GSPO 消融阶段用:那三个阶段的**输入**(最终 SFT 适配器 +
    clean 数据集)来自上一轮已经跑完的 run,产物仍然写进 root,互不覆盖。
    """
    py = sys.executable
    inp = os.path.abspath(input_root) if input_root else root
    inp_models = os.path.join(inp, "models")
    inp_clean = os.path.join(inp, "data", "clean")
    data = os.path.join(root, "data")
    v1 = os.path.join(data, "v1")
    clean = os.path.join(data, "clean")
    models = os.path.join(root, "models")
    evald = os.path.join(root, "eval")
    preds = os.path.join(evald, "predictions")
    judge = os.path.join(root, "judge")
    evolved = os.path.join(data, "evolved.jsonl")
    onpolicy = os.path.join(data, "onpolicy.jsonl")
    rejected = os.path.join(data, "onpolicy_rejected.jsonl")
    pairs = os.path.join(data, "dpo_pairs.jsonl")

    return {
        # 1) 纯 gold 数据集(v1)
        "data_v1": [[
            py, "data_eng/build_dataset.py",
            "--out-dir", v1,
            "--reuse-split-from", "",          # 强制全新 group 切分(不冻结旧 split)
        ]],
        # 2) Self-Instruct 蒸馏(调智谱 GLM API)
        "evolve": [[
            py, "data_eng/evolve.py",
            "--seeds", "30", "--variants", "3", "--model", "glm-4-flash",
            "--out", evolved,
        ]],
        # 3) SFT v1(用于 RFT 的 on-policy 采样)
        "sft_v1": [[
            py, "sft/train_sft.py",
            "--data-file", os.path.join(v1, "train.jsonl"),
            "--max-steps", "600", "--max-examples", "2000", "--seed", "42",
            "--output-dir", os.path.join(models, "sft_v1"),
        ]],
        # 4) RFT:从 sft_v1 采样 + 验证器筛选(含 rejected 与通过率落盘)
        "rft": [[
            py, "rl/rft.py",
            "--adapter", os.path.join(models, "sft_v1"),
            "--data-file", os.path.join(v1, "train.jsonl"),
            "--max-rows", "400", "--num-samples", "4", "--min-score", "1.0",
            "--batch-size", "8", "--seed", "42",
            "--out", onpolicy, "--rejected-out", rejected,
        ]],
        # 5) 合并 evolved + onpolicy → clean 数据集(血缘 + group 切分 + leakage audit)
        "data_clean": [[
            py, "data_eng/build_dataset.py",
            "--extra-file", evolved,
            "--extra-file", onpolicy,
            "--out-dir", clean,
            "--reuse-split-from", "",
        ]],
        # 6) 最终 SFT(用 clean train)
        "sft": [[
            py, "sft/train_sft.py",
            "--data-file", os.path.join(clean, "train.jsonl"),
            "--max-steps", "600", "--max-examples", "2000", "--seed", "42",
            "--output-dir", os.path.join(models, "sft"),
        ]],
        # 7) DPO(从最终 SFT 出发;偏好对落盘)
        "dpo": [[
            py, "rl/train_dpo.py",
            "--adapter", os.path.join(models, "sft"),
            "--data-file", os.path.join(clean, "train.jsonl"),
            "--max-pairs", "400", "--num-samples", "4", "--pairs-batch-size", "8",
            "--max-steps", "200", "--seed", "42", "--pairs-out", pairs,
            "--output-dir", os.path.join(models, "dpo"),
        ]],
        # 8) RL / DAPO(从最终 SFT 出发)
        "rl": [[
            py, "rl/train_grpo.py",
            "--adapter", os.path.join(models, "sft"),
            "--data-file", os.path.join(clean, "train.jsonl"),
            "--mode", "domain", "--max-steps", "300", "--seed", "42",
            "--output-dir", os.path.join(models, "rl"),
        ]],
        # 9) 评测:SFT/base 用 chat(NFT 训练格式),DPO/RL 用 raw(它们自己的训练格式)
        "eval_chat": [[
            py, "sft/eval_domain.py",
            "--data-file", os.path.join(clean, "test.jsonl"),
            "--adapters", os.path.join(models, "sft"),
            "--prompt-format", "chat",
            "--predictions-dir", preds,
            "--report", os.path.join(evald, "eval_chat.md"),
        ]],
        "eval_raw": [[
            py, "sft/eval_domain.py",
            "--data-file", os.path.join(clean, "test.jsonl"),
            # checkpooint-100=DPO 早停点(对应旧矩阵里的"DPO 10 步");checkpoint-200=DPO 最终点。
            # 不再单列 models/dpo —— 已用 sha256 验证它与 checkpoint-200 的 adapter 完全一致,重复评测无信息量。
            "--adapters", os.path.join(models, "dpo", "checkpoint-100"),
            os.path.join(models, "dpo", "checkpoint-200"),
            os.path.join(models, "rl"),
            "--prompt-format", "raw",
            "--predictions-dir", preds,
            "--report", os.path.join(evald, "eval_raw.md"),
        ]],
        # 10) LLM-as-Judge(复用已有 predictions)
        "judge": [[
            py, "data_eng/run_llm_judge.py",
            "--predictions-dir", preds, "--out-dir", judge,
            "--judge-model", "glm-4-flash",
        ]],
        # ---------- GSPO 消融(唯一变量 = --seq-level) ----------
        # 与 DAPO 那轮相比,只有 loss 配置不同:
        #   DAPO : loss_type=dapo + importance_sampling_level=token
        #   GSPO : loss_type=grpo + importance_sampling_level=sequence(TRL 官方文档指定的 GSPO 复现配置)
        # 其余(起始适配器 / 数据 / 步数 / seed / batch / lr / temperature / beta)完全一致。
        "gspo": [[
            py, "rl/train_grpo.py",
            "--adapter", os.path.join(inp_models, "sft"),
            "--data-file", os.path.join(inp_clean, "train.jsonl"),
            "--mode", "domain", "--max-steps", "300", "--seed", "42",
            "--seq-level",
            "--output-dir", os.path.join(models, "grpo_seqlevel"),
        ]],
        # 与 DAPO 用同一份 clean test、同一 evaluator、同一 raw prompt 口径
        "eval_gspo": [[
            py, "sft/eval_domain.py",
            "--data-file", os.path.join(inp_clean, "test.jsonl"),
            "--adapters", os.path.join(models, "grpo_seqlevel"),
            "--prompt-format", "raw",
            "--predictions-dir", preds,
            "--report", os.path.join(evald, "eval_gspo.md"),
        ]],
        "judge_gspo": [[
            py, "data_eng/run_llm_judge.py",
            "--predictions-dir", preds, "--out-dir", judge,
            "--judge-model", "glm-4-flash",
        ]],
    }


ORDER = ["data_v1", "evolve", "sft_v1", "rft", "data_clean", "sft", "dpo", "rl",
         "eval_chat", "eval_raw", "judge",
         "gspo", "eval_gspo", "judge_gspo"]


def sha256_head(path, n=16):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def snapshot_outputs(root, stage):
    """记录该阶段关键产物的 hash(用于审计"是否真的产出了东西")。"""
    out = {}
    for sub in ("data", "models", "eval", "judge"):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            if stage not in dirpath and not dirpath.startswith(os.path.join(root, sub)):
                continue
            for fn in files:
                if fn.endswith((".jsonl", ".json", ".md", ".pt", ".safetensors")) and "manifest" not in fn:
                    p = os.path.join(dirpath, fn)
                    rel = os.path.relpath(p, root)
                    out[rel] = sha256_head(p)
    return out


def main():
    ap = argparse.ArgumentParser(description="按项目顺序重跑全流程(产物写入独立 root)")
    ap.add_argument("--root", type=str, default="runs/2026-10-12")
    ap.add_argument("--input-root", type=str, default=None,
                    help="GSPO 消融阶段的输入来源 run(默认与 --root 相同)")
    ap.add_argument("--only", action="append", default=[], help="只跑指定阶段(可多次)")
    ap.add_argument("--from", dest="from_stage", type=str, default=None, help="从该阶段开始跑")
    ap.add_argument("--list", action="store_true", help="只列出阶段")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    stages = build_stages(root, args.input_root)
    if args.list:
        for i, s in enumerate(ORDER, 1):
            print(f"{i:2d}. {s}")
        return 0

    todo = list(ORDER)
    if args.from_stage:
        todo = todo[todo.index(args.from_stage):]
    if args.only:
        todo = [s for s in todo if s in args.only]

    for sub in ("logs", "data", "models", "eval", "judge"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    manifest_path = os.path.join(root, "manifest.jsonl")

    print(f"[pipeline] root = {root}")
    print(f"[pipeline] 计划执行:{todo}")
    for stage in todo:
        log_path = os.path.join(root, "logs", f"{stage}.log")
        print(f"\n[pipeline] === {stage} === -> {log_path}")
        t0 = time.time()
        rc = 0
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n\n===== {stage} @ {datetime.now().isoformat(timespec='seconds')} =====\n")
            lf.flush()
            for cmd in stages[stage]:
                lf.write("CMD: " + " ".join(cmd) + "\n")
                lf.flush()
                rc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    break
        entry = {
            "stage": stage,
            "start": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            "duration_sec": round(time.time() - t0, 1),
            "returncode": rc,
            "log": os.path.relpath(log_path, root),
            "outputs": snapshot_outputs(root, stage),
        }
        with open(manifest_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[pipeline] {stage} rc={rc} 用时 {entry['duration_sec']}s")
        if rc != 0:
            print(f"[pipeline] 阶段 {stage} 失败,已停止;日志:{log_path}")
            return rc

    # 生成可读清单
    lines = ["# 全流程重跑 MANIFEST", "", f"- root:`{root}`", ""]
    with open(manifest_path, encoding="utf-8") as mf:
        for line in mf:
            e = json.loads(line)
            lines.append(f"## {e['stage']}")
            lines.append(f"- 时间:{e['start']} | 用时 {e['duration_sec']}s | 退出码 {e['returncode']}")
            lines.append(f"- 日志:`{e['log']}`")
            if e["outputs"]:
                lines.append("- 产物:")
                for k, v in list(e["outputs"].items())[:12]:
                    lines.append(f"  - `{k}` (sha256:{v})")
            lines.append("")
    with open(os.path.join(root, "MANIFEST.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[pipeline] 全部完成,清单:{os.path.join(root, 'MANIFEST.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
