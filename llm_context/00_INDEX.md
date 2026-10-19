# LLM-Project 代码上下文包（用于上传网页版 LLM）

生成时间：2026-10-14 17:27:04
项目根目录：`D:\LLM-Project`

## 包含哪些文件

| 文件 | 内容 | 源文件数 | 估算 tokens | 体积 |
| --- | --- | ---: | ---: | ---: |
| `10_pretrain.md` | 预训练模块 pretrain/ (SwiGLU + RMSNorm + FlashAttention) | 8 | ~9,853 | 39 KB |
| `20_sft.md` | 监督微调模块 sft/ (SFT + 评测) | 7 | ~20,183 | 78 KB |
| `30_rl.md` | 对齐模块 rl/ (DPO / GRPO / RFT) | 6 | ~12,346 | 48 KB |
| `40_data_eng.md` | 数据工程模块 data_eng/ (数据集构建 + 质检 + 评审) | 8 | ~45,996 | 173 KB |
| `50_root.md` | 顶层文件 (requirements / 启动脚本) | 2 | ~143 | 1 KB |
| `60_docs_core.md` | 核心文档 (README/面试地图) | 4 | ~15,973 | 56 KB |
| **合计** | | 35 | **~104,494** | 393 KB |

## 已排除的内容（重要）

- `.venv/` —— Python 虚拟环境（第三方库，几万个文件）
- `models/` —— 模型权重（5.6GB，.safetensors / .pt）
- `runs/` —— 训练产物与 checkpoint（1.6GB）
- `data/` —— 训练数据集（43MB）
- `data_eng/output/`、`output_clean/` —— 数据工程中间产物
- `.env` —— **密钥文件，绝不外传**
- `*.docx` —— **简历，含个人信息，绝不外传**
- `__pycache__/`、`.git/`、`.workbuddy/`

## 上传建议

1. 单文件 token 超过 ~60k 的，优先只传你当前要讨论的那个模块；
2. 推荐用 ChatGPT **Projects** 功能建一个「LLM-Project」项目，把这些文件一次性加进项目知识库；
3. 首次提问时附上 `README.md` 内容 + 你的具体问题，不要问「帮我看看整个项目」。

## 目录树（自研代码部分）

```text
pretrain/
  build_data.py
  config.py
  eval_gen.py
  model.py
  plot_results.py
  sample.py
  smoke_test.py
  train.py
sft/
  eval_bench.py
  eval_breakdown.py
  eval_domain.py
  eval_stratified.py
  infer_sft.py
  prompt_ablation.py
  train_sft.py
rl/
  eval_rl.py
  model_utils.py
  rft.py
  train_dpo.py
  train_grpo.py
  train_grpo_standard.py
data_eng/
  GRPO_EXPERIMENT_ARCHIVE.md
  INTERVIEW_FACTS.md
  PROJECT_FINAL_AUDIT.md
  build_dataset.py
  evolve.py
  gold_eval_seeds.py
  kb.py
  leakage_audit.py
  run_llm_judge.py
  run_pipeline.py
  seeds.py
```
