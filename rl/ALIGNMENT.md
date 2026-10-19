# DPO / SimPO / GRPO 对齐对比

这一组实验固定:

- 同一个 clean SFT:`runs/2026-10-12/models/sft`
- 同一份 clean train:`runs/2026-10-12/data/clean/train.jsonl`
- 同一份 clean test(168 条)
- 同一个 chat prompt template
- 同一个 evaluation script

## 数据

DPO 和 SimPO 共用 `best-of-N` 偏好对:

- 从 clean train 分层抽 400 个 prompt;
- SFT 每题采样 4 个回答;
- `chosen = 规则分最高的采样回答`;
- `rejected = 规则分最低的采样回答`;
- 只保留 `chosen_score > rejected_score` 的对;
- 规则分包含关键词命中/拒答、≤80 字和复读惩罚。

GRPO 不使用离线偏好对,而是从 clean train prompt 在线采样,用同一规则奖励。

## 三个算法

### DPO

- TRL `DPOTrainer`
- sigmoid DPO loss
- `beta=0.1`
- reference model:TRL 为 PEFT 自动创建的 `ref` adapter
- 100 步,batch 1 × grad_accum 8
- 偏好对为 on-policy best-of-N,不是 gold-vs-worst

### SimPO

- reference-free
- 长度归一化的序列 log-prob
- `loss = -logsigmoid(beta * (avg_logp_chosen - avg_logp_rejected) - gamma)`
- `beta=2.0, gamma=0.5`
- 不加载 reference model
- 100 步,batch 1 × grad_accum 8

### GRPO

- TRL `GRPOTrainer`
- `loss_type="grpo"`(标准 GRPO,不是 DAPO)
- `importance_sampling_level="token"`
- online rollout,G=4,max_completion=96
- `beta=0.04` KL
- 100 步,batch 2 × grad_accum 4

## 运行

```powershell
# 1. 构造偏好对
python rl/alignment/build_preferences.py `
  --max-prompts 400 --num-samples 4 `
  --output runs/2026-10-12/alignment/pairs.jsonl

# 2. DPO
python rl/train_dpo_alignment.py `
  --pairs runs/2026-10-12/alignment/pairs.jsonl `
  --max-steps 100 --output-dir models/alignment_dpo

# 3. SimPO
python rl/train_simpo_alignment.py `
  --pairs runs/2026-10-12/alignment/pairs.jsonl `
  --max-steps 100 --output-dir models/alignment_simpo

# 4. GRPO
python rl/train_grpo_alignment.py `
  --max-steps 100 --output-dir models/alignment_grpo

# 5. 同一评测
python sft/eval_domain.py `
  --data-file runs/2026-10-12/data/clean/test.jsonl `
  --prompt-format chat --max-new-tokens 64 `
  --adapters runs/2026-10-12/models/sft models/alignment_dpo models/alignment_simpo models/alignment_grpo `
  --predictions-dir runs/alignment_compare/predictions `
  --report runs/alignment_compare/eval_chat.md
```

## 口径说明

- DPO/SimPO 用离线偏好对;GRPO 用在线标量奖励。这不是“同一种监督信号”的严格对照,
  而是“同一 SFT、同一数据和评测下三种后训练方法”的工程对比。
- 规则奖励仍以关键词为主,会存在 reward hacking;所以必须同时看 LLM Judge、长度、EOS 和人工抽查。
- 当前是单 seed,结论只能作为趋势,不能作为显著性结论。

## 实测结果(168 条 clean test)

| 模型 | 领域正确率 | 总正确率 | OOD 拒答 | gold_eval | 熟场景 | 未见过 | EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 10.5% | 10.7% | 13.3% | 20.5% | 14.3% | 6.6% | 5.4% |
| SFT | 33.3% | 33.9% | 40.0% | 53.8% | 54.5% | 11.8% | 0.0% |
| DPO | 36.6% | 36.9% | 40.0% | 50.0% | **64.9%** | 7.9% | 0.6% |
| SimPO | **37.3%** | **38.1%** | **46.7%** | 47.4% | 63.6% | 10.5% | 0.0% |
| GRPO | **37.3%** | 37.5% | 40.0% | **55.1%** | 62.3% | **11.8%** | 0.0% |

结论:

- 三种对齐都比 SFT 提高了「熟场景新问法」:54.5% → 62–65%;
- DPO 的熟场景最高,但 gold_eval 低于 SFT;
- SimPO 的总体关键词/OOD 最高,但 gold_eval 只有 47.4%;
- GRPO 的 gold_eval 最高(55.1%),总体关键词与 SimPO 持平在 37.3%;
- 差距只有 1–2 个百分点,单 seed、168 条 test 下不能声称显著;
- 所有模型 EOS 仍接近 0,长度/终止问题没有被对齐算法解决。

完整报告:

- `runs/alignment_compare/eval_chat.md`
- `runs/alignment_compare/stratified/`
