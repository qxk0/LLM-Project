# Standard GRPO 实验归档(只读审计)

> 归档时间:2026-09-29
> 归档方式:**只读取文件 / 计算 hash / 检查配置 / 汇总已有日志**。本文件生成过程中
> **没有启动任何训练或推理、没有重新生成 predictions、没有修改数据集 / clean split / Gold Eval /
> evaluator / SFT / DPO / DAPO / README / 简历,也没有删除或修复任何 checkpoint**。
> 目的:把当前 Standard GRPO 实验的真实状态固定下来,便于项目收口与面试复盘。

---

## 1. 实验目的

尝试增加一个 **Standard GRPO baseline**,与项目已有的 **DAPO**(`rl/train_grpo.py` + `models/rl/`)
形成算法层面的实验对照,使对齐链路补齐为:

```
SFT → DPO → GRPO(本次) → DAPO(已有)
```

约束:不改动既有 DAPO 产物,不新建训练数据,只新增一个独立脚本与独立输出目录
(`rl/train_grpo_standard.py`、`models/grpo/`)。

## 2. 实现

| 项 | 值 |
|---|---|
| trainer | `trl.GRPOTrainer`(TRL **1.9.2**) |
| **loss_type** | **`grpo`**(标准 GRPO objective;按序列长度归一化的 token-level loss) |
| **importance_sampling_level** | **`token`** |
| 与 DAPO 的关系 | 同一套 model loading / tokenizer / dataset 构造 / reward / logging / checkpoint 约定,**唯一差异是 loss_type** |
| GSPO | **未使用**(脚本里只有一行注释说明"不使用 --seq-level,那是 GSPO 风格,本脚本不涉及") |
| PPO | **未使用**(无 PPO trainer、无 PPO 相关配置) |
| reward | `domain_reward`(非 OOD 命中 keyword +1 否则 −1;OOD 含拒答词 +1 否则 −1)+ `format_reward`(回答 ≤80 字符 +0.1)——与 DAPO 完全相同的函数 |
| 训练数据 | `data_eng/output_clean/train.jsonl`(clean split,**leak-free**) |
| 输出目录 | `models/grpo/`(独立,不触碰 `models/rl/`) |
| 代码 | `rl/train_grpo_standard.py` |

算法可验证性(从 `models/grpo/checkpoint-300/training_args.bin` 与 `models/grpo/training_args.json` 读取):

```
loss_type = grpo                 → 不是 dapo
importance_sampling_level = token
is_dapo = False
```
脚本内还有硬断言兜底:`assert training_args.loss_type != "dapo"`、`assert importance_sampling_level == "token"`。

## 3. 训练配置

```
max_steps                    = 300
per_device_train_batch_size  = 2
gradient_accumulation_steps  = 4
learning_rate                = 1.5e-4  (cosine + warmup)
seed                         = 42
num_generations              = 4
max_completion_length        = 64
temperature                  = 0.7
top_p                        = 0.95
top_k                        = 50
beta (KL)                    = 0.04
bf16                         = true
mask_truncated_completions   = false
save_steps / save_total_limit = 50 / 2
```

## 4. 数据(实际读取结果)

| 项 | 值 |
|---|---|
| 训练文件 | `data_eng/output_clean/train.jsonl`(clean split) |
| 该文件行数 | **2676** |
| 其中 OOD 行数 | **41** |
| RL prompt 数(经 `make_domain_dataset`,OOD 上采样到 20%) | **205**(其中 OOD 41,占 20%) |
| 与 clean test(183)的重叠校验 | **exact question = 0、base question = 0**(脚本启动时强制校验,不通过直接退出) |

对照说明(影响后续对比解读,见第 8 节):已有 DAPO 当年用的是旧 `data_eng/output/train.jsonl`
(2275 行 / OOD 100 → RL prompt **500**),该文件与 clean test **存在 77/183 条精确同题的历史重合**。
本次 GRPO **没有**沿用旧文件(否则会把 clean test 喂进训练),因此两者训练数据不同。

## 5. 第一阶段训练(step 1–200,连续)

- 第一次启动:`rl/train_grpo_standard.py --max-steps 300 --seed 42`,从 **SFT 适配器** `models/sft` 起步,
  连续训练到 **step 200**。
- 该阶段 stdout 保存在 `models/grpo_train.log`;其指标被完整保留在 `models/grpo/training_log.json`
  (step ≤ 200 的 **40 条**日志,每 5 步一条,覆盖 step 5→200)。
- 该阶段是**一次未被中断的连续训练**,日志可用于描述"standard GRPO 前 200 步"的真实趋势。

## 6. Resume failure(核心问题)

第一次训练在 step ≈211 因外部原因被中断。随后执行:

```
python rl/train_grpo_standard.py --max-steps 300 --seed 42 \
    --resume-from-checkpoint models/grpo/checkpoint-200
```

**实际恢复情况:**

| 状态 | 是否恢复 |
|---|---|
| optimizer state(`optimizer.pt`) | ✅ 恢复 |
| scheduler state(`scheduler.pt`) | ✅ 恢复 |
| RNG state(`rng_state.pth`) | ✅ 恢复 |
| global_step / epoch(`trainer_state.json`) | ✅ 恢复(从 200 继续,最终 300) |
| **policy active adapter 权重** | ❌ **未恢复** |

**证据 1 — KL 在 resume 边界出现 49× 跳变:**

```
step 200:  KL = 0.4317  (loss 0.01727, reward +0.393, lr 5.05e-05)
step 205:  KL = 0.0088  (loss 0.00035, reward +0.050, lr 4.80e-05)
```
若 policy 真的从 step 200 继续,KL(policy‖SFT) 不可能在 5 步、LR≈5e-5 的情况下从 0.43 掉到 0.009。
该数值量级与训练刚起步时(KL≈0.028)一致,表明 policy 回到了接近 SFT initialization 的状态。

**证据 2 — 参考适配器 hash 等于 SFT:**

```
models/grpo/checkpoint-300/ref/adapter_model.safetensors  sha256(B0C44CC9…) 
models/sft/adapter_model.safetensors                      sha256(B0C44CC9…)   ← 完全一致
```
说明 KL 的锚点(ref)仍是 SFT policy(与第一次训练一致),问题不在 ref,而在 **policy 没有被恢复**。

**证据 3 — 库的代码路径(只读查阅,未修改):**

`transformers 5.14.1` 的 `Trainer._load_from_checkpoint`(`.venv/.../transformers/trainer.py:3410-3426`):
当 checkpoint 目录下存在 adapter **子目录**(本例为 `ref/`)时,代码只遍历这些子目录执行
`model.load_adapter(peft_id, subdir_name, ...)`,随后 `model.set_adapter(active_adapter)`;
**不会**执行 else 分支里的 `model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)`,
因此主 `adapter_model.safetensors`(即 `default` 适配器的权重)**没有载回**。

**后果**:续训段的 policy 实际从 SFT 权重出发,只带着 step 200 遗留的 AdamW 动量与已被消耗的
LR 调度继续跑;同时 `save_total_limit=2` 在保存 `checkpoint-250/300` 时**删除了 `checkpoint-150/200`**,
step-200 的 policy 权重已不存在,无法再据此做正确的续训。

(附带修复:脚本里"续训结束后合并日志时读取起点 checkpoint"的逻辑原先会因目录被删除而报错,
现已加存在性判断与警告;**该修复不影响任何已完成的训练结果**。)

## 7. 后续 100 steps(205–300)的正确解读

- 205–300 这 100 步**不能**被解释为"从 step 200 的 policy 连续训练得到的结果";
- 它是"**SFT initialization + 100 步低 LR(2.3e-5→5e-7,且继承 step-200 动量)**"的另一条轨迹;
- 因此 `models/grpo/checkpoint-250`、`checkpoint-300`、`models/grpo/adapter_model.safetensors`
  都不能代表"从同一 policy 连续训练 300 步的 Standard GRPO"。

## 8. Artifact 有效性分类

### 8.1 可以作为历史实验记录保留

| Artifact | 状态 | 说明 |
|---|---|---|
| `models/grpo/training_log.json` 中 **step ≤ 200 的 40 条记录** | ✅ 保留 | 第一阶段连续训练的完整指标(每 5 步一条) |
| `models/grpo_train.log` | ✅ 保留 | 第一阶段训练 stdout(含 tqdm 与每 5 步 metrics) |
| `models/grpo/checkpoint-150` | ❌ **不存在** | 已被 `save_total_limit=2` 删除(不假设其存在) |
| `models/grpo/checkpoint-200` | ❌ **不存在** | 同上;这也是无法重新正确续训的原因 |

### 8.2 不作为最终 GRPO benchmark

```
models/grpo/checkpoint-250          INVALID_FOR_FINAL_300_STEP_BENCHMARK
models/grpo/checkpoint-300          INVALID_FOR_FINAL_300_STEP_BENCHMARK
models/grpo/adapter_model.safetensors  INVALID_FOR_FINAL_300_STEP_BENCHMARK
```

> `INVALID` 在这里**仅表示**:不能作为"最终 300-step 连续 Standard GRPO baseline"使用。
> **不表示**文件损坏或不可读取——它们仍可正常加载、可用于说明问题现象。
> 这些文件**没有被删除**,按任务要求原样保留。

### 8.3 未受影响的既有产物

`models/rl/**`(DAPO,含 `checkpoint-250/300`、`adapter_model.safetensors`、`training_log.json`)
与 `models/sft`、`models/dpo` **均未被本次实验修改**(mtime 保持 8/21 与 8/20)。

## 9. 训练过程中观察到的问题(全部读自现有日志)

### 9.1 第一阶段(step ≤ 200,40 条日志)

| 指标 | min | mean | max |
|---|---|---|---|
| loss | 0.001118 | 0.01487 | 0.03107 |
| reward | −0.1225 | 0.449 | 0.99 |
| reward_std | 0.2154 | 0.6446 | 1.034 |
| rewards/domain_reward/mean | −0.20 | 0.3625 | 0.90 |
| KL | 0.02795 | 0.3719 | 0.7767 |
| entropy | 0.1235 | 0.2667 | 0.4218 |
| **completions/mean_length** | **64** | **64** | **64** |
| **completions/clipped_ratio** | **1.0** | **1.0** | **1.0** |
| **completions/mean_terminated_length** | **0** | **0** | **0** |
| **frac_reward_zero_std** | **0.2** | **0.5975** | **0.9** |
| learning_rate | 5.05e-05 | 9.925e-05 | 1.48e-04 |

### 9.2 续训段(205–300,20 条日志,仅作现象记录)

| 指标 | min | mean | max |
|---|---|---|---|
| loss | 0.000337 | 0.001955 | 0.003718 |
| reward | 0.05 | 0.6378 | 1.06 |
| reward_std | 0.04949 | 0.5427 | 0.9982 |
| rewards/domain_reward/mean | 0.0 | 0.575 | 1.0 |
| KL | 0.008413 | 0.04887 | 0.09296 |
| entropy | 0.2099 | 0.3066 | 0.4276 |
| completions/mean_length | 63.88 | 63.99 | 64 |
| completions/clipped_ratio | 0.975 | 0.9988 | 1.0 |
| completions/mean_terminated_length | 0 | 0.59 | 11.8 |
| frac_reward_zero_std | 0.1 | 0.51 | 0.7 |

### 9.3 关键现象与措辞

1. **生成长度顶格**:第一阶段 `mean_length` 恒为 **64**、`clipped_ratio` 恒为 **1.0**、
   `mean_terminated_length` 恒为 **0** → 所有 rollout 都跑满 `max_completion_length`,没有输出终止符。
   这与项目其它实验观察一致(from-scratch/评测中 base、SFT、DAPO 的 EOS 命中率同样为 0%)。
   `max_new_tokens`/`max_completion_length` 等**未做任何调整**。
2. **组内奖励大量相同**:**`frac_reward_zero_std` 均值 ≈ 0.60**(区间 0.2–0.9),即在约 60% 的采样组里,
   4 条 response 拿到**完全相同**的 reward。这意味着这些组的 relative advantage 可能退化为零,
   **可能导致 / 表明存在退化学习信号**(而非断言整个 GRPO 训练失败);续训段该指标均值 ≈0.51,同样存在。
3. **KL 与 entropy**:第一阶段 KL 从 0.028 单调升到 0.777,entropy 在 0.12–0.42 之间波动 → 策略在
   持续偏离 SFT,但没有出现 NaN/Inf(loss 全程有限)。
4. **无 NaN / Inf**:两段日志中 loss 均为有限值。
5. **未观察到"全部拒答"或"单一固定模板"**:`rewards/domain_reward/mean` 在 −0.2~1.0 之间波动,
   说明 rollout 中既有答对也有答错,没有退化成单一输出;但"是否有大量重复输出"仅凭训练日志无法判断,
   需要看生成样本(本次未做推理,故不下结论)。

## 10. 最终结论

> 本实验成功验证了 **Standard GRPO training path 可以启动并运行**(TRL `GRPOTrainer`、
> `loss_type=grpo`、`importance_sampling_level=token`,第一阶段连续训练到 step 200,无 NaN);
> 但由于 **PEFT checkpoint resume 时 active adapter 未正确恢复**,最终 300-step artifact
> 不满足"连续 300-step Standard GRPO baseline"的实验定义。
> 因此本实验**不用于最终项目性能对比**。

## 11. 如果将来要得到可用的 300-step GRPO baseline(仅说明,本次未执行)

1. **不要用 `--resume-from-checkpoint` 续训 PEFT 模型**;改为**一次性跑满 300 步**
   (单次运行,中途不中断),或
2. 若必须续训:在创建 trainer 后**显式把 checkpoint 的主 adapter 权重载入 `default`**
   (而不是依赖 `Trainer._load_from_checkpoint`),并显式避免让 `save_total_limit`
   删掉续训起点;同时校验 resume 前后 KL 是否连续(出现 49× 跳变即视为 resume 失败)。
3. 训练完成后应做一次**生成样本人工抽查**,以补足"是否大量重复 / 是否退化成固定模板"的判断。

## 12. 本次归档的合规声明

- 未启动任何训练或 evaluation / inference;未重新生成任何 predictions;
- 未修改数据集、clean split、Gold Eval、evaluator 判定逻辑、SFT / DPO / DAPO 代码与产物、README、简历;
- 未删除任何 checkpoint 或训练日志;未自动修复或重跑 GRPO;
- 本次只新增文档:`data_eng/GRPO_EXPERIMENT_ARCHIVE.md`,并在 `data_eng/INTERVIEW_FACTS.md` **末尾追加**一节;
- `models/grpo/training_log.json`、`training_log_resumed_part.json`、`training_args.json` 是在上一轮任务中
  为保留训练记录而写入的(含 resume 缺陷说明),本次仅读取、未改动。
