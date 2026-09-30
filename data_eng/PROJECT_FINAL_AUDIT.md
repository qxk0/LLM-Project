# LLM-Project 最终收尾审计(只读)

> 审计方式:只读取当前工作区文件 / git 历史 / 训练日志 / checkpoint 元数据(sha256、training_args.bin、trainer_state.json),
> **未训练、未推理、未调用任何 API、未生成数据、未修改任何文件**。
> 唯一新增文件:`data_eng/PROJECT_FINAL_AUDIT.md`、`data_eng/INTERVIEW_FACTS.md`。
> 审计时间:2026-09-28。

---

## 一、从零 GPT(Base GPT)审计

代码位置:`pretrain/model.py`(模型)、`pretrain/config.py`(配置)、`pretrain/train.py`(训练)。

| 能力 | 代码 | 是否真的跑过实验 | 结论 |
|---|---|---|---|
| RMSNorm | `model.py:45 class RMSNorm`(可切换 `norm_type`) | ✅ `models/exp_swiglu`(norm=rmsnorm 已记录) | **REAL** |
| LayerNorm | `model.py:35 class LayerNorm` | ✅ 所有基线实验 | REAL |
| RoPE | `model.py:96-104` 预计算 cos/sin + `_apply_rope()`(手写旋转) | ✅ `models/exp_rope` | **REAL** |
| ALiBi | `model.py:106-113` + `137-144` 手写 bias 路径 | ❌ 无 `exp_alibi`、无对应 log | **NOT FOUND(仅代码)** |
| MHA | 默认 `attention_type="mha"` | ✅ 基线 | REAL |
| MQA | `model.py:72-73`(`n_kv_heads=1`) | ❌ 无 `exp_mqa` | **NOT FOUND(仅代码)** |
| GQA | `model.py:74-75` + `132-135`(`repeat_interleave`) | ✅ `models/exp_gqa` | **REAL** |
| GELU | `model.py:178-180` | ✅ 基线 | REAL |
| SwiGLU | `model.py:171-176` + `184-186`(8/3 宽度对齐参数) | ✅ `models/exp_swiglu` | **REAL** |
| tied embedding | `model.py:224`(`lm_head.weight = token_embedding.weight`) | ✅ 默认开启,所有实验都是 tie=True | REAL |
| untied embedding | 开关存在(`--tie-embeddings false`) | ❌ 无相关实验目录 | **NOT FOUND(仅代码)** |
| 注意力实现方式 | `model.py:147` `F.scaled_dot_product_attention(...)`;ALiBi 分支是手写 `softmax(QK^T/√d + bias)` | — | **调用 PyTorch SDPA,不是自研 FlashAttention** |
| KV Cache | `generate()`(`model.py:257-271`)每步 `self.forward(idx_cond)` 重算整个上下文 | — | **NOT FOUND(未实现 KV Cache)** |

补充说明(容易被面试官抓):

1. `exp_learned / exp_12m` 的 log 里 `activation=None / norm_type=None`,因为这些实验(8/4–8/17)早于 `activation/norm_type` 字段加入(8/19)。当时的代码只有 GELU+LayerNorm 一条路,所以基线实质就是 **GELU + LayerNorm**。
2. `exp_learned` 与 `exp_12m` 的 `n_params`(12.29M)与末次 val_loss(2.5148)完全相同 → 同一个配置的两次运行,**不能算两个独立数据点**。
3. generate 里没有 KV Cache,所以 "GQA 省 KV Cache 显存" 在本项目**没有被测到**;GQA 的实测收益只有参数量与注意力计算量。

---

## 二、Scaling 实验审计(参数规模 / 步数 / val loss)

数据来源:`models/exp_*/log.jsonl` 的 `meta.n_params` 与最后一次 `eval` 事件(全部为真实日志,非推测)。

| params | training steps | val loss | evidence |
|---|---:|---:|---|
| 0.95M | 2000 | 3.5068 | `models/exp_1m/log.jsonl` + `ckpt_1000/2000.pt`、`best.pt` |
| 4.26M | 2000 | 2.8285 | `models/exp_5m/log.jsonl` + 同上 |
| 12.29M | 2000 | 2.5148 | `models/exp_12m/log.jsonl` + 同上(与 `exp_learned` 同配置) |
| 27.40M | 2000 | 2.3005 | `models/exp_25m/log.jsonl` + 同上 |
| (参考)12.29M | **6000** | **1.9714** | `models/pretrain/log.jsonl`(23 次 eval,含 ckpt_300/1000/2000/3000/4000/5000/6000) |

判定:

```
NOT STRICT SCALING LAW
```

理由(全部可核查):

1. 只有 **4 个规模点**,且**训练步数/数据量完全相同**(都是 2000 步),没有 compute-optimal 的 token 预算扫描;
2. `pretrain/plot_results.py` 里**没有任何幂律拟合**(`grep polyfit|log10|curve_fit` 无结果),只有线性坐标的折线 + 标注;
3. 因此正确的表述是 **"不同参数规模下的 validation-loss scaling trend"**,不是 scaling law 验证。

---

## 三、Alignment Pipeline 审计

### SFT(`sft/train_sft.py` + `models/sft`)

| 项 | 实测值 | 证据 |
|---|---|---|
| base model | `Qwen/Qwen2.5-0.5B` | `models/sft/adapter_config.json` |
| 量化 | 4bit NF4 + double quant,计算精度 fp16 | `train_sft.py` + `training_args.bin(fp16=True)` |
| LoRA | r=16, alpha=32, dropout=0.05, 7 个投影层 | `adapter_config.json` |
| 训练数据 | `data_eng/output/train.jsonl` 前 2000 行(旧版 dataset) | epoch=2.4 × 600 步 × batch1×accum8 → 2000 条 |
| steps / batch | **600 步**,batch 1 × grad_accum 8 | `checkpoint-600/trainer_state.json`(global_step=600, epoch=2.4) |
| lr | 2e-4,cosine + 3% warmup | `training_args.bin` |
| assistant-only mask | ✅ `labels[:mask_len] = -100` | `train_sft.py:153-170` |
| checkpoint | `models/sft/{adapter,checkpoint-500,checkpoint-600}`,根目录 = checkpoint-600(sha256 相同) | `B0C44CC92A6AF5F8` |
| evaluation | **无验证集评测**(Trainer 未传 eval_dataset),只有 train_loss 曲线(均值 0.7875,末步 0.5167) | `models/sft/training_log.json` |

### DPO(`rl/train_dpo.py` + `models/dpo`)

| 项 | 实测值 | 证据 |
|---|---|---|
| base / reference | `models/sft` 适配器 + Qwen2.5-0.5B(4bit) | `train_dpo.py:123-128` |
| 偏好数据来源 | **在线构造、不落盘**:每题采样 4 条 → chosen = **知识库 gold 答案**,rejected = 规则分最低的采样回答 | `train_dpo.py:86-92`(注:与 docstring/README 里"最高分当 chosen"的说法矛盾) |
| beta / lr | 0.1 / 5e-5 | `checkpoint-200/training_args.bin` |
| max_length / loss | 128 / `['sigmoid']` | 同上 |
| steps | **两轮**:(a) 200 步(epoch 7.70,约 208 对)8/19;(b) **10 步**(epoch 0.22,约 360 对)8/20 | `checkpoint-200/trainer_state.json`、`checkpoint-10/trainer_state.json` |
| 最终 checkpoint | **`models/dpo/` 根目录 == `checkpoint-10`(sha256 完全一致 `562CD423…`)**;200 步版本只在 `checkpoint-200`(`8AD7D6A3…`) | 文件哈希 |
| 评测 | 当前 `data_eng/output/eval_domain.md`(288 条)**没有 DPO 行**;历史 138 条报告有 dpo 82.4/53.8;另有两份小样本报告(40 条/12 条)只在 git 历史里 | 文件 + `git show` |

结论:**DPO 真实训练过,但 (1) 偏好对没有落盘 (2) 仓库里"默认 DPO 模型"是一次 10 步实验 (3) 从未在当前测试集上评测过** → 状态 **PARTIAL**。

### RL(`rl/train_grpo.py` + `models/rl`)

| 项 | 实测值 | 证据 |
|---|---|---|
| 算法 | TRL `GRPOTrainer`,但 **loss_type = dapo** | `checkpoint-300/training_args.bin` |
| importance_sampling_level | **token**(不是 sequence) | 同上 |
| reward | `domain_reward`(非 OOD:命中 keywords +1 否则 −1;OOD:含拒答词 +1 否则 −1)+ `format_reward`(≤80 字 +0.1) | `train_grpo.py:128-145` |
| rollout | `num_generations=4`,max_completion_length=**64**,temperature 0.7,top_p 0.95,top_k 50 | `training_args.bin` |
| 训练数据 | `data_eng/output/train.jsonl`,OOD 上采样到 20% → 约 500 条 prompt | `make_domain_dataset()` + train.jsonl(ood=100) |
| steps / batch | **300 步**,batch 2 × grad_accum 4,lr 1.5e-4,beta=0.04,bf16 | `checkpoint-300/trainer_state.json`(global_step=300, epoch=1.2) |
| checkpoint | `models/rl/{adapter,checkpoint-250,checkpoint-300}`,根目录 = checkpoint-300(`451923CE…`,与 SFT 的 `B0C44CC9…` 不同 → 权重确实被更新) | 文件哈希 |
| 训练日志 | 61 条记录;末步 `reward=0.8275`、`rewards/domain_reward/mean=0.75`、`kl≈0.57`;`completions/clipped_ratio=1.0`、`mean_terminated_length=0.0`(全程 64 token 截断、无 EOS) | `models/rl/training_log.json` |

**GRPO / DAPO / GSPO 三者的真实情况**:

```
GRPO : REAL(用的是 TRL GRPOTrainer,GRPO 算法族)
DAPO : REAL(loss_type=dapo 实际写入 checkpoint,虽然它就是 TRL 的默认值)
GSPO : NOT FOUND
       - 只有 --seq-level 代码开关(loss_type=grpo + importance_sampling_level=sequence)
       - 从未运行过,无任何产物
       - 且当前 TRL 1.9.2 的 loss_type 列表里没有 "gspo"
```

---

## 四、数据构建审计

代码:`data_eng/kb.py`(知识库)、`seeds.py`(55 条人工种子)、`build_dataset.py`(扩展/清洗/血缘/切分)、`evolve.py`(GLM 蒸馏)、`rl/rft.py`(on-policy 自采样筛选)、`gold_eval_seeds.py`(78 条人工 Gold Eval)、`leakage_audit.py`(审计)。

### 各来源真实数量与去向(`data_eng/output_clean/stats.json`)

| source | 数量(生成阶段) | train | val | test | 说明 |
|---|---:|---:|---:|---:|---|
| seed(人工) | 55 条 → ×5 前缀 = 275 | 含在无 source 行内 | — | — | 人工基于 KB 编写 |
| template(模板+实体) | 201 条 → ×5 = 1005 | 同上 | — | — | price/recommend/customize/topping/ood |
| paraphrase | 1024(相对种子行的增量) | 同上 | — | — | 5 种前缀口语化改写 |
| evolved(Self-Instruct/GLM) | **86** | 71 | 0 | 0 | 有产物,无 API 调用日志 |
| onpolicy(RFT) | **1521** | 1480 | 0 | 0 | SFT policy 自采样 + 关键词验证器筛选 |
| gold_eval(人工评测题) | **78** | 0 | 0 | **78** | 本轮新增,只进 test |
| 合计(去重清洗后) | 2924 | **2676** | **65** | **183** | 旧版(泄漏)为 2847 / 2275 / 284 / 288 |

注:seed/template/paraphrase 三类在旧版没有 `source` 字段(无法拆分),clean 版已补齐 `source/variant_type/seed_id/family_id/base_question/entity_key/lineage_confidence`。

### 数据泄漏治理结果

| 指标 | 旧 split(288 条 test) | clean split(183 条 test) |
|---|---|---|
| exact normalized question train∩test | 188/288 行同题 | **0** |
| base question(去前缀)train∩test | 277/288 行同源 | **0** |
| OOD base question train∩test | 13/13 同源 | **0/60** |
| test 中的模型生成数据 | onpolicy 146 + evolved 11 = 157 | **0** |
| family / split-group 跨 split | 无 lineage 字段,无法保证 | **0**(group 级切分保证) |
| gold_eval ∩ train/val/old_test | — | **0 / 0 / 0** |

`data_eng/output_clean/leakage_audit.json` 中 18 条 assertion **全部 PASS**(`all_passed: true`),且 split 可复现(固定 seed=42,两次运行 train/val/test 的 sha256 完全一致)。

**已知仍未解决的残余风险(如实记录)**:clean test 仍与 train 共享"实体"(如 train 有"杨枝甘露什么价位",test 有"杨枝甘露和茉莉绿茶哪个更贵")——这是"同产品、不同业务场景"的新场景评测,已逐题在 `GOLD_EVAL_AUDIT.md` 说明;无法完全避免,除非彻底换产品线。

---

## 五、评测审计(只读已有结果,未重跑)

### 1. 历史结果(都存在,但**都不能作为最终指标**)

| 报告 | 测试集 | base | SFT | DPO | RL | 问题 |
|---|---|---|---|---|---|---|
| `cb62588` 版本(README 引用) | 128 条 | 5.2/0.0 | 90.5/58.3 | (dpo 87.9) | 91.4/100 | 泄漏测试集 |
| `HEAD` 已提交版 | 138 条 | 7.2 | 88.0/76.9 | 82.4/53.8 | 67.2/84.6 | 泄漏测试集 |
| 工作区未提交版(`data_eng/output/eval_domain.md`) | 288 条 | 7.3 | 88.0/**0.0** | 未评测 | 83.3/100 | 泄漏 + test 混入 157 条模型生成数据 |
| `data_eng/output_clean/eval_domain_clean.md`(clean 183,Pass-1) | **183 条(干净)** | 21.1/5.0 | 60.2/8.3 | 70.7/51.7 | 47.2/90.0 | ⚠️ 仍用**修复前的裸 prompt + 64 token 上限**,不是最终口径 |

(表格中"x/y"= 领域正确率 / 拒答率;百分号。)

> clean 183 的 Pass-1 结果**只能当历史参考**:它跑在修 prompt format / 改 max_new_tokens **之前**(裸 prompt、64 token、全部跑满不停),已确认会系统性压低 SFT/RL 的领域正确率(示例:sft 输出"自治区教育厅 2019 年 10 月 1 日…"这类跑偏内容)。

### 2. prompt format 消融(40 次生成,`prompt_ablation/summary.json`)

| model | format | accuracy(5题) | OOD refusal(1题) | EOS hit | avg tokens |
|---|---|---|---|---|---|
| base | chat / raw | 0% / 0% | 0% / 0% | 0% / 0% | 32.0 / 32.0 |
| sft | chat / raw | 40% / 20% | 0% / 0% | 0% / 0% | 32.0 / 32.0 |
| dpo | chat / raw | 60% / 60% | 100% / 100% | **0% / 40%** | 32.0 / **28.4** |
| rl | chat / raw | 60% / **80%** | 100% / 100% | 0% / 0% | 32.0 / 32.0 |

结论:只有 **DPO 对 prompt format 有明确的行为差异(EOS 0% → 40%)**;SFT/RL 只有 1 题量级的准确率差异(n=5,不可当结论);base 无指标差异。

### 3. rule-based evaluator 的已知局限(只记录,未修改)

判定规则(`sft/eval_domain.py`):

```python
hit     = any(k in resp for k in row["keywords"])      # 子串命中即算答对
refused = any(m in resp for m in REFUSAL_MARKERS)      # 出现拒答词即算正确拒答
```

已实测到的 false positive / 局限:

1. **无事实校验**:`dpo/chat` 在"外卖结束时间 vs 门店关门时间"题回答"门店关门时间是晚上 **21:30**"(事实错误,应为 22:00),因命中关键词 `21:30` 被判答对。
2. **拒答词位置无关**:`dpo/raw` 在 OOD 题先输出"最近的酒店在哪里?最近的火车站在哪里?"再补一句"不好意思,这个问题超出了我的服务范围",仍计为成功拒答。
3. **数字子串误匹配**:keyword 如 `"12"` 会被 `"2012"`/`"125"`/`"12:00"` 命中(无词边界)。
4. **拒答词过宽**:`"咨询"`/`"抱歉"` 出现在正常业务回答里也会被计为拒答,导致拒答率虚高。
5. **判定与语义无关**:回答长度、重复、是否真的回答了问题都不影响判定 → 复读/截断的回答只要含关键词就算对。
6. LLM-as-Judge 只对 **15 条抽样**(且需要 GLM API),不能代表整个 test set;当前 clean 评测**没有** judge 结果。

---

## 六、最终项目事实表

状态取值:`REAL`(代码+产物+可核查证据)/ `PARTIAL`(部分成立或证据不完整)/ `NOT FOUND`。

| 项目 | 状态 | 真实证据 | 是否建议写简历 |
|---|---|---|---|
| Self-Instruct | **PARTIAL** | `evolve.py`(GLM API)+ `evolved.jsonl` 86 条(已进 train 71 条);**无调用日志**,无法逐条证明来自 API | ✅ 可写,但写"86 条",不要写"数千条" |
| SFT | **REAL** | `models/sft` + `training_log.json` 600 步 + `checkpoint-600` + 均值 train_loss 0.7875 | ✅ |
| LoRA | **REAL** | r=16/alpha=32/dropout=0.05/7 投影层,三份 adapter_config.json | ✅ |
| QLoRA | **PARTIAL** | 4bit NF4 + double quant + LoRA(未启用 paged optimizer,严格说是 QLoRA-style) | ✅ 写 "4bit QLoRA" |
| DPO | **PARTIAL** | 真实训练(200 步 + 10 步)+ checkpoint;但偏好对未落盘、根目录是 10 步版、当前测试集无 DPO 行 | ⚠️ 写"实现并跑通 DPO",数字必须重测 |
| GRPO | **REAL** | TRL `GRPOTrainer` + `models/rl` 300 步 + 日志 | ✅(口径见下) |
| DAPO | **REAL** | `loss_type=dapo` 写入 `checkpoint-300/training_args.bin` | ✅ 写"DAPO loss" |
| GSPO | **NOT FOUND** | 仅 `--seq-level` 代码开关,从未运行;TRL 1.9.2 无 gspo loss | ❌ **绝对不写** |
| RoPE | **REAL** | `model.py` 手写实现 + `exp_rope`(2.3779 vs learned 2.5148,−5.4%) | ✅ |
| GQA | **REAL** | `model.py` + `exp_gqa`(11.41M vs 12.29M,−7.2% 参数;2.5261 vs 2.5148) | ✅(注意别写 KV Cache) |
| MQA | **NOT FOUND** | 代码分支存在,无实验目录 | ❌ |
| ALiBi | **NOT FOUND** | 代码分支存在,无实验目录 | ❌ |
| SwiGLU | **REAL** | `model.py` + `exp_swiglu`(2.4356 vs 2.5148,−3.2%) | ✅ |
| RMSNorm | **REAL** | 同上(`norm_type=rmsnorm` 已记录) | ✅ |
| FlashAttention | **PARTIAL** | 用 `F.scaled_dot_product_attention`(可能调度到 FlashAttention 内核),**不是自研 kernel**;ALiBi 分支还是手写 attention | ⚠️ 写"PyTorch SDPA" |
| KV Cache | **NOT FOUND** | `generate()` 每步重算全上下文,无 cache | ❌ |
| Scaling law | **NOT FOUND** | 4 个规模点、同 2000 步、无 power-law 拟合 | ❌ 改写为 scaling trend |
| Scaling trend | **REAL** | 0.95M/4.26M/12.29M/27.4M → val 3.5068/2.8285/2.5148/2.3005 + 6000 步基线 1.9714 | ✅ |
| Clean split | **REAL** | `output_clean/` 183 test,全部 leakage 指标为 0,18 条 assertion PASS,可复现 | ✅ 强项 |
| Gold evaluation | **REAL** | `gold_eval_seeds.py` 78 条(10 intent,事实标注 kb.py 出处),只进 test | ✅ 强项 |
| Prompt-format ablation | **REAL** | `prompt_ablation/` 40 条记录 + `PROMPT_ABLATION.md` | ✅(写"诊断实验") |
| (补充)RFT / on-policy | **PARTIAL** | `rft.py` + `onpolicy.jsonl` 1521 条(已进 train 1480);但只覆盖 price/recommend、无 OOD、无 rejected/reward 落盘 | ⚠️ 写"简化版拒绝采样" |
| (补充)Rule-based 评测 | **REAL** | `sft/eval_domain.py` + 多个报告 | ✅ 但需说明局限 |
| (补充)LLM-as-Judge | **PARTIAL** | 代码 + 历史 138 条集上 15 条抽样(sft 4.27/dpo 4.00/rl 3.20);clean 集无结果 | ⚠️ 只能写"抽样 15 条" |

---

## 七、README / 简历不一致清单

| 位置 | 现在的说法 | 实际情况 | 处理 |
|---|---|---|---|
| README 指标表 | 基座 5.2% / SFT 90.5% / RL 91.4% / 总正确率 92.2% / 长度 73.8→61.9 | 这些数字来自 **128 条旧(泄漏)测试集**;clean 183 集上(修复前口径)是 base 21.1 / sft 60.2 / dpo 70.7 / rl 47.2 | **必须改**:换成 clean 集 + 新口径重测后的数字,或标注"旧口径/旧测试集" |
| README | "55 条种子 → 1250 条(22.7x)" | 实际:55 seed → 基础样本 256 条 → 前缀改写 1280 个问题(1250 个唯一问题);clean 版最终 2924 条(含 1480 on-policy + 86 evolved + 78 gold_eval) | 改数字或加限定 |
| README | "1000+ 条领域数据" | SFT 实际用 2000 条(旧 train.jsonl 前 2000 行,其中 1188 条 on-policy、0 条 OOD) | 写准确数字 + 说明构成 |
| README | "按 0.2 比例混入通用指令数据" | 当前 `--mix-general` 默认 **0.0**;0.2 混比是历史实验 | 标注"实验过 0.2,最终默认 0" |
| README | "0.2 混合导致拒答率 92.3%→76.9%" | 76.9% 有报告(138 条集),**92.3% 在任何产物中都找不到**(只在 README 出现) | 删掉 92.3% 或补证据 |
| README | "Scaling Law 验证" | 4 点同预算,无 power-law 拟合 | 改为 "scaling trend" |
| README | "Flash Attention(PyTorch SDPA)" | 表述本身没错,但要避免被读成"自研" | 保留,但简历里写明是 SDPA |
| README | "截断掩码留作开关(实验发现开启后领域正确率下降)" | 有开关,但该对比实验的产物未保存 | 弱化表述 |
| README(DPO 段) | "让 SFT 采样多个回答,最高分当 chosen" | 代码实际是 **gold 答案当 chosen**(`train_dpo.py:86`) | 与代码对齐 |
| 简历 | "评测集与训练集严格分离" | 只有 clean split 成立 | 改为"发现并修复了 family 级泄漏;clean split 所有泄漏指标为 0" |
| 简历 | "拒答率提升至 100%,总正确率 92%+" | 来自 128 条泄漏集;clean 集(修复前口径)rl 拒答 90%、总 61.2% | **必须改** |
| 简历 | "LLM-Judge 平均 4+/5" | 仅 15 条抽样、138 条集、针对历史 checkpoint | 改为"抽样 15 条得 4.27/5(旧测试集)"或删除 |
| 简历 | "DAPO 动态裁剪 + 支持 GSPO 序列级损失" | GSPO 从未运行 | 删掉 GSPO |
| 简历 | "1M/5M/12M/25M 验证缩放定律" | 实际 0.95M/4.26M/12.29M/27.4M,趋势而非定律 | 改数字 + 改措辞 |

---

## 八、最终结论

### A. 项目真实完成度

- **从零 GPT 架构研究**:REAL。12.29M 参数 decoder-only Transformer,自写 RoPE/ALiBi/MQA/GQA/SwiGLU/RMSNorm 与 SDPA 注意力;真实跑过 3 组消融(RoPE、GQA、SwiGLU+RMSNorm)+ 4 档规模趋势 + 6000 步基线;有 checkpoint、log、曲线图。**缺口**:无幂律拟合、无 MQA/ALiBi/untied 实验、无 KV Cache。
- **领域数据管道**:REAL(且是本项目最扎实的部分)。KB → 55 seed → 模板/实体/前缀改写 → 去重/长度过滤 → 血缘元数据 → group 级切分 → 自动化 leakage audit;并额外接入 GLM 蒸馏 86 条 + on-policy 自采样 1521 条 + 人工 Gold Eval 78 条。
- **对齐训练**:SFT REAL / DPO PARTIAL / RL(GRPO+DAPO)REAL。三个阶段的 checkpoint 与训练日志都在,LoRA 权重哈希互不相同(证明确实更新)。
- **评测**:clean split REAL(183 条,零泄漏、可复现);evaluator 已修 prompt/EOS/进度/落盘/离线统计;但**最终 clean 指标尚未在修复后的口径下产出**。

### B. 可以写进简历的内容(有硬证据)

1. 自建垂直领域数据管道:55 条人工 seed(10 intent)+ 模板/实体/前缀改写 + `(q,a)` 去重 + 长度过滤 + 分层切分;输出数据统计报告。
2. **数据泄漏治理**(建议作为主线亮点):发现"同一 seed 的改写跨 train/test"的 family 级泄漏(旧集 188/288 同题、277/288 同源、13/13 OOD 同源),实现 `seed_id/family_id` 血缘 + **group 级切分**(同一 family/实体/seed 只进一个 split)+ 6 层 leakage audit(全部 assertion PASS)+ 183 条零泄漏 test + 78 条人工 Gold Eval。
3. QLoRA SFT:Qwen2.5-0.5B + 4bit NF4 + LoRA(r16/alpha32/7 层),600 步,assistant-only loss masking,4GB 显存跑通(有训练日志)。
4. DPO:知识库 gold 作 chosen、规则低分采样作 rejected 自动构造偏好对,跑通 200 步与 10 步两轮训练(有 checkpoint)。
5. RL 对齐:TRL GRPOTrainer + **DAPO loss**(正负优势分开裁剪)+ 规则对称奖励(命中/拒答 ±1、简洁 +0.1),300 步,KL=0.04(有日志)。
6. 从零 GPT:12.29M 参数,自写 RoPE/GQA/SwiGLU/RMSNorm,用 PyTorch SDPA 做高效注意力。
7. 架构消融 + 规模趋势:RoPE −5.4% val loss、GQA −7.2% 参数(+0.4% loss)、SwiGLU+RMSNorm −3.2% loss、6000 步基线 1.9714、四档规模 3.51→2.30。
8. 评测工程:发现并修复 evaluator 的 prompt format / EOS / 速度问题,支持逐条 prediction 落盘 + 离线复算(不重复推理)+ 实时进度/ETA。
9. prompt-format 诊断消融:发现 DPO 在 raw/chat 下 EOS 行为差异(0%→40%),定性出"各阶段训练 prompt format 不一致"的工程问题。

### C. 必须修改的简历内容

见第七节的表格。最关键的 6 条:

1. 删掉 **92.2% / 90.5% / 91.4% / 拒答 100% / 5.2%→90.5%** 这类旧集数字,或明确标注"旧 128 条测试集(已废弃)"。
2. 删掉 **GSPO**(以及"DAPO+GSPO")。
3. 删掉"自研 **FlashAttention**"→ 改"PyTorch SDPA"。
4. 删掉 **MQA / ALiBi / untied embedding 消融**、**KV Cache**。
5. "**Scaling Law 验证**" → "不同参数规模下的 validation-loss scaling trend(4 档,无幂律拟合)"。
6. "**2500+ 条数据**" → 写清构成(去重后 2847/2924 条,其中 1480 条 on-policy + 86 条蒸馏 + 78 条人工评测,只有 ~1200 条是人工/模板 gold)。

### D. 当前项目最大的 5 个技术风险

1. **最终 clean 指标仍缺**(最高优先):183 条 clean test 只在"修复前 evaluator"下跑过一次;prompt format/EOS/max_new_tokens 都变了之后必须重测,否则所有数字都无法写进简历。
2. **keyword evaluator 偏宽松**:无事实校验 + 无词边界 + 拒答词位置无关,已实测到至少 2 例误判(把答错的判成答对)。最终报告若只给这一列指标,面试官一旦抽查样本就会暴露。
3. **模型生成质量本身很差**:0.5B + 600 步 SFT 在干净测试上领域正确率仅 60.2%(且修复前口径),回答普遍"答对后复读",EOS 命中率 0%(base/sft/rl 在两种 prompt format 下都是 0),这是模型/数据规模问题,不是评测能掩盖的。
4. **训练数据构成失衡**:train 2676 条里 1480 条(55%)是 on-policy 自生成、且只覆盖 price/recommend;SFT 实际消费的前 2000 行里 OOD = 0 条 — 这直接解释了"拒答率学不会";另外 RFT 的 `load_rows` 只取文件前 N 行,存在系统性覆盖偏差。
5. **DPO 链条不完整**:当前 `models/dpo` 是 10 步实验、偏好对未落盘、从未在 clean test 上评测过;简历里任何 DPO 数字目前都没有可复现来源。

### E. 如果只允许再做 3 个实验,优先做什么

1. **在 clean 183 上跑一次修复后的正式评测**(4 个 checkpoint;若时间允许再加 dpo/checkpoint-200),按"各自训练格式"或两种格式各跑一遍并分开报告 → 这是唯一能把项目"补齐最后一块"的动作,产出可直接写简历的数字 + 4 份 prediction JSONL。成本:每模型约 7–14 分钟。
2. **扩一版 Gold Eval 覆盖到每 intent ≥20 题(现 10–60)**,并补上知识库目前缺失的事实(杯型/套餐、坚果过敏原、漏发/超时售后、积分有效期、加盟等),把 test 从 183 扩到 ~250–300,让"clean 评测"在统计上站得住(现有 105→183 已做过一次,这是同一套流程的增量)。
3. **给 rule-based evaluator 加一层"事实级"校验**(只用于报告,不改主指标):例如要求命中**全部** keywords、数字做词边界匹配、拒答必须出现在前 2 句内 → 量化当前指标的 false positive 幅度,并在报告里给"严格口径 / 宽松口径"两列。

### F. 如果不再做任何实验,能否形成完整面试故事?

**能,但必须换主线。** 现在最有说服力、也最经得起追问的不是"92% 准确率",而是:

> "我在一个垂直客服项目里**发现并修复了训练/评测数据泄漏**:旧测试集 288 条里 188 条与训练集同题、13/13 OOD 同源;我引入了 seed/family 血缘 + group 级切分 + 6 层自动化 leakage audit,把泄漏清零并产出 183 条零泄漏 test 和 78 条人工 Gold Eval;在这个过程中我还发现评测器本身的 prompt format 与训练不一致、EOS 不生效导致所有回答跑满上限、评测耗时被生成长度主导,并做了诊断与修复。"

配套可讲的"负结果/踩坑"同样真实:模型在干净测试上指标大幅回落(说明旧指标是泄漏撑起来的)、0.5B+600 步后模型答对后复读且不输出终止符、DPO 的 chosen 用 gold 而不是采样最优、GRPO 实际用的是 DAPO loss、RFT 只覆盖 price/recommend。

需要主动打预防针的:**别说 GSPO、别说自研 FlashAttention、别说 scaling law、别引用 92% 系列数字**;被问到 clean 指标时,直接说"clean 集最终评测尚未完成,目前只有修复前的一次(60.2%/47.2% 等),我不引用旧泄漏集的数字" —— 这种回答在面试里比引用错数字安全得多。

---

## 九、本次审计的合规声明

- 未训练任何模型(无 SFT/DPO/GRPO/RFT/Self-Instruct)
- 未推理(未运行 183 条评测)
- 未调用任何外部 API(无 GLM/Qwen/Judge)
- 未生成/修改任何数据(train/val/test、Gold Eval、stats 全部原样)
- 未修改模型/checkpoint/evaluator 判定逻辑/README/简历
- 唯一写入:`data_eng/PROJECT_FINAL_AUDIT.md`、`data_eng/INTERVIEW_FACTS.md`
