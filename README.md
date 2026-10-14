# 垂直领域客服助手:数据构建 → SFT → DPO / DAPO 对齐

一个有**真实业务场景**和**可量化指标**的大模型对齐项目:

虚构奶茶品牌"茶语时光"的智能客服 —— 从知识库出发,自建后训练数据管道,
在 Qwen2.5-0.5B 上完成 LoRA 微调(SFT)、DPO 偏好对齐、DAPO(GRPO 框架)强化学习对齐,
并对比 基座 / SFT / DPO / RL 在**领域正确率、拒答率、回答简洁度**上的差异。

**场景问题(项目主线)**:客服模型对知识库外问题会"硬答、编造答案"——
因此整个项目围绕一个问题展开:"**会的就答对、不会的拒绝答**",
数据、验证器、对齐方法全部为这个目标服务。

**数据质量主线**:项目中期发现早期 train/test 随机切分会让同一批问题的不同改写
同时进入训练和测试,导致指标虚高。据此重建了按样本血缘切分的 clean 数据管道,
并加入自动 leakage audit(详见第三节)。

项目的另一条线是**从零预训练的研究型底座**(1200 万参数 GPT,
RoPE / GQA / SwiGLU+RMSNorm 结构消融 + 参数量-验证损失缩放趋势)。

## 为什么这样做

- **有现实场景**:客服是垂直领域 LLM 最典型的落地场景,问题、答案、拒答边界都清晰可定义
- **数据是重点**:不下载现成数据集,自建"种子编写 → 模板扩展 → 清洗去重 → 血缘切分"
  的完整后训练数据管道,并支持规则奖励自动构造偏好对(DPO 数据)
- **对齐演进线完整**:SFT 之后分两条平行分支 —— DPO 偏好对齐与 DAPO 强化学习对齐
- **有指标结果**:同一 clean 评测集(零泄漏)上给出基座/SFT/DPO/RL 的对比表
  (正确率、拒答率、简洁度),并用规则命中 + LLM-as-Judge 双轨交叉验证
- **有思考深度**:文末讨论"后训练的收益 vs 工程化",用本项目自己的数字说话

## 方法与技巧

- **SFT 领域 + 通用数据混合**:按 0.2 比例混入通用指令数据(alpaca-zh),
  防止领域过拟合、保持通用对话能力;但实验观察到混合会明显拉低拒答率,
  说明客服场景对"拒答边界"敏感,混比需按场景调优(默认混合比例已调回 0)
- **on-policy 采样 + 验证器筛选(RFT)**:让 SFT 模型自采样回答,用规则验证器
  (关键实体命中/正确拒答/简洁度)筛出高质量数据;去重需按(问题,答案)对,
  否则同问题的不同回答会被误删
- **DPO 偏好对自动构造**:知识库金标准答案作 chosen、验证器低分采样回答作 rejected
- **GRPO 用 DAPO loss(TRL 默认)**:正负优势分开裁剪(clip-higher),缓解奖励饱和;
  截断掩码留作开关(实验中开启后领域正确率下降,保留供复现)
- **GSPO 风格序列级重要性采样做成了对照实验**(`--seq-level`,与 DAPO 仅 loss 配置不同):
  结果见第五节 —— 序列级 IS 把训练奖励推到接近饱和,但 held-out 关键词正确率未提升,
  说明重要性采样的粒度不是本项目的瓶颈
- **评测双轨制**:关键词命中(可复现)+ LLM-as-Judge(智谱 GLM 语义打分)互补,
  前者看事实命中、后者看回答质量;两者在本项目上给出过相反的排序,因此必须交叉验证

## 环境

- GPU:RTX 3050 Laptop(4GB 显存),PyTorch 2.7.1+cu118
- 工具链:transformers / datasets / accelerate / peft / trl / bitsandbytes
- 所有代码、虚拟环境、下载缓存都在 D:\LLM-Project 下

## 目录结构

```
data_eng/   后训练数据构建管道(知识库 + 种子 + 扩展 + 清洗 + 血缘切分 + 泄漏审计)
data_eng/output_clean/   clean 数据集与审计报告(当前评测口径)
data_eng/run_pipeline.py 全流程编排(逐步记录命令/耗时/产物 hash 到 runs/)
pretrain/   从零预训练微型 GPT(1200 万参数)+ 结构消融与参数规模对照
sft/        Qwen2.5-0.5B LoRA 监督微调 + 领域评测
rl/         DPO 与 GRPO/DAPO 对齐(规则奖励自动构造偏好对)
models/     训练产物(不入库)
runs/       全流程重跑留痕(日志、逐条预测、评测报告、manifest)
```

## 一、后训练数据构建(项目核心)

### 1. 知识库(kb.py)

虚构品牌"茶语时光"的完整业务资料:菜单与价格、小料加价、营业时间、外卖规则、
会员积分、优惠券、售后政策、过敏原、门店信息。所有答案都从知识库生成,保证事实一致。

### 2. 种子(seeds.py)

人工基于知识库编写 55 条高质量问答,覆盖 10 类意图(价格/推荐/定制/营业时间/
外卖/会员/售后/过敏原/门店/**知识库外拒答**),并标注答案关键实体。

### 3. 扩展 + 清洗 + 切分(build_dataset.py)

```powershell
python data_eng/build_dataset.py
```

- 模板 + 实体替换 + 前缀改写:55 条种子 → 1250 个唯一问题
- 多层扩展:Self-Instruct 蒸馏(`evolve.py`,智谱 GLM)+ on-policy 采样+规则验证器筛选
  (`rl/rft.py`,RFT)作为额外数据合并进 clean 数据集
- 清洗:标点规范化、按(问题,答案)对去重、长度过滤
- 切分:**按样本血缘 group 切分** —— 为每条样本建立 `seed_id` / `family_id`,
  同一 base question 的所有改写只能进同一个 split,再做意图覆盖校验
- 输出:`data_eng/output_clean/` 下 train/val/test.jsonl + `stats.json`
  + `leakage_audit.json` / `DATASET_LEAKAGE_AUDIT.md`(审计报告)

> `data_eng/output/` 是早期的随机切分版本,保留仅作历史对照:它的 183 条测试集里
> 有 **105 条与训练集同源**(77 条问题逐字相同),用它算出来的指标不能作为泛化能力证明。

## 二、领域 SFT

```powershell
python sft/train_sft.py --data-file data_eng/output_clean/train.jsonl --max-steps 600
```

NF4 4bit 量化 + LoRA(r=16,覆盖全部线性层,可训练参数 8.8M / 502.8M = 1.75%),
数据为 clean train(2533 条,按意图分层抽样 2000 条训练 600 步),产物保存到 models/sft。

## 三、领域评测(指标结果)

```powershell
python sft/eval_domain.py --data-file data_eng/output_clean/test.jsonl --prompt-format chat --adapters models/sft
python sft/eval_domain.py --data-file data_eng/output_clean/test.jsonl --prompt-format raw --adapters models/dpo models/rl
# base 模型会自动一起评测,无需单独指定
```

评测集:`runs/2026-10-12/data/clean/test.jsonl`(168 条,血缘隔离零泄漏)。
各模型按其**训练时的 prompt 格式**评测(base/SFT 用 chat,DPO/RL 用 raw),
基座两种格式都跑,作为格式敏感性对照。

### 为什么必须分层看

clean 切分是按样本血缘 group 做的 —— 同一个 `(实体, 意图)` 只能进一个 split。
好处是零泄漏,代价是**测试集里必然有一部分题所需的答案事实,训练集里从未出现**
(最典型的是"布丁推荐"那一族:60 条题,训练集里一条都没有)。
把这两类题混在一起算一个"领域正确率"既不能反映能力,也不能反映泛化,
所以按「本题判定所需关键词是否在同一 intent 的训练集中出现过」拆成三片分别报告:

| 切片 | n | 含义 |
|---|---|---|
| 熟场景新问法 | 77 | 该事实学过,只是换了个问法 |
| 未见过场景 | 76 | 该事实训练集里完全没有,模型无从得知 |
| OOD 拒答 | 15 | 知识库外问题,单独统计 |

**规则评测(关键词命中准确率)**

| 模型 | prompt | 熟场景新问法 | 未见过场景 | OOD 拒答 | 总体 |
|---|---|---|---|---|---|
| 基座 | chat | 14.3% | 6.6% | 6.7% | 10.1% |
| SFT(LoRA) | chat | 54.5% | 10.5% | 33.3% | 32.7% |
| 基座 | raw | 16.9% | 4.0% | 0.0% | 9.5% |
| DPO(100 步) | raw | 45.5% | 6.6% | 13.3% | 25.0% |
| DPO(200 步) | raw | 41.6% | 5.3% | 13.3% | 22.6% |
| RL(DAPO) | raw | **57.1%** | 5.3% | **46.7%** | 32.7% |
| RL(GSPO 序列级) | raw | 42.9% | 6.6% | 46.7% | 26.8% |

**LLM-as-Judge(语义均分,1-5)**

| 模型 | 熟场景新问法 | 未见过场景 | OOD 拒答 | 总体 |
|---|---|---|---|---|
| 基座(chat) | 2.90 | 3.24 | 2.93 | 3.05 |
| SFT(LoRA,chat) | **3.44** | 3.32 | **3.53** | **3.39** |
| DPO(200 步,raw) | 3.23 | 3.16 | 2.60 | 3.14 |
| RL(DAPO,raw) | 2.32 | 2.13 | 2.27 | 2.23 |
| RL(GSPO 序列级,raw) | 2.53 | 2.57 | 3.20 | 2.61 |

指标定义:关键词命中率 = 回答包含答案关键实体(判定器要求命中任一关键词);
拒答率 = 知识库外问题被正确拒绝;LLM-Judge = 智谱 GLM 对逐条预测的语义打分。

**怎么读这三片**

- 与 SFT/DPO/RL 训练目标直接对应的是「**熟场景新问法**」:DAPO 57.1%、SFT 54.5%,
  基座只有 14~17% —— 后训练把这一片提升了约 3.4 倍。
- 「未见过场景」普遍是个位数,这是**预期结果**:该事实没有任何训练信号,
  这一片应读作"模型的泛化/拒答边界",而不是"模型能力差"。
- 两片分开后,"总体正确率 32.7%"这种数字的迷惑性就消除了。
- 两种口径**多次给出相反排序**(如 GSPO 分支:关键词更低、Judge 更高),
  所以主指标必须交叉验证,不能只看其一。
- 分层报告可复现:`python sft/eval_stratified.py ...`(只读 predictions,不推理),
  输出见 `runs/2026-10-14/stratified_eval/`。

> 命名说明:`data_eng/output_clean/`(183 条)是更早一轮的 clean 数据集,
> 当时评测所用的模型是在**与它同源的旧训练集**上训的(183 条测试里 105 条与训练集同源),
> 因此那一轮的指标不作为结论。本节的 168 条数据集与全部模型都来自
> `runs/2026-10-12/` 这一轮端到端重跑,训练/测试血缘严格隔离。
> 逐条预测在 `runs/2026-10-12/eval/predictions/`;两次运行 base 预测逐字一致。

## 四、领域 DPO(偏好对齐)

```powershell
python rl/train_dpo.py
```

DPO 不需要奖励模型:对每条问题让 SFT 采样多个回答,用规则奖励打分,
以**知识库金标准答案**为 chosen、**规则奖励评分最低的采样回答**为 rejected
(仅当最低分低于金标准分时才保留该题),自动构造偏好对后做二分类式训练。
本项目的偏好对构造已批量采样化,并把逐题打分与保留情况落盘,便于审计。

对齐结构:**SFT 之后分两条平行分支**,DPO 与 RL 都从同一个最终 SFT 出发,
而不是串行链;三者在同一 clean 评测集上横向对比。

## 五、领域 RL 对齐(DAPO / GSPO 对照)

```powershell
python rl/train_grpo.py                    # 默认 domain 模式,从 SFT 继续
python rl/train_grpo.py --seq-level        # GSPO 风格:序列级重要性采样
python sft/eval_domain.py --adapters models/sft models/rl
```

实际配置:`loss_type=dapo`(TRL 默认:按全局有效 token 数归一化 + clip-higher 动态裁剪),
`importance_sampling_level=token`,300 步,每题 4 候选,`max_completion_length=64`。

奖励设计(纯规则):
- 知识库内问题:回答包含答案关键实体 +1,否则 -1
- 知识库外问题:正确拒答 +1,瞎编 -1
- 回答简洁(≤80 字符)+0.1

RL 的目标是让模型"会的就答对、不会的就拒答",而不是胡编 —— 这正是对齐的实际意义。

### GSPO 风格序列级重要性采样对照

`--seq-level` 会把配置切成 `loss_type=grpo` + `importance_sampling_level=sequence`
(TRL 文档中用于复现 GSPO 论文设置的那组配置)。以它为唯一变量做了 300 步对照,
起始适配器、数据、步数、seed、batch、lr、temperature、beta 全部一致:

| 指标 | DAPO(IS=token) | GSPO 风格(IS=sequence) |
|---|---|---|
| 训练奖励末值 | 0.8425 | **0.9975**(接近饱和) |
| held-out 总正确率 | 32.7% | 26.8% |
| held-out 拒答率 | 46.7% | 46.7% |
| LLM-Judge 均分 | 2.23 | **2.61** |
| mean_length / EOS 命中率 | 64 / 0% | 64 / 0% |

**结论**:序列级 IS 把训练奖励推到接近饱和,但 held-out 关键词正确率反而更低;
两种口径(Judge 与关键词)给出了相反的排序。说明(a)本项目的瓶颈不在重要性采样粒度上,
而更可能在 completion 长度预算、截断处理与组内奖励区分度
(同期 `frac_reward_zero_std ≈ 0.71`、`clipped_ratio ≈ 1.0`、
`mean_terminated_length ≈ 0`);(b)单一关键词指标不足以衡量对齐效果,必须与语义指标交叉验证。
完整对照见 `runs/2026-10-13/gspo_ablation/GSPO_VS_DAPO.md`。

## 六、从零预训练研究

```powershell
# 结构消融
python pretrain/train.py --max-steps 2000 --pos-encoding rope --out-dir models/exp_rope
python pretrain/train.py --max-steps 2000 --attention-type gqa --out-dir models/exp_gqa
python pretrain/train.py --max-steps 2000 --activation swiglu --norm-type rmsnorm --out-dir models/exp_swiglu
python pretrain/plot_results.py --scaling
```

已完成的结论(参数量持平的公平对比):
- 6000 步基线 val_loss 1.97,能生成连贯英文小故事
- **RoPE vs 学习式位置编码:val_loss 2.378 vs 2.515(-5.4%)**
- **GQA vs MHA:省 7.2% 参数,仅损失 0.4% 效果**
- **SwiGLU+RMSNorm vs GELU+LayerNorm:val_loss 2.436 vs 2.515(-3.2%)**
- **注意力实现:走 PyTorch SDPA(`F.scaled_dot_product_attention`,可调度 FlashAttention kernel),
  非自研 kernel,也没有做"用/不用"的对比实验**
- **参数规模趋势:0.95M / 4.26M / 12.29M / 27.4M 四档同数据同步数(各 2000 步)的
  val_loss 为 3.51 → 2.83 → 2.51 → 2.30。这是缩放**趋势**,本项目没有做幂律拟合,
  因此不称其为 scaling law**


仅作为可复现的代码路径保留,不计入结论;生成函数未实现 KV Cache,每步做全量前向。

