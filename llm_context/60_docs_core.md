# 项目核心文档（建议第一个上传 / 先读）

### `README.md`

````markdown
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

代码里另有 ALiBi、MQA、untied embedding 的开关分支,但**没有做实验**,
仅作为可复现的代码路径保留,不计入结论;生成函数未实现 KV Cache,每步做全量前向。

## 七、讨论:后训练的收益 vs 工程化


1. **后训练的直接收益(按分层口径看,clean 评测集 168 条零泄漏)**:
   - SFT:「熟场景新问法」14.3% → 54.5%,OOD 拒答率 6.7% → 33.3% ——
     从"不会对话"到"按知识库回答"
   - DPO:回答显著变短(平均 25.6 字符,SFT 为 40.1)、正常终止(EOS 命中率 81.5%),
     但「熟场景新问法」低于 SFT(41.6% vs 54.5%),
     说明偏好对齐在这个数据量下是取舍而非全面提升
   - RL(DAPO):「熟场景新问法」57.1%(全模型最高),OOD 拒答率升到 46.7%
     (基座 0.0%、SFT 33.3%)—— 后训练主要收紧了拒答边界,
     而对训练集未覆盖的场景几乎没有泛化(5.3%)
2. **指标口径比指标高低更重要**:项目早期用随机切分,测试集与训练集同源
   (183 条里 105 条同源),曾得到明显虚高的数字;改为血缘切分 + 自动 leakage audit 后,
   数字下修但可信。同样地,关键词口径与 LLM-Judge 在 GSPO 对照中给出**相反排序**,
   说明单一指标不足以衡量对齐效果。
3. **后训练对奖励设计与工程细节极其敏感**:本项目在 RL 阶段实际踩过三个坑 ——
   completion 截断处理、prompt 格式不一致(同一批权重在 chat 下 EOS 命中率 0%、
   在 raw 下 82%)、PEFT 参考适配器错位导致策略未真正更新;另外组内奖励区分度偏低
   (`frac_reward_zero_std ≈ 0.71`),使相对优势大量退化为 0。这些都不是算法问题,
   但都会让训练白跑。
4. **工程化同样重要**:4bit 量化、LoRA、PyTorch SDPA、评测自动化 —— 工程能力决定了
   模型能不能落地。后训练和工程不是二选一,是"模型能力"和"交付能力"两条腿。
````

### `INTERVIEW_MAP.md`

````markdown
# 《LLM-Project 大模型算法面试学习地图》

> 生成方式:通读 `data_eng/`、`sft/`、`rl/`、`pretrain/` 全部源码 + README + 两份审计文档 + 全部实验产物,
> 并对关键结论**用脚本重新算了一遍**(而不是照抄文档)。
> 标注 `【实测】` 的条目 = 本次重新验证过;`【文档】` = 来自仓库文档但未复算。
>
> 使用方式:本文**不替你写答案**。它告诉你「会被问什么 / 为什么问 / 要看哪行代码 / 学到什么深度」。
> 你按第七部分的学习顺序,逐个问题来找我练。

---

## 0. 先读这一段:项目的真实状态(决定你讲什么主线)

这个项目做过一次很彻底的自我审计(`data_eng/PROJECT_FINAL_AUDIT.md`、`INTERVIEW_FACTS.md`),
文档质量高于绝大多数实习项目。**但审计之后又跑出了新结果,导致部分审计结论已过时;
而且有几条最关键的事实,审计文档自己也漏了。**

### 0.1 我重新算出来的、审计文档里没有的事实(最重要)

| # | 事实 | 证据 | 严重性 |
|---|---|---|---|
| 1 | **clean test 183 条里,77 条与「旧 train.jsonl」精确同题** | 【实测】`old train`(2275)∩`clean test` unique exact q = 77 | 🔥 致命 |
| 2 | 非 gold_eval 的 105 条测试题,**105/105 去前缀后与旧 train 同源** | 【实测】base question overlap = 105/105 | 🔥 致命 |
| 3 | **78 条人工 gold_eval 与旧 train 精确/去前缀重叠 = 0** | 【实测】 | ✅ 唯一可信子集 |
| 4 | sft / dpo / rl 三个 checkpoint **全部是用旧 train 训的** | 【文档】audit 第三节 + 默认 `--data-file data_eng/output/train.jsonl` | 🔥 致命 |
| 5 | 结论:clean test 上 **只有 78 条 gold_eval 的数字可以引用** | 由 1+3+4 推出 | — |
| 6 | 最终对比表 `eval_final_matrix.md` 里 **sft 用 chat 格式,其余 4 个用 raw 格式** | 【实测】predictions 元数据 `prompt_format` 字段 | 🔥 |
| 7 | 表里叫 `checkpoint-200` 的**不是 GRPO,是 DPO 的 200 步 checkpoint** | 【实测】`models/dpo/checkpoint-200` 存在;`models/grpo/checkpoint-200` 已被删除;报告名 `eval_final_raw_dpo200_rl.md` | 🔥 |
| 8 | **标准 GRPO(`models/grpo`)从未进入任何对比表** | 【实测】predictions 里没有 grpo 模型 | 🟠 |
| 9 | 评测 `max_new_tokens=32`,但 RL 训练时 `max_completion_length=64` | 【实测】 | 🟠 |
| 10 | LLM-as-Judge 给 **11 条空回答全部打了 4 分** | 【实测】dpo 4 条 + checkpoint-200 7 条 | 🟠 |
| 11 | `models/dpo` 根目录 = **10 步** DPO(不是 200 步) | 【实测】mtime 与 checkpoint-10 一致,audit 有 sha256 | 🔥 |

### 0.2 唯一可信的那组数字(78 条 gold_eval 子集)

从 `eval_final_matrix.md` 的 gold_eval 分段读出:

| 模型 | 领域正确率 | 拒答率 | 总正确率 | 备注 |
|---|---|---|---|---|
| base | 19.1% | 0.0% | 16.7% | n=78 |
| sft | 58.8% | 0.0% | 51.3% | 600 步 LoRA |
| dpo(=**10 步**) | 60.3% | 0.0% | 52.6% | 与 sft 差 1.3pt |
| checkpoint-200(=**DPO 200 步**) | 44.1% | 40.0% | 43.6% | EOS 93.6% |
| rl(=DAPO 300 步) | 38.2% | 30.0% | 37.2% | **低于 sft** |

**这张表你必须在面试前背下来,并且主动讲。** 因为它说明三件事:
1. SFT 确实有效(16.7% → 51.3%);
2. DPO/RL 在**真正干净**的题上没有带来净收益,RL 甚至更差;
3. 你没有被自己的旧数字骗到。

⚠️ 但注意:gold_eval 里 OOD 只有 **10 题**,所以"拒答率 40%/30%/0%"的 n=10,**统计上不可信**。
被追问时必须自己先说这一点。

### 0.3 主线建议

**不要再讲"准确率 92%"。** 换成:

> "这个项目我最有价值的工作不是把指标做高,而是**发现并量化了评测泄漏**:
> 我发现最终 checkpoint 的训练集和测试集有 77/183 条完全同题,于是把测试集拆成
> '可信子集(78条人工题)'和'污染子集(105条)',并如实报告——
> 在可信子集上我的 RL 反而比 SFT 差,这是真实的负结果,我没有美化它。"

这条主线经得起追问,因为它**主动暴露了对你不利的证据**——面试官很难再攻击你。

---

# 第一部分:项目地图(A–L)

## A. 数据构建
- **技术**:知识库 → 人工种子 → 模板×实体扩展 → 口语化前缀改写 → 去重 → 切分
- **核心知识**:指令数据构造、模板增强的多样性与偏差、intent 分层、数据规模 vs 质量
- **文件**:`data_eng/kb.py`(知识库)、`data_eng/seeds.py`(55 条种子/10 intent)、`data_eng/build_dataset.py`
- **核心代码**:`build_dataset.py:338 _base_samples()`(种子+模板)、`:487 expand_seeds()`(前缀改写)
- **真实数字**:55 seed → 256 基础问题 → ×5 前缀 → 最终 clean 版 2924 条(train 2676 / val 65 / test 183)
  - 其中 on-policy 自生成 **1480 条(占 train 55%)**、GLM 蒸馏 86 条、人工 gold_eval 78 条
- **面试官会问**:为什么不用现成数据集?模板扩展的多样性上限在哪?55 条种子够吗?
- **要理解到**:能说清"我的数据里只有约 1200 条是人工/模板 gold,超过一半是模型自己生成的"

## B. 数据清洗 / 去重 / 防泄漏 ★项目最强项★
- **技术**:`(q,a)` 对级去重、长度过滤、**样本血缘(seed_id/family_id/entity_key)**、**并查集 group 级切分**、6 层泄漏审计
- **核心知识**:数据泄漏的层次(exact / base / family / entity)、group 切分 vs 样本级随机切分、可复现性
- **文件**:`data_eng/build_dataset.py`、`data_eng/leakage_audit.py`、`data_eng/output_clean/DATASET_LEAKAGE_AUDIT.md`
- **核心代码**:
  - `build_dataset.py:716 dedup_and_filter()` — 按 (问题,答案) 去重
  - `:737 class UnionFind` + `:760 build_groups()` — 血缘连通分量
  - `:822 split_groups()` — group 级切分
  - `:982 audit_gold_eval()` — 逐题泄漏检查
- **修复前 vs 后**:旧集 188/288 同题、277/288 同源、13/13 OOD 同源、test 混入 157 条模型生成数据 → clean 版全部为 0
- **面试官会问**:为什么按 (q,a) 而不是按 q 去重?family 和 group 的区别?group 切分怎么保证可复现?
- **要理解到**:能手画并查集合并过程;知道 `SPLIT_TARGETS` 常量实际未使用(切分比例由 clean group 数量倒推)

## C. SFT
- **技术**:4bit NF4 QLoRA + LoRA(r16/α32/dropout .05,7 个投影层)、chat template、assistant-only loss mask
- **文件**:`sft/train_sft.py`
- **核心代码**:
  - `:97 bnb_config`(4bit NF4 + double quant,compute dtype **fp16**)
  - `:109 LoraConfig`
  - `:153 tokenize_fn()` — **重点**:`mask_len = min(len(prompt_ids), len(full_ids))`
  - `:175 TrainingArguments`(600 步,batch1×accum8,lr 2e-4,cosine+3% warmup)
- **真实配置**:600 步 / 2000 条(旧 train 前 2000 行,**其中 OOD = 0 条**)/ train_loss 0.7875→0.5167
- **可训练参数**:8,798,208(≈8.80M),336 个张量,占基座 ≈1.8%
- **必须能回答**:为什么 fp16 而 DPO/GRPO 用 bf16?为什么 4bit 还要 `prepare_model_for_kbit_training`?LoRA 缩放系数 α/r=2 意味着什么?

## D. DPO ★危险区★
- **技术**:TRL `DPOTrainer`,`loss_type=sigmoid`,beta=0.1,max_length=128
- **文件**:`rl/train_dpo.py`
- **核心代码**:
  - `:65 build_preference_pairs()` — **:86-92 chosen = `r["a"]`(知识库 gold),不是采样最高分**
  - `:54 score_response()` — 与 GRPO 同款规则分
  - `:134 DPOConfig`(beta=0.1)
  - `:149 DPOTrainer(...)` — **没传 ref_model**
  - `:156 model.set_adapter("default")`
- **reference model 真相**【实测】:TRL 1.9.2 在 PEFT + `ref_model=None` 时会 `model.add_adapter("ref", default_config)`
  并把 default 的权重复制过去 → **ref = 训练开始时的 SFT 策略**(正确做法)。
  源码位置:`.venv/Lib/site-packages/trl/trainer/dpo_trainer.py` 中 `add_adapter("ref", ...)` 与
  `use_adapter(..., adapter_name="ref")`。
- **坑**:`models/dpo` 根目录是 **10 步**版本;偏好对**不落盘**;docstring/README 写"最高分当 chosen"与代码冲突

## E. RL / DAPO / GRPO ★最危险区★
- **技术**:TRL `GRPOTrainer` + `loss_type="dapo"`、`importance_sampling_level="token"`、规则奖励
- **文件**:`rl/train_grpo.py`(DAPO)、`rl/train_grpo_standard.py`(标准 GRPO baseline)
- **核心代码**:
  - `train_grpo.py:133 domain_reward()` — 非 OOD 命中 +1 否则 −1;OOD 含拒答词 +1 否则 −1
  - `:128 format_reward()` — ≤80 字 +0.1
  - `:89 make_domain_dataset()` — **OOD 上采样到 20%**
  - `:179 GRPOConfig` — 300 步,lr 1.5e-4,beta(KL)=0.04,temperature 0.7,`mask_truncated_completions=False`
  - `:200 loss_type` / `:201 importance_sampling_level`
- **奖励**:对称 ±1 + 简洁 0.1,**纯规则,不是 reward model**
- **训练日志**:末步 reward≈0.8275,domain_reward 0.75,`clipped_ratio=1.0`、`mean_terminated_length=0.0`
- **GSPO**:只有 `--seq-level` 开关,**从未运行**,TRL 1.9.2 也没有 `gspo` loss → **简历绝对不能写**

## F. Evaluation
- **技术**:规则关键词命中、intent 分层统计、gold_eval 单独统计、逐条 prediction 落盘 + 离线复算
- **文件**:`sft/eval_domain.py`
- **核心代码**:
  - `:146 judge_row()` — `hit = any(k in resp for k in keywords)`;`refused = any(m in resp for m in REFUSAL_MARKERS)`
  - `:23 REFUSAL_MARKERS = ["抱歉","不好意思","无法回答","超出","服务范围","咨询"]`
  - `:301 aggregate_records()`
  - `:30 默认 --data-file = data_eng/output/test.jsonl`(**旧泄漏集**)但默认报告名是 `..._clean.md` ← 陷阱
- **已知缺陷**(文档已记录,你要能复述):无事实校验、数字无词边界(`"12"` 命中 `"2012"`)、拒答词过宽/位置无关、复读也算对
- **实测反例**:`eval_final_chat.md` 里 sft 回答"外卖洒了,是找骑手还是找客服? 问题1:外卖洒了…"被判**命中✓**(复读题目即命中)

## G. LLM-as-Judge
- **技术**:智谱 `glm-4-flash`,1–5 分,把【标准答案】一起喂给裁判
- **文件**:`sft/eval_domain.py:166 judge_llm()`、`data_eng/run_llm_judge.py`
- **最新结果**【实测】:`output_clean/llm_judge/summary.md`,clean 183 全量,5 个模型
  - base 2.61 / sft 3.09 / rl 3.45 / dpo 3.55 / **checkpoint-200 4.02**
- **重大不一致**:
  - 规则评测排名:dpo(61.2%) > checkpoint-200(60.7%) > rl(59.6%) > sft(48.1%)
  - Judge 排名:**checkpoint-200(4.02) > dpo(3.55) > rl(3.45) > sft(3.09)**
  - **排名翻转**
  - rl 规则拒答率 88.3%,Judge 判 OOD 正确拒答只有 **41.7%**
  - **Judge 给 11 条空回答全部打 4 分**【实测】
- **结论**:两条轨道都不能单独作为结论,你必须能解释分歧来源

## H. GPT from scratch
- **技术**:12.29M decoder-only Transformer,Pre-LN,tied embedding,自写 RoPE/GQA/SwiGLU/RMSNorm
- **文件**:`pretrain/model.py`、`pretrain/config.py`、`pretrain/train.py`
- **核心代码**:
  - `model.py:157 _apply_rope()` — **半分(half-split / NeoX 风格)**旋转,不是 interleaved
  - `:96-104` RoPE 预计算 cos/sin(`emb = cat((freqs, freqs))`,与半分风格自洽)
  - `:132-135` **GQA 用 `repeat_interleave` 把 KV 复制回全头数** ← 只省参数,不省 attention 计算
  - `:147 F.scaled_dot_product_attention(..., is_causal=True)` ← **是 SDPA,不是自研 FlashAttention**
  - `:171-176` SwiGLU 用 `int(4*n_embd*2/3)` 宽度对齐 GELU 参数量
  - `:45 RMSNorm`、`.:224` tied embedding
  - `:257 generate()` — **无 KV cache**,每步重算全上下文

## I. Transformer 结构消融
- **实测结果**:
  - RoPE 2.3779 vs learned 2.5148(−5.4%),参数 12.19M vs 12.29M(RoPE 少一个 256×384 位置表)
  - GQA 11.41M vs 12.29M(**−7.2% 参数**),val loss 2.5261 vs 2.5148(+0.4%)
  - SwiGLU+RMSNorm 2.4356 vs 2.5148(−3.2%)
- **坑**:
  - `exp_learned` 与 `exp_12m` 是**同一配置的两次运行**(n_params 与末次 val_loss 完全相同),不能算两个独立点
  - SwiGLU 和 RMSNorm **一起做的**,没有单独消融
  - **没有 KV cache**,所以"GQA 省缓存显存"没测到
  - ALiBi / MQA / untied embedding 只有代码,**没跑过**

## J. 实验设计与 ablation
- **做过的**:架构消融 3 组、规模趋势 4 档、prompt-format 诊断消融(40 次生成)、通用数据混合 0.2(历史)、截断掩码开关(历史)
- **没做的**:标准 GRPO vs DAPO 的**最终对比**(GRPO 跑完了但没评测)、GSPO、reward 权重扫描、SFT 步数消融、seed 多次重复跑
- **样本量问题**:prompt ablation 是 **n=5**;gold_eval OOD 只有 **n=10** → 都不能下强结论

## K. 工程实现
- 4bit 量化 + LoRA + gradient checkpointing + `use_cache=False`
- 逐条 prediction 落盘 + `--from-predictions` 离线复算(很好的工程点,可以讲)
- 实时进度/ETA、慢题告警、stop_ids 自动解析(`resolve_stop_ids`)
- PEFT `ref` adapter 陷阱:`train_dpo.py:156` / `train_grpo.py:214` 的 `model.set_adapter("default")`
- 断点续训适配器的坑:见 `data_eng/GRPO_EXPERIMENT_ARCHIVE.md`(KL 从 0.43 掉到 0.009)

## L. 项目不足与风险(汇总)
见「第六部分:最危险的 10 个问题」。

---

# 第二部分:六层追问清单

> 我只列**问题**,不给答案。答案要你自己按第五部分的方法去补。

## A/B. 数据
- L1:什么是 instruction tuning 数据?为什么要做数据去重?
- L2:为什么不下载现成中文指令数据集?55 条种子怎么保证覆盖?10 个 intent 怎么定的?
- L3:模板扩展带来的"伪多样性"是什么?为什么同一 base question 的改写跨 split 就是泄漏?
- L4:`build_dataset.py:760 build_groups()` 里并查集把哪几类节点 union 了?`entity_key` 怎么推断(`:314 _infer_entity_key`)?`split_groups()` 的 Phase A / Phase B 分别保证什么?`clean_test_ratio=0.6` 是什么意思?
- L5:为什么 clean train 只有 2676 却不能做 80/10/10?为什么 clean group 全给 val/test?为什么 evolved/onpolicy 一律只进 train?
- L6:**"你说 clean test 零泄漏,那为什么 77 条测试题和旧训练集完全同题?"** ← 见 §0.1

## C. SFT
- L1:SFT 的 loss 是什么?为什么只对 assistant 部分算 loss?
- L2:为什么用 Qwen2.5-0.5B?为什么 LoRA r=16?为什么 7 个投影层全挂?
- L3:LoRA 为什么能降参数?α/r 缩放的作用?为什么 4bit 量化后还能训?NF4 是什么?double quant 是什么?
- L4:`train_sft.py:153 tokenize_fn()` 里 `mask_len = min(len(prompt_ids), len(full_ids))` 有没有 off-by-one 风险?
  为什么 `full` 和 `prompt` 分开 tokenize 可能错位?`attention_mask` 全是 1,那 padding 怎么处理?
- L5:600 步 / 2000 条够吗?为什么前 2000 行 OOD=0 条?这怎么解释"拒答学不会"?
- L6:"你的 SFT 数据 55% 是模型自己生成的,这不就是自我蒸馏吗?能带来新信息吗?"

## D. DPO
- L1:DPO 是什么?和 RLHF 什么关系?
- L2:你的 chosen/rejected 怎么来的?为什么 beta=0.1?
- L3:DPO 为什么不需要显式 reward model?loss 怎么从 Bradley-Terry 推出来?reference model 的作用?
- L4:`train_dpo.py:86-92` 到底把什么当 chosen?和 docstring 为什么不一样?
  `:149 DPOTrainer` 没传 ref_model,那 ref 是什么?`:156` 为什么要 `set_adapter("default")`?
  `score_response` 的分数域是什么(可能取哪些值)?`worst_score < gold_score` 会过滤掉多少?
- L5:200 步和 10 步两个版本差多少?为什么 models/dpo 是 10 步?
- L6:**"你的 DPO 只训了 10 步(或 200 步),这个结论可靠吗?"** / **"偏好对没落盘,你怎么复现?"**

## E. GRPO / DAPO
- L1:PPO / GRPO / DAPO 的关系?为什么 GRPO 不需要 critic?
- L2:为什么用规则奖励而不是 reward model?为什么 OOD 上采样到 20%?
- L3:group relative advantage 怎么算?为什么除以组内 std?KL 约束在哪?DAPO 的 clip-higher 改了什么?
  token-level 和 sequence-level 重要性采样的区别?长度偏差是什么?
- L4:
  - `train_grpo.py:133 domain_reward()` 的签名为什么是 `(prompts, completions, answer_keywords, ood, **kwargs)`?
    `answer_keywords` / `ood` 是怎么从 dataset 传进来的?(提示:GRPOTrainer 把 dataset 的额外列当 kwargs 透传)
  - `:179 GRPOConfig` 里 `num_generations=4`、`batch=2`、`grad_accum=4` → 一个 optimizer step 用几个 prompt、几个 completion?
  - `beta=0.04` 对应论文里的哪一项?
  - `mask_truncated_completions=False` 但 `:218` 打印"截断掩码已开启" ← **代码与日志矛盾**
  - `:200` `loss_type="grpo" if args.seq_level else args.loss_type` 为什么要强制改?
- L5:为什么 reward 上升到 0.83 但领域正确率反而下降?`clipped_ratio=1.0` 说明什么?
- L6:**"你说用了 DAPO,TRL 的默认 loss_type 本来就是 dapo,你到底改了什么?"**
  / **"你的标准 GRPO 跑完了为什么没进对比表?"**

## F/G. 评测
- L1:什么是 keyword hit rate?拒答率怎么定义?
- L2:183 条 clean test 怎么来的?为什么还要 78 条 gold_eval?
- L3:为什么需要 OOD?OOD 的拒答边界怎么定义才合理?
- L4:`eval_domain.py:146 judge_row()` 两行判定有什么漏洞?`resolve_stop_ids()` 为什么不硬编码 151645?
  `--data-file` 默认是旧集但 `--report` 默认 clean,这个默认值组合有什么风险?
- L5:规则评测说 dpo 最好、Judge 说 checkpoint-200 最好,你信哪个?
  rl 规则拒答 88.3% 而 Judge 只认 41.7%,差在哪?
- L6:**"Judge 给空回答打 4 分,你的 Judge 还有效吗?"** / **"你的对比表里 sft 用 chat、其他用 raw,这公平吗?"**

## H/I. from scratch
- L1:Pre-LN 和 Post-LN 区别?为什么要残差?
- L2:RoPE 怎么实现?GQA 怎么实现?
- L3:RoPE 为什么能表达相对位置?为什么外推更好?GQA 为什么省 KV cache?
- L4:
  - `model.py:157 _apply_rope()` 是 interleaved 还是 half-split?和 `:102 emb=cat((freqs,freqs))` 是否自洽?
  - `:134 repeat_interleave` 之后,K/V 的 shape 是多少?**这样 GQA 还省计算吗?**
  - `:173 swiglu_width = int(4*n_embd*2/3)` 为什么是 8/3?参数量怎么和 GELU 对齐?
  - `:53 RMSNorm.forward` 的 eps 加在 sqrt 里面还是外面?和标准实现一样吗?
  - `:257 generate()` 的复杂度是多少?为什么没有 KV cache?
- L5:为什么 4 个规模点不能叫 scaling law?`exp_learned` 和 `exp_12m` 能不能算两个点?
- L6:**"你说用了 FlashAttention,代码里具体哪一行?"** / **"GQA 省 KV cache 你测到了吗?"**

---

# 第三部分:危险等级排序

排序原则:**这个问题能不能判断我是不是真的做过**。

## 🔥 P0(答不上 = 直接暴露)

| # | 问题 | 为什么是 P0 |
|---|---|---|
| P0-1 | 你的 clean test 183 条,和训练 SFT/DPO/RL 用的旧 train 有多少条同题? | 答不出 = 你根本没意识到自己的评测是脏的 |
| P0-2 | 那 78 条 gold_eval 上的数字是多少?和 183 条的数字差多少? | 这是唯一可信指标,必须脱口而出 |
| P0-3 | DPO 的 chosen 到底是什么? | 代码是 gold,README 说最高分——说错就说明你没读代码 |
| P0-4 | 你用了 DAPO,具体改了什么?TRL 默认不就是 dapo 吗? | 必须承认"默认即 dapo,我显式指定并理解了差异" |
| P0-5 | reward 函数具体怎么写的?OOD 上采样多少?beta 多少? | 三个数字必须准确:±1 / 20% / 0.04 |
| P0-6 | 你的对比表里各模型的 prompt 格式一致吗? | 不一致(sft=chat,其他=raw),必须主动讲 |
| P0-7 | models/dpo 是多少步?checkpoint-200 是哪个模型的? | 10 步 / 是 DPO 不是 GRPO,答错就是没看过产物 |
| P0-8 | 你的规则评测有什么已知误判?举一个例子 | 复读被判对、21:30 被判对——必须能举出实例 |
| P0-9 | 为什么选 Qwen2.5-0.5B + LoRA?可训练参数多少? | 4GB 显存 + 8.80M(≈1.8%) |
| P0-10 | group 级切分到底怎么做的?并查集合并了哪些节点? | 项目最强项,讲不清 = 不是你做的 |

## 🟠 P1(大概率问)

P1-1 DPO 为什么不需要 reward model(ref 是什么?beta 作用?)
P1-2 GRPO 的 advantage 怎么算?为什么不用 critic?
P1-3 你的 RL 为什么拒答率涨、正确率跌?怎么解释这个 trade-off?
P1-4 `mask_truncated_completions` 你开了吗?`clipped_ratio=1.0` 意味着什么?
P1-5 Judge 和规则评测为什么排序不一致?你信谁?
P1-6 on-policy(RFT)数据是怎么生成的?它和 SFT 训练格式一致吗?
P1-7 你的 SFT loss mask 怎么做的?有 off-by-one 风险吗?
P1-8 RoPE 你是 interleaved 还是 half-split?
P1-9 GQA 在你的实现里真的省了计算吗?
P1-10 为什么叫 scaling trend 不叫 scaling law?
P1-11 gold_eval 的 OOD 只有 10 题,你的拒答率可信吗?
P1-12 reward hacking 你怎么防的?做了哪些、哪些没做?

## 🟡 P2(可能追问)
DPO 的 max_length=128 够吗 / 4bit NF4 与 double quant 原理 / `prepare_model_for_kbit_training` 做了什么 /
cosine schedule + warmup 3% / tied embedding 的利弊 / SwiGLU 8/3 宽度推导 / RMSNorm vs LayerNorm 计算量 /
ALiBi 斜率公式 / `exp_learned` 与 `exp_12m` 重复问题 / prefix paraphrase 的 5 种模板 / 通用数据混合 0.2 的负结果

## 🟢 P3(了解即可)
ALiBi / MQA / untied embedding 的代码分支 / KV cache 的实现思路 / GSPO / PPO 完整推导 /
BPE tokenizer / flash attention kernel 细节

---

# 第四部分:项目面试知识树(按实际代码调整版)

```
LLM-Project
├── 数据(★最强项)
│   ├── kb.py: 虚构知识库(菜单/价格/小料/营业/外卖/会员/售后/过敏原/门店)
│   ├── seeds.py: 55 条人工种子 × 10 intent
│   ├── 扩展: 模板 × 实体 → 前缀改写 ×5 (PARAPHRASE_TEMPLATES)
│   ├── 额外来源
│   │   ├── evolved (Self-Instruct / GLM) 86 条 —— 无调用日志, PARTIAL
│   │   ├── onpolicy (RFT 拒绝采样) 1521 条 —— 只覆盖 price/recommend
│   │   └── gold_eval 78 条人工题 —— 只进 test
│   ├── 血缘: seed_id / family_id(intent+base_q) / entity_key / lineage_confidence
│   ├── 切分: UnionFind group → split_groups()  ← 核心
│   └── 审计: leakage_audit.py 6 层 + 18 assertion + sha256 可复现
│
├── SFT (sft/train_sft.py)
│   ├── 4bit NF4 QLoRA (bnb) + prepare_model_for_kbit_training
│   ├── LoRA r16 α32 7 投影层 → 8.80M (1.8%)
│   ├── causal LM / cross_entropy / next-token prediction
│   ├── assistant-only mask  ← labels[:mask_len] = -100
│   └── 600 步 / 2000 条 / lr2e-4 / fp16  ← 注意 OOD=0 条
│
├── DPO (rl/train_dpo.py)
│   ├── preference data: 在线构造, 不落盘
│   ├── chosen = gold 答案 / rejected = 规则分最低采样  ← 与 docstring 冲突
│   ├── DPO objective: sigmoid loss, beta=0.1
│   ├── reference model = TRL 自动创建的 "ref" adapter (复制自 SFT)
│   └── 产物: checkpoint-200 (200步) / checkpoint-10 (10步) = models/dpo 根目录
│
├── RL (rl/train_grpo.py, train_grpo_standard.py)
│   ├── 算法族: GRPO (TRL GRPOTrainer)
│   ├── DAPO: loss_type="dapo" (TRL 默认!) + clip-higher
│   ├── 标准 GRPO: loss_type="grpo" + importance_sampling_level="token"  ← 跑了但没评测
│   ├── GSPO: --seq-level 开关, 从未运行, 不能写简历
│   ├── reward: domain_reward(±1) + format_reward(≤80字 +0.1)  ← 纯规则
│   ├── advantage: TRL 组内归一化 (无需 critic)
│   ├── KL: beta=0.04 (对 SFT ref adapter)
│   └── 采样: num_generations=4, max_completion=64, temp 0.7, OOD 上采样 20%
│
├── Evaluation (sft/eval_domain.py)
│   ├── test 183 = 78 gold_eval(可信) + 105(与旧 train 100% 同源, 污染)
│   ├── rule evaluator: keyword 子串命中 / refusal marker
│   ├── LLM-as-Judge: glm-4-flash 1-5 分, 给【标准答案】
│   ├── 双轨冲突: 规则→dpo 最好; Judge→checkpoint-200 最好
│   └── 工程: 逐条落盘 + --from-predictions 离线复算 + stop_ids 自动解析
│
└── GPT from scratch (pretrain/)
    ├── 12.29M decoder-only, Pre-LN, vocab 4096, block 256, 6 层 6 头
    ├── RoPE (half-split 手写) — REAL
    ├── GQA/MQA (repeat_interleave) — GQA REAL, MQA 只有代码
    ├── SwiGLU (8/3 宽度对齐) + RMSNorm — REAL, 但两者未分离消融
    ├── SDPA (F.scaled_dot_product_attention) — 不是自研 FlashAttention
    ├── tied embedding — REAL (untied 只有开关)
    ├── 无 KV cache — generate 每步重算
    └── scaling: 0.95/4.26/12.29/27.4M → 3.5068/2.8285/2.5148/2.3005 (4点, 无幂律拟合)
```

---

# 第五部分:P0/P1 的学习路径(理论 / 代码 / 实验 / 表达)

> 每个问题给"要学什么 + 看哪行代码 + 看哪个实验 + 最后怎么讲",**不给完整答案**。

### P0-1 泄漏与可信子集
- **理论**:数据泄漏的层次(exact / normalized / base / family / entity);为什么"同实体不同场景"不算泄漏;测试集可信度的判据
- **代码**:`data_eng/build_dataset.py:760 build_groups()`、`:822 split_groups()`、`:982 audit_gold_eval()`
- **实验**:`output_clean/leakage_audit.json`(18 assertion)、`DATASET_LEAKAGE_AUDIT.md`、以及你自己重算的 77/183
- **表达**:先承认旧集脏 → 给出量化数字(188/288、77/183)→ 讲修复机制(血缘+并查集+group 切分)→
  **主动说代价**:"即便修完,我最终 checkpoint 仍是旧数据训的,所以只剩 78 条可信" → 收尾"我宁可报低但可信"

### P0-3 DPO 的 chosen/rejected
- **理论**:Bradley-Terry 模型、RLHF 目标 → DPO 的闭式解、隐式奖励
- **代码**:`rl/train_dpo.py:65-98`(尤其 `:86-92`)、`:54 score_response`、`:149-156`
- **实验**:两种 DPO 步数(10 / 200)在 gold_eval 上的差:52.6% vs 43.6%
- **表达**:先讲 DPO 思想 → 再讲 loss → 再讲**本项目 chosen 用的是知识库 gold 而不是采样最优** →
  主动指出 "README 那句话写错了,代码是这样" → 讲这带来的偏差(gold 是 off-policy 的)

### P0-4 DAPO 到底改了什么
- **理论**:PPO 的 clip、GRPO 的 group advantage、DAPO 的 4 个改动(clip-higher、dynamic sampling、token-level loss、overlong 惩罚)
- **代码**:`rl/train_grpo.py:196-204`、`train_grpo_standard.py:160-167` 的对照
- **实验**:`models/rl/training_log.json`(DAPO)vs `models/grpo/training_log.json`(标准 GRPO)
- **表达**:**必须诚实**——"TRL 的 GRPOConfig 默认 loss_type 就是 dapo,所以我显式指定它,
  并额外写了一个标准 GRPO baseline 脚本做对照;但**这个 baseline 最终没进对比表**,这是我的缺口"

### P0-5 reward 设计
- **理论**:可验证奖励(verifiable reward)、奖励稀疏性、reward hacking、KL 的作用
- **代码**:`train_grpo.py:133-145`、`:128-130`、`:89-110`(OOD 上采样)
- **实验**:末步 reward 0.8275 / domain 0.75 / format 0.078;`clipped_ratio=1.0`
- **表达**:讲清为什么用对称 ±1(而不是 0/1)、为什么简洁只给 0.1、为什么 OOD 上采样 →
  然后**主动讲副作用**:"对称奖励 + 上采样让模型偏向拒答,clean 子集上领域正确率确实掉了"

### P0-10 group 级切分
- **理论**:并查集、连通分量、分层抽样 vs 分组抽样
- **代码**:`build_dataset.py:737 UnionFind`、`:760 build_groups()`、`:822 split_groups()`(Phase A/B)
- **实验**:`split_groups.json`(每个 group 的 size/clean/split)
- **表达**:画一遍"同 family / 同实体 / 同 seed / 歧义候选"四条 union 规则 → 讲 clean 判定的定义 →
  讲为什么 80/10/10 做不到

### P1-9 GQA 是否真的省计算
- **理论**:MHA/MQA/GQA 的参数量与 KV cache 公式、SDPA 的 `enable_gqa`
- **代码**:`pretrain/model.py:132-135`
- **实验**:`models/exp_gqa` 参数 11.41M vs 12.29M
- **表达**:先讲机理(KV cache 按 KV 头数缩小)→ 再讲**我的实现用 repeat_interleave 把 KV 复制回全头,
  所以 attention 计算量与 MHA 相同,我只实测到了参数量 −7.2%**

---

# 第六部分:最危险的 10 个问题

### ❶ 最终 checkpoint 的评测是脏的,但报告叫 "clean"
- **事实**:sft/dpo/rl 都用 `data_eng/output/train.jsonl` 训练;该文件与 clean test 有 **77/183 精确同题**、
  非 gold_eval 的 105 条 **105/105 去前缀同源**【实测】
- **位置**:`rl/train_dpo.py:31`、`rl/train_grpo.py:56` 的默认 `--data-file`
- **为什么危险**:面试官只要算一次 overlap 就全穿
- **他会问**:"你说 clean split 零泄漏,那你的模型是在 clean train 上训的吗?"
- **诚实答法**:"不是。clean split 保证的是**新数据管道**的 train/test 不相交;
  但我的三个 checkpoint 是在修复前的旧 train 上训的,所以它们在 clean test 上仍有 42% 同题。
  我把它拆成可信的 78 条人工题和污染的 105 条,只引用前者。"
- **要补学**:train/test 泄漏的所有层次;如何设计一个"训练数据也干净"的复训实验

### ❷ README 与代码直接冲突(DPO chosen)
- **事实**:README §四 与 `train_dpo.py:10` docstring 写"最高分当 chosen";代码 `:86-92` 用 gold 答案
- **为什么危险**:你若按 README 答,面试官翻代码立刻发现
- **他会问**:"你的 chosen 到底是采样最优还是知识库答案?"
- **诚实答法**:"是知识库 gold。我之前文档写错了,后来审计时发现并记进了 INTERVIEW_FACTS Q3。
  这个选择有代价:gold 是 off-policy 的,与 policy 分布不一致。"
- **要补学**:on-policy vs off-policy 偏好数据;DPO 对分布偏移的敏感性

### ❸ `train_grpo.py` 的 docstring 完全过期 + 日志与配置矛盾
- **事实**:文件头 docstring 讲"两位数加减法 / 答案是…+0.5",但默认 `--mode domain`;
  `:218` 打印"截断掩码已开启" 而 `:202` 是 `mask_truncated_completions=False`
- **为什么危险**:面试官打开文件第一眼就看到
- **诚实答法**:"docstring 是早期 math 模式留下的,没同步更新,这是我的工程债;
  打印那行也是个 bug,实际配置是 False。"
- **要补学**:自己动手修掉这两处(成本 5 分钟,收益极大)

### ❹ 对比表口径不一致
- **事实**:`eval_final_matrix.md` 里 sft 是 `prompt_format=chat`,base/dpo/checkpoint-200/rl 是 `raw`【实测】
- **为什么危险**:任何跨模型的结论都不成立
- **他会问**:"你这表公平吗?"
- **诚实答法**:"这是按'各阶段训练格式'分别评测的设计(SFT 用 chat template 训,DPO/RL 用裸 prompt),
  所以它衡量的是'部署在各自训练格式下的表现',不是严格的同口径对比。严格对比需要两种格式各跑一遍——我没做。"
- **要补学**:training-serving mismatch;如何设计 2×N 的对照实验

### ❺ `checkpoint-200` 身份误导
- **事实**:它是 **DPO 200 步**,不是 GRPO;`models/grpo/checkpoint-200` 已被 `save_total_limit` 删除
- **为什么危险**:你自己都可能讲错
- **诚实答法**:"那是 DPO 的 200 步 checkpoint;标准 GRPO 的 200 步早被清理了,
  而且那次 resume 还踩了 PEFT adapter 没恢复的坑(KL 从 0.43 掉到 0.009),所以我没用它做 benchmark。"
- **要补学**:`GRPO_EXPERIMENT_ARCHIVE.md` 全文;Trainer 断点续训与 PEFT 的交互

### ❻ LLM-as-Judge 给空回答打 4 分
- **事实**:11 条空回答全部 4 分【实测】;且 Judge 与规则排序翻转
- **为什么危险**:如果你把 Judge 当"更可信的第二个证据",会被反杀
- **诚实答法**:"Judge 也不能独立采信。我发现了它对空输出给出 4 分的失效案例,
  所以 Judge 只能作为辅助证据,不能替代事实校验。"
- **要补学**:LLM-as-Judge 的 bias(位置偏差、长度偏差、自我偏好)、如何做 judge 有效性校验

### ❼ 规则评测的 false positive
- **事实**:复读题目被判"命中✓";"21:30"(事实错误)被判对;`"12"` 会命中 `"2012"`
- **为什么危险**:面试官抽查任意一条 prediction 就可能撞上
- **诚实答法**:主动举出这两个例子,然后说改进方向(全 keyword 命中、数字词边界、拒答位置约束、重复率指标)
- **要补学**:评测指标的有效性与可操纵性(Goodhart 定律)

### ❽ 训练数据构成失衡
- **事实**:train 2676 条里 1480 条(55%)是 on-policy 自生成;SFT 实际消费的前 2000 行 **OOD = 0 条**;
  RFT 只覆盖文件前 400 行
- **为什么危险**:"为什么拒答学不会"的根因就在这,答不出说明你没分析过数据
- **诚实答法**:直接归因:"SFT 阶段没见过 OOD 样本,所以拒答率 0%;
  后面 DPO/RL 才靠上采样补。这是我数据管道的设计缺陷。"
- **要补学**:类别失衡与上采样;数据配比实验

### ❾ RFT / onpolicy 的格式不一致 + 自我蒸馏
- **事实**:`rl/rft.py:66` 用裸 prompt `tokenizer(r["q"])` 采样,但这些数据最终按 chat template 训练
- **为什么危险**:数据生成条件与训练条件不一致
- **诚实答法**:"是,生成用 raw、训练用 chat,这是我后来通过 prompt-format 消融才发现的系统性问题。"
- **要补学**:rejection sampling fine-tuning(RAFT/STaR)的标准做法

### ❿ 简历/口述数字与产物对不上
- **事实**:README 表里的 5.2% / 90.5% / 91.4% / 92.2% / 拒答 100% 全部来自 **128 条旧泄漏集**;
  92.3% 这个数字**在任何产物里都找不到**
- **为什么危险**:这是"造假"指控级别
- **诚实答法**:**永远不要引用这几个数字**。改用 §0.2 的 78 条表
- **要补学**:把 README 指标表改掉(强烈建议,半天工作量)

---

# 第七部分:学习顺序(面试收益 / 学习成本)

## Day 1 — 项目主线 + 泄漏故事(最高收益)
- 读:`README.md`(带着批判)、`data_eng/INTERVIEW_FACTS.md` 全文、`data_eng/PROJECT_FINAL_AUDIT.md` 第七/八节
- 跑:自己写脚本复算 77/183、105/105、gold_eval=0
- 必须能答:P0-1、P0-2、P0-10 + 第六部分 ❶
- 产出:2 分钟项目复述(见第八部分)

## Day 2 — 数据管道深挖
- 读:`data_eng/build_dataset.py`(重点 737–940)、`leakage_audit.py`、`kb.py`、`seeds.py`
- 必须能答:并查集四条 union 规则、clean 判定、为什么不能 80/10/10、(q,a) 去重的原因
- 必须能答:P0-10、P1-6、第六部分 ❽❾

## Day 3 — SFT + LoRA/QLoRA
- 读:`sft/train_sft.py` 全文
- 理论:LoRA 低秩分解与 α/r 缩放、NF4 量化、QLoRA、assistant-only mask
- 必须能答:P0-9、P1-7、C 组全部

## Day 4 — DPO
- 读:`rl/train_dpo.py` 全文 + TRL `dpo_trainer.py` 里 `add_adapter("ref", ...)` 那段
- 理论:Bradley-Terry → RLHF → DPO 闭式解;β 的作用;隐式奖励
- 必须能答:P0-3、P1-1、D 组全部

## Day 5 — GRPO / DAPO
- 读:`rl/train_grpo.py` + `rl/train_grpo_standard.py` + `models/rl/training_log.json`
- 理论:PPO→GRPO(去 critic)、group advantage、DAPO 四点改动、token vs sequence IS、长度偏差
- 必须能答:P0-4、P0-5、P1-2、P1-3、P1-4

## Day 6 — Evaluation + Judge
- 读:`sft/eval_domain.py` + `output_clean/llm_judge/summary.md` + 两份 eval_final 报告
- 理论:指标有效性、双轨评测的分歧、judge bias
- 必须能答:P0-2、P0-6、P0-8、P1-5、P1-11、第六部分 ❹❻❼

## Day 7 — from scratch GPT + 压力面试
- 读:`pretrain/model.py` 全文(尤其 96–165、171–187、257–271)、`pretrain/train.py`
- 理论:RoPE / GQA / SwiGLU / RMSNorm / SDPA / Pre-LN / tied embedding
- 必须能答:I 组全部、P1-8、P1-9、P1-10、第六部分 ❶–❿ 全部串讲
- 最后:完整模拟一次 30 分钟面试(第八部分的追问链)

> 顺序理由:Day1–2 直接建立"这个人做过项目"的印象,这是面试官在前 5 分钟就下的判断;
> 架构消融(Day7)虽然技术含量高,但在"是不是你自己做的"这件事上区分度最低,所以放最后。

---

# 第八部分:项目复述模板

## 2 分钟版本

> "我做的是一个垂直客服场景的对齐项目:虚构奶茶品牌'茶语时光',
> 从知识库出发自建后训练数据,在 Qwen2.5-0.5B 上跑 SFT → DPO → GRPO/DAPO,
> 目标不是答得漂亮,而是**会的答对、不会的拒答**。
>
> 硬件是 4GB 显存的笔记本,所以全程 4bit 量化 + LoRA,可训练参数 880 万,占基座 1.8%。
>
> 项目最有价值的部分其实在后面:我做完了三轮对齐后发现**评测是脏的**——
> 最终 183 条测试集里,有 77 条和我训练用的旧训练集完全同题。
> 于是我引入样本血缘和并查集 group 级切分重建了数据管道,把 train/test 泄漏清零,
> 又人工写了 78 条带事实出处的新题作为**可信子集**。
>
> 在可信子集上:base 16.7%、SFT 51.3%、DPO 52.6%、RL 37.2%。
> 也就是说 SFT 提升很大,但**我的 RL 在真正干净的题上反而更差**——
> 因为我的对称奖励让模型过度倾向拒答。这个负结果我没有藏,它是我这个项目最真实的部分。"

## 连续追问链(按真实面试节奏)

1. **"为什么这样设计?"** → 场景驱动:客服的"拒答边界"可定义,适合做可验证奖励
2. **"数据怎么来的?"** → 知识库 → 55 条种子 → 模板×实体 → 前缀×5 → 去重 → group 切分;
   并主动交代 1480 条 on-policy + 86 条蒸馏
3. **"为什么用 LoRA?"** → 4GB 显存放不下 AdamW 全参状态;给 880 万 / 1.8% 的数字
4. **"DPO 怎么做的?"** → 偏好对在线构造,chosen = **知识库 gold**,rejected = 规则分最低采样;beta=0.1
5. **"为什么不用 PPO?"** → 0.5B 模型 + 4GB 显存,再训一个 critic 不现实;GRPO 用组内归一化省掉 value model
6. **"DAPO 具体改了什么?"** → 诚实回答 TRL 默认即 dapo,讲 clip-higher / token-level,并说明标准 GRPO baseline 没进表
7. **"reward 怎么设计?"** → 命中/拒答 ±1、简洁 +0.1、OOD 上采样 20%、KL 0.04
8. **"为什么拒答率这么高?"** → 对称 ±1 + 上采样的直接后果,换来领域正确率下降,给出 gold_eval 数字
9. **"你的评测可靠吗?"** → 不可靠,两道防线都有洞:规则无事实校验(举复读判对的例子)、Judge 给空回答打 4 分
10. **"有没有数据泄漏?"** → 有,77/183,主动全讲
11. **"你做了哪些 ablation?"** → 架构 3 组、规模 4 档、prompt-format 诊断、通用数据混比;并说明 n 太小不能下强结论
12. **"如果重做你会怎么改?"** → ① 用 clean train 重训三个 checkpoint(否则一切免谈)
    ② 测试集扩到 ~300 条、每 intent ≥20
    ③ 统一所有阶段的 prompt 格式
    ④ 给规则评测加事实级校验(全 keyword + 数字词边界 + 拒答位置)
    ⑤ 做 reward 权重扫描 + 奖励与真实准确率的相关性分析

---

# 第九部分:红线清单(面试中绝对不能说)

| 不能说 | 应该说 |
|---|---|
| "准确率 92% / 90.5% / 拒答 100%" | "那是 128 条泄漏测试集的数字,我已废弃;clean 可信子集上是 51.3% / 37.2%" |
| "用了 GSPO" | GSPO 只有代码开关,从未运行 |
| "自研 FlashAttention" | "用的是 PyTorch SDPA" |
| "验证了 scaling law" | "4 档规模的 validation-loss 趋势,未做幂律拟合" |
| "做了 MQA / ALiBi 消融" | 只有代码分支,没有实验 |
| "实现了 KV Cache" | 没实现,`generate()` 每步重算 |
| "DPO 让效果大幅提升" | 10 步 DPO 在 gold_eval 上 +1.3pt,在噪声内 |
| "数据 2800+ 条" | 说清构成:约 1200 条人工/模板 gold + 1480 条自生成 + 86 条蒸馏 + 78 条评测 |
| "评测集与训练集严格分离" | 只对新数据管道成立;最终 checkpoint 用的是旧数据 |
| "我用 DAPO 取得了比 GRPO 更好的效果" | 标准 GRPO baseline 跑了但从未评测,没有这个结论 |

---

## 附:建议你立刻动手修的 3 件事(每个 < 30 分钟)

1. **修 `rl/train_grpo.py`**:删掉过期的 math docstring;把 `:218` 的"截断掩码已开启"改成按 `args` 判断
2. **修 `rl/train_dpo.py` docstring 和 README §四**:把"最高分当 chosen"改成"知识库 gold 作 chosen"
3. **修 README 指标表**:换成 §0.2 的 78 条 gold_eval 表,并标注污染子集口径

这三件事做了,你面试时"被抓现行"的概率会大幅下降——因为它们都是**面试官打开文件第一眼就会看到的**。
````

### `requirements.txt`

```text
# HuggingFace 工具链(已在 .venv 中装好)
# PyTorch 2.5.1+cu121 是从官方 CUDA 索引单独安装的,见 README
transformers
datasets
accelerate
peft
trl
bitsandbytes
sentencepiece
tiktoken
matplotlib
```

### `start.ps1`

```powershell
# 一键进入项目环境:激活虚拟环境 + 把所有缓存指到 D 盘
# 用法:每次新开终端后运行 .\start.ps1
$env:PIP_CACHE_DIR = 'D:\LLM-Project\.cache\pip'
$env:HF_HOME = 'D:\LLM-Project\.cache\huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
& D:\LLM-Project\.venv\Scripts\Activate.ps1
```
