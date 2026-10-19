# 垂直领域客服 LLM:数据治理 → SFT → DPO / SimPO / GRPO → 多轮工具 Agent RL

虚构奶茶品牌"茶语时光"的智能客服 —— 从知识库出发,自建后训练数据管道,
在 4GB 单卡上用 Qwen2.5-0.5B 完成 QLoRA SFT 与 DPO / SimPO / GRPO 三对齐对照,
并进一步扩展到 Qwen3-0.6B 的多轮工具客服 Agent(订单 / 退款 / 会员 / 知识库检索)。
项目同时保留从零预训练的微型 GPT 研究线。

场景问题(项目主线):客服模型对知识库外问题会"硬答、编造答案"——
因此整个项目围绕一个问题展开:"会的就答对、不会的拒绝答",
数据、验证器、对齐方法全部为这个目标服务。

数据质量主线:项目中期发现早期 train/test 随机切分会让同一批问题的不同改写
同时进入训练和测试,导致指标虚高。据此重建了按样本血缘切分的 clean 数据管道,
并加入自动 leakage audit(详见第三节)。

## 当前结果摘要

单轮 168 条 clean test(同一 clean SFT / chat prompt):

| 模型 | 领域正确率 | 总正确率 | OOD 拒答 | gold_eval | 熟场景 |
|---|---:|---:|---:|---:|---:|
| 基座 | 10.5% | 10.7% | 13.3% | 20.5% | 14.3% |
| SFT | 33.3% | 33.9% | 40.0% | 53.8% | 54.5% |
| DPO | 36.6% | 36.9% | 40.0% | 50.0% | **64.9%** |
| SimPO | **37.3%** | **38.1%** | **46.7%** | 47.4% | 63.6% |
| GRPO | **37.3%** | 37.5% | 40.0% | **55.1%** | 62.3% |

多轮 60 条 test(2/4/6/8 轮各 15 条):

| 模型 | 2 轮 | 4 轮 | 6 轮 | 8 轮 | 总体 |
|---|---:|---:|---:|---:|---:|
| SFT-all-30(冷启动) | 100% | 100% | 0% | 0% | 50% |
| terminal-only-10 | 100% | 100% | 0% | 100% | 75% |
| turn-level-10 scaled | 100% | 100% | 0% | 100% | 75% |

核心结论:三种单轮对齐都把熟场景从 54.5% 提到 62–65%,但未见过场景仍只有 8–12%,
EOS 仍接近 0;多轮 RL 把训练过的 8 轮从 0% 提到 100%,但 6 轮因训练集缺少对应
scenario 仍是 0%。这说明 cold start、奖励设计和数据覆盖比 loss 名称更关键。


## 方法与技巧

- **SFT 领域 + 通用数据混合**:按 0.2 比例混入通用指令数据(alpaca-zh),
  防止领域过拟合、保持通用对话能力;但实验观察到混合会明显拉低拒答率,
  说明客服场景对"拒答边界"敏感,混比需按场景调优(默认混合比例已调回 0)
- **on-policy 采样 + 验证器筛选(RFT)**:让 SFT 模型自采样回答,用规则验证器
  (关键实体命中/正确拒答/简洁度)筛出高质量数据;去重需按(问题,答案)对,
  否则同问题的不同回答会被误删
- **DPO / SimPO / GRPO 同条件对照**:DPO 用 TRL 标准 sigmoid + reference model;
  SimPO 自实现 reference-free、长度归一化 margin loss;GRPO 使用标准
  `loss_type=grpo` + token-level IS。三者固定同一 SFT、clean 数据和 chat prompt
- **多轮工具 Agent RL**:自建订单/退款/会员/KB 四个工具和 2–8 轮环境,
  对比 terminal-only 与 turn-level return-to-go 两种信用分配
- **早期 DAPO / GSPO-style 研究分支**:DAPO 仅使用 TRL 的 dapo loss normalization,
  没有完整实现 clip-higher / dynamic sampling / overlong shaping;GSPO-style 只做
  sequence-level IS 对照,结果见第五节
- **评测双轨制**:关键词命中(可复现)+ LLM-as-Judge(智谱 GLM 语义打分)互补,
  前者看事实命中、后者看回答质量;两者在本项目上给出过相反的排序,因此必须交叉验证

## 环境

- GPU:RTX 3050 Laptop(4GB 显存),PyTorch 2.7.1+cu118
- 工具链:transformers / datasets / accelerate / peft / trl / bitsandbytes
- 所有代码、虚拟环境、下载缓存都在 D:\LLM-Project 下

## 目录结构

```
data_eng/   后训练数据构建管道(知识库 + 种子 + 扩展 + 清洗 + 血缘切分 + 泄漏审计)
runs/2026-10-12/data/clean/  当前 clean 数据集与审计报告(168 test)
data_eng/run_pipeline.py 全流程编排(逐步记录命令/耗时/产物 hash 到 runs/)
pretrain/   从零预训练微型 GPT(1200 万参数)+ 结构消融与参数规模对照
sft/        Qwen2.5-0.5B / Qwen3-0.6B LoRA 监督微调 + 多轮 SFT
rl/         DPO / SimPO / GRPO、DAPO 历史分支、多轮工具 Agent RL
tools/      本地网页客服与交互测试
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
- 输出:默认 `data_eng/output_clean/`;本轮端到端结果写入
  `runs/2026-10-12/data/clean/`(train/val/test + `stats.json`
  + `leakage_audit.json` / `DATASET_LEAKAGE_AUDIT.md`)

> `data_eng/output/` 是早期的随机切分版本,保留仅作历史对照:它的 183 条测试集里
> 有 **105 条与训练集同源**(77 条问题逐字相同),用它算出来的指标不能作为泛化能力证明。

## 二、领域 SFT

```powershell
python sft/train_sft.py --data-file runs/2026-10-12/data/clean/train.jsonl --max-steps 600
```

NF4 4bit 量化 + LoRA(r=16,覆盖全部线性层,可训练参数 8.8M / 502.8M = 1.75%),
数据为 clean train(2533 条,按意图分层抽样 2000 条训练 600 步),产物保存到 models/sft。
本轮 clean 重训产物在 `runs/2026-10-12/models/sft`。
脚本同时支持 Qwen3-0.6B(默认关闭 thinking),并修复了 Qwen3 非 thinking 模式下
prompt/full 序列 token 对齐问题。

### Qwen3-0.6B 低显存 smoke test

```powershell
powershell -ExecutionPolicy Bypass -File sft/smoke_qwen3.ps1
```

脚本默认使用 `Qwen/Qwen3-0.6B`、clean train、32 条样本、10 步、seq_len=256、
LoRA r=8、batch=1、grad_accum=4,并强制单卡运行(显存不足时直接 OOM,便于定位),
产物保存到 `models/sft_qwen3_smoke`。Qwen3 默认关闭 thinking 模式;
如需正式训练,把 `--smoke-test` 去掉,并按需设置 `--max-seq-len 512 --max-steps 600`。
本机实测:10 步耗时 45.8s,峰值显存 1.31 GiB(保留 1.55 GiB),适配器可正常加载。

## 三、领域评测(指标结果)

```powershell
python sft/eval_domain.py --data-file runs/2026-10-12/data/clean/test.jsonl --prompt-format chat --adapters runs/2026-10-12/models/sft models/alignment_dpo models/alignment_simpo models/alignment_grpo
# base 模型会自动一起评测,无需单独指定
```

评测集:`runs/2026-10-12/data/clean/test.jsonl`(168 条,血缘隔离零泄漏)。
当前主表(顶部)的三对齐统一使用 chat prompt;早期 DPO/DAPO/GSPO 分支使用 raw,
保留在下面作为历史对照,不再作为当前主结论。

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

**早期 raw 分支(历史对照,不作为当前主表)**

| 模型 | prompt | 熟场景新问法 | 未见过场景 | OOD 拒答 | 总体 |
|---|---|---|---|---|---|
| 基座 | chat | 14.3% | 6.6% | 6.7% | 10.1% |
| SFT(LoRA) | chat | 54.5% | 10.5% | 33.3% | 32.7% |
| 基座 | raw | 16.9% | 4.0% | 0.0% | 9.5% |
| DPO(100 步) | raw | 45.5% | 6.6% | 13.3% | 25.0% |
| DPO(200 步) | raw | 41.6% | 5.3% | 13.3% | 22.6% |
| RL(DAPO) | raw | **57.1%** | 5.3% | **46.7%** | 32.7% |
| RL(GSPO 序列级) | raw | 42.9% | 6.6% | 46.7% | 26.8% |

**历史 raw 分支的 LLM-as-Judge(语义均分,1-5)**

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

- 下面这段只针对**早期 raw 分支**;当前主表(顶部)是统一 chat prompt 的
  DPO / SimPO / GRPO 三对齐结果。
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
python rl/alignment/build_preferences.py --max-prompts 400 --num-samples 4
python rl/train_dpo_alignment.py --max-steps 100
```

当前 DPO 使用 on-policy best-of-N 偏好对:SFT 每题采样 4 个回答,
规则分最高作 chosen、最低作 rejected,只保留 chosen_score > rejected_score 的对;
得到 359 对后,用 TRL 标准 sigmoid DPO + reference model 训练 100 步。

早期 `rl/train_dpo.py` 的 gold-vs-worst 版本(216 对、200 步)保留为历史对照,
不作为当前主结果。当前 DPO 在 clean test 上:领域 36.6%、总 36.9%、
gold_eval 50.0%、熟场景 64.9%。

### SimPO(reference-free 偏好优化)

```powershell
python rl/train_simpo_alignment.py --max-steps 100
```

SimPO 与 DPO 共用 best-of-N 偏好对,但不需要 reference model:
对 chosen/rejected 的序列 log-prob 做长度归一化,使用 target margin
`beta=2.0, gamma=0.5`。100 步后在 clean test 上:领域 37.3%、总 38.1%、
OOD 46.7%、gold_eval 47.4%。训练过程比 DPO 更震荡,结果只作为对照。

## 五、领域 RL 对齐(标准 GRPO / 早期 DAPO / GSPO-style)

```powershell
python rl/train_grpo_alignment.py --max-steps 100   # 标准 GRPO
python rl/train_grpo.py --seq-level                 # GSPO-style 历史分支
```

当前主对照使用标准 GRPO:`loss_type=grpo`、`importance_sampling_level=token`、
`num_generations=4`、KL `beta=0.04`、100 步、chat prompt。结果是领域 37.3%、
总 37.5%、gold_eval 55.1%。

早期 `rl/train_grpo.py` 使用 `loss_type=dapo`,但只用到 TRL 的 token normalization;
没有显式设置 `epsilon_high=0.28`,也没有 dynamic sampling / overlong shaping,
因此它是 **DAPO loss 的部分实现**,不是完整 DAPO recipe。

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

### 多轮工具客服 RL(更深一层的扩展)

在单轮 64-token 对齐之外,新增了一个**多轮工具客服**的最小闭环:
确定性环境 + `lookup_order` / `check_refund_policy` / `get_member_info` / `search_kb`
四个工具 + 规则用户 simulator,支持 1/2/4/6/8 轮 episode。

它同时实现了两种信用分配口径:

- `terminal-only`:只使用终局 reward;
- `turn-level`:每轮叠加工具正确性、追问正确性、轮数成本,并用 return-to-go 做组内优势归一化。

train_all 覆盖 horizon 1/2/4/8,test 覆盖 2/4/6/8,因此可以单独报告 horizon 外推结果。
详见 `rl/MULTITURN_RL.md`。

60 条 test(每个 horizon 15 条)结果:

| 模型 | 2 轮 | 4 轮 | 6 轮 | 8 轮 | 总体 |
|---|---:|---:|---:|---:|---:|
| SFT-100(只训 1/2/4) | 100% | 100% | 100% | 0% | 75% |
| SFT-all-30(冷启动) | 100% | 100% | 0% | 0% | 50% |
| terminal-only-10(从 SFT-all) | 100% | 100% | 0% | 100% | 75% |
| turn-level-10 scaled(从 SFT-all) | 100% | 100% | 0% | 100% | 75% |

结论是:RL 把训练过的 8 轮从 0% 提到 100%,但 6 轮因训练集没有对应 scenario
仍是 0%;terminal-only 和 turn-level 最终成功率相同,turn-level 的优势主要是
稀疏终局下的信号密度,不是无条件更高收益。
完整结果与训练动态见 `runs/multiturn/RESULTS.md`。

```powershell
powershell -ExecutionPolicy Bypass -File sft/smoke_multiturn.ps1
powershell -ExecutionPolicy Bypass -File rl/smoke_multiturn.ps1
```

### DPO / SimPO / GRPO 对比

在同一 clean SFT、同一 clean train/test、同一 chat prompt 下完成了 100 步
DPO / SimPO / 标准 GRPO 对照:

| 模型 | 领域正确率 | 总正确率 | OOD 拒答 | gold_eval | 熟场景 |
|---|---:|---:|---:|---:|---:|
| SFT | 33.3% | 33.9% | 40.0% | 53.8% | 54.5% |
| DPO | 36.6% | 36.9% | 40.0% | 50.0% | **64.9%** |
| SimPO | **37.3%** | **38.1%** | **46.7%** | 47.4% | 63.6% |
| GRPO | **37.3%** | 37.5% | 40.0% | **55.1%** | 62.3% |

三种方法都把熟场景从 54.5% 提到 62–65%,但未见过场景仍在 8–12%,
且 EOS 接近 0。差距只有 1–2 个点,单 seed 下不能声称显著性。
完整说明见 `rl/ALIGNMENT.md`,逐条结果见 `runs/alignment_compare/`。

### 交互测试客服

```powershell
# 单轮 SFT:直接输入自然语言问题
.\.venv\Scripts\python.exe tools\chat_customer_service.py --mode single

# 多轮工具 Agent:自动追问、调工具、给最终答复
.\.venv\Scripts\python.exe tools\chat_customer_service.py --mode agent
```

输入 `:quit` 退出。单轮默认加载 `runs/2026-10-12/models/sft`;
多轮默认加载 `models/multiturn_grpo_turn_h8_scaled`。
如果 PowerShell 中文乱码,先执行 `chcp 65001`。

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
  val_loss 为 3.51 → 2.83 → 2.51 → 2.30。这是缩放趋势,本项目没有做幂律拟合,
  因此不称其为 scaling law**
- ALiBi / MQA / untied embedding 仅作为可复现代码路径保留,未计入结论;
  生成函数未实现 KV Cache,每步做全量前向。

