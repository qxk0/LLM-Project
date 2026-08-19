# 垂直领域客服助手:数据构建 → SFT → DPO → GRPO 对齐(自研底座)

一个有**真实业务场景**和**可量化指标**的大模型对齐项目:

虚构奶茶品牌"茶语时光"的智能客服 —— 从知识库出发,自建后训练数据管道,
在 Qwen2.5-0.5B 上完成 LoRA 微调(SFT)、DPO 偏好对齐、GRPO 强化学习对齐,
并对比 基座 / SFT / DPO / RL 在**领域正确率、拒答率、回答简洁度**上的差异。

项目的另一条线是**从零预训练的研究型底座**(1200 万参数 GPT,
RoPE/GQA/SwiGLU+RMSNorm/Flash Attention 完整架构消融 + 缩放定律验证)。

## 为什么这样做(对应面试反馈)

- **有现实场景**:客服是垂直领域 LLM 最典型的落地场景,问题、答案、拒答边界都清晰可定义
- **数据是重点**:不下载现成数据集,自建"种子编写 → 模板扩展 → 清洗去重 → 分层切分"
  的完整后训练数据管道,并支持规则奖励自动构造偏好对(DPO 数据)
- **对齐演进线完整**:SFT → DPO → GRPO,覆盖 2023-2026 对齐技术演进
- **有指标结果**:同一评测集上给出基座/SFT/DPO/RL 的对比表(正确率、拒答率、简洁度)
- **有思考深度**:文末讨论"后训练的收益 vs 工程化",用本项目自己的数字说话

## 环境

- GPU:RTX 3050 Laptop(4GB 显存),PyTorch 2.7.1+cu118
- 工具链:transformers / datasets / accelerate / peft / trl / bitsandbytes
- 所有代码、虚拟环境、下载缓存都在 D:\LLM-Project 下

## 目录结构

```
data_eng/   后训练数据构建管道(知识库 + 种子 + 扩展 + 清洗 + 切分)
pretrain/   从零预训练微型 GPT(1200 万参数)+ 完整架构消融
sft/        Qwen2.5-0.5B LoRA 监督微调 + 领域评测
rl/         DPO 与 GRPO 对齐(规则奖励自动构造偏好对)
models/     训练产物(不入库)
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

- 模板 + 实体替换 + 前缀改写:55 条种子 → 1250 条(22.7x)
- 清洗:规范化、精确去重、长度过滤
- 切分:按意图分层 80/10/10,评测集与训练集严格分离
- 输出:`data_eng/output/` 下 train/val/test.jsonl + stats.json(数据报告)

## 二、领域 SFT

```powershell
python sft/train_sft.py --data-file data_eng/output/train.jsonl --max-steps 600
```

4bit 量化 + LoRA,1000+ 条领域数据,训练产物保存到 models/sft。

## 三、领域评测(指标结果)

```powershell
python sft/eval_domain.py --adapters models/sft models/dpo models/rl
```

同一评测集上的最终结果:

| 模型 | 领域正确率 | 拒答率 | 总正确率 | 未命中率 | 平均回答长度 |
|---|---|---|---|---|---|
| 基座 | 5.2% | 0.0% | 4.7% | 94.8% | 83.7 |
| SFT | 90.5% | 58.3% | 87.5% | 9.5% | 73.8 |
| DPO | 待填 | 待填 | 待填 | 待填 | 待填 |
| RL | 91.4% | 100.0% | 92.2% | 8.6% | 61.9 |

指标定义:领域正确率 = 回答包含答案关键实体;拒答率 = 知识库外问题被正确拒绝;
未命中率 = 领域问题上答非所问(幻觉代理指标)。

## 四、领域 DPO(偏好对齐)

```powershell
python rl/train_dpo.py
```

DPO 不需要奖励模型:对每条问题让 SFT 采样多个回答,用规则奖励打分,
最高分当 chosen、最低分当 rejected,自动构造偏好对后做二分类式训练。
对齐技术演进线:**SFT → DPO → GRPO**,三种方法同评测集横向对比。

## 五、领域 GRPO(拒答对齐)

```powershell
python rl/train_grpo.py                    # 默认 domain 模式,从 SFT 继续
python sft/eval_domain.py --adapters models/sft models/rl
```

奖励设计(纯规则):
- 知识库内问题:回答包含答案关键实体 +1,否则 -1
- 知识库外问题:正确拒答 +1,瞎编 -1
- 回答简洁(≤80 字符)+0.1

RL 的目标是让模型"会的就答对、不会的就拒答",而不是胡编 —— 这正是对齐的实际意义。

## 六、从零预训练研究(算法功底)

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
- **Flash Attention(PyTorch SDPA):注意力显存 O(T),训练自动启用**
- **缩放定律:1M/5M/12M/25M 四个规模验证参数量与 loss 的幂律关系**

## 七、讨论:后训练的收益 vs 工程化

用本项目的数据回答这个面试高频问题:

1. **后训练的直接收益(用本项目数字)**:
   - SFT:领域正确率 5.2% → 90.5%,拒答率 0% → 58.3% —— 从"不会对话"到"按知识库回答"
   - RL:拒答率 58.3% → 100%,领域正确率 90.5% → 91.4%,总正确率 92.2%,
     平均回答长度 73.8 → 61.9 —— 从"会答但会硬答"到"会的答对、不会的拒答、回答简洁"
2. **后训练的边际收益递减**:SFT 解决了大部分"会说话",DPO/RL 解决的是剩余的
   "守边界、少幻觉";对齐方法对奖励设计和数据质量极其敏感(本项目经历过 GRPO 不收敛,
   通过诊断奖励信号定位到回答截断、提示词不一致、适配器更新目标错位三个坑)。
3. **工程化同样重要**:4bit 量化、LoRA、Flash Attention、评测自动化 —— 工程能力决定了
   模型能不能落地。后训练和工程不是二选一,是"模型能力"和"交付能力"两条腿。

## 简历写法(参考)

项目名:**垂直领域客服助手:数据构建与 SFT/DPO/GRPO 三段式对齐(自研 1200 万参数底座)**

- 自建后训练数据管道:55 条人工种子经模板+实体替换+前缀改写扩展至 1250 条,
  规则清洗去重、意图分层切分,含 9.6% 拒答样本
- 基于 Qwen2.5-0.5B 完成 SFT+DPO+GRPO 三段式对齐:领域正确率 5.2%→91.4%,
  拒答率 0%→100%,总正确率 92.2%(基座/SFT/DPO/RL 同评测集对比);
  用规则奖励自动构造 DPO 偏好对,GRPO 用对称奖励训练拒答边界
- 从零预训练 1200 万参数 GPT:实现 RoPE/GQA/ALiBi/SwiGLU/RMSNorm/Flash Attention,
  完整消融 RoPE(-5.4%)、GQA(省 7.2% 参数)、SwiGLU+RMSNorm(-3.2%),
  四个规模验证缩放定律
