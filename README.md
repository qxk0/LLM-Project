# 垂直领域客服助手:数据构建 → SFT → RL 对齐


虚构奶茶品牌"茶语时光"的智能客服 —— 从知识库出发,自建后训练数据管道,
在 Qwen2.5-0.5B 上完成 LoRA 微调(SFT)与 GRPO 强化学习对齐,
并对比 基座 / SFT / RL 在**领域正确率、拒答率、回答简洁度**上的差异。

项目的另一条线是**从零预训练的研究型底座**(1200 万参数 GPT +
RoPE/GQA/缩放定律消融),作为后训练的模型能力基础与算法功底证明。


## 环境

- GPU:RTX 3050 Laptop(4GB 显存),PyTorch 2.7.1+cu118
- 工具链:transformers / datasets / accelerate / peft / trl / bitsandbytes
- 所有代码、虚拟环境、下载缓存都在 D:\LLM-Project 下

## 目录结构

```
data_eng/   后训练数据构建管道(知识库 + 种子 + 扩展 + 清洗 + 切分)
pretrain/   从零预训练微型 GPT(1200 万参数)+ 结构消融实验
sft/        Qwen2.5-0.5B LoRA 监督微调 + 领域评测
rl/         GRPO 强化学习对齐(领域模式 + 数学模式)
models/     训练产物(不入库)
```

## 一、后训练数据构建(项目核心)

### 1. 知识库(kb.py)

虚构品牌"茶语时光"的完整业务资料:菜单与价格、小料加价、营业时间、外卖规则、
会员积分、优惠券、售后政策、过敏原、门店信息。所有答案都从知识库生成,保证事实一致。

### 2. 种子(seeds.py)

人工基于知识库编写 60 余条高质量问答,覆盖 10 类意图(价格/推荐/定制/营业时间/
外卖/会员/售后/过敏原/门店/**知识库外拒答**),并标注答案关键实体。

### 3. 扩展 + 清洗 + 切分(build_dataset.py)

```powershell
python data_eng/build_dataset.py
```

- 模板 + 实体替换:价格模板 × 全部产品、推荐模板 × 产品标签、拒答模板 × 随机主题
- 清洗:规范化(统一标点/去空白)、精确去重、长度过滤
- 切分:按意图分层 80/10/10,评测集与训练集严格分离
- 输出:`data_eng/output/` 下 train/val/test.jsonl + stats.json(数据报告)

## 二、领域 SFT

```powershell
python sft/train_sft.py --data-file data_eng/output/train.jsonl
```

4bit 量化 + LoRA,1000+ 条领域数据,训练产物保存到 models/sft。

## 三、领域评测(指标结果)

```powershell
python sft/eval_domain.py --adapters models/sft
```

在同一评测集上给出:

| 模型 | 领域正确率 | 拒答率 | 总正确率 | 未命中率 | 平均回答长度 |
|---|---|---|---|---|---|
| 基座 | 5.2% | 0.0% | 4.7% | 94.8% | 83.7 |
| SFT | 90.5% | 58.3% | 87.5% | 9.5% | 73.8 |
| RL | 91.4% | 100.0% | 92.2% | 8.6% | 61.9 |

指标定义:领域正确率 = 回答包含答案关键实体;拒答率 = 知识库外问题被正确拒绝;
未命中率 = 领域问题上答非所问(幻觉代理指标)。

## 四、领域 GRPO(拒答对齐)

```powershell
python rl/train_grpo.py                    # 默认 domain 模式,从 SFT 继续
python sft/eval_domain.py --adapters models/sft models/rl
```

奖励设计(纯规则):
- 知识库内问题:回答包含答案关键实体 +1.0
- 知识库外问题:正确拒答(不瞎编)+1.0
- 回答简洁(≤40 字符)+0.5

RL 的目标是让模型"会的就答对、不会的就拒答",而不是胡编 —— 这正是对齐的实际意义。

## 五、从零预训练研究(算法功底)

```powershell
python pretrain/train.py --max-steps 2000 --pos-encoding rope --out-dir models/exp_rope
python pretrain/train.py --max-steps 2000 --pos-encoding learned --out-dir models/exp_learned
python pretrain/plot_results.py --scaling
```

已完成的结论:
- 6000 步基线 val_loss 1.97,能生成连贯英文小故事
- **RoPE vs 学习式位置编码:val_loss 2.378 vs 2.515(-5.4%)**
- **GQA vs MHA:省 7.2% 参数,仅损失 0.4% 效果**
- **缩放定律:1M/5M/12M/25M 四个规模验证参数量与 loss 的幂律关系**




