# 大模型全链路实践(简历项目)

一个完整跑通「预训练 -> 监督微调(SFT) -> 强化学习对齐(RL)」的大模型实践项目。
三个阶段是同一根主线的递进,不是三个零散 demo。

## 环境

- GPU:RTX 3050 Laptop(4GB 显存)
- Python 3.11.9 + CUDA 版 PyTorch 2.5.1+cu121
- 工具链:transformers / datasets / accelerate / peft / trl / bitsandbytes

所有代码、虚拟环境、下载缓存都在 D:\LLM-Project 下,C 盘零占用。

## 目录结构

```
pretrain/   第 1 天:从零预训练微型 GPT(约 1200 万参数)
sft/        第 2 天:Qwen2.5-0.5B LoRA 监督微调
rl/         第 3 天:GRPO 强化学习对齐
data/       处理后的训练数据(不入库)
models/     训练产物:tokenizer、检查点(不入库)
```

## 每日路线

### 第 1 天:从零预训练微型 GPT

学什么:tokenizer 原理、数据处理管道、Transformer 结构、训练循环。

```powershell
.\start.ps1
python pretrain/tokenize.py      # 下载 TinyStories + 训练 BPE + 生成数据
python pretrain/smoke_test.py    # 快速自检模型
python pretrain/train.py         # 正式训练(约 1~2 小时)
python pretrain/sample.py --prompt "Once upon a time,"   # 生成故事
```

常用参数:`--max-steps` 控制训练步数(先跑 `--max-steps 300` 验证流程),
`--vocab-size` 控制词表大小,`--resume` 从检查点继续。

### 第 2 天:Qwen2.5-0.5B LoRA 监督微调(SFT)

(代码待编写)

### 第 3 天:GRPO 强化学习对齐(RL)

(代码待编写)

## 简历写法(参考)

项目名:**大模型全链路实践:从零预训练到强化学习对齐**

- 从零预训练 1200 万参数 GPT,自训练 BPE tokenizer,数据管道全自建
- 基于 Qwen2.5-0.5B 完成 LoRA 监督微调,并在指令数据上评测
- 使用 GRPO 完成强化学习对齐,量化对比 SFT 前后 / RL 前后效果
