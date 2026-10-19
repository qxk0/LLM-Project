# 多轮工具客服 RL(最小闭环)

这次实现的是三个方向的最小可运行版本:

1. **多轮 RL**:确定性环境 + 4 个工具 + 规则用户 simulator;
2. **信用分配**:terminal-only vs turn-level return-to-go 两种 reward 模式;
3. **long horizon**:1/2/4/8 轮课程数据,测试按 2/4/6/8 轮分桶。

异步 RL 没有放进这次实现。单卡 4GB 上真正的 rollout/trainer 并行收益有限,先把环境、奖励和信用分配做对更重要。

## 目录

```text
rl/multiturn/env.py           环境、工具、用户 simulator、终局判定
rl/multiturn/formatting.py    Qwen chat template、assistant mask、动作解析
rl/multiturn/rollout.py       多轮 rollout
rl/multiturn/modeling.py      4bit / LoRA 加载
data_eng/build_multi_turn.py  episode 与 SFT 轨迹构建
sft/train_multiturn_sft.py    多轮 SFT
rl/train_multiturn_grpo.py    terminal-only / turn-level GRPO
rl/eval_multiturn.py          按 horizon 分桶评测
```

## 快速 smoke

```powershell
powershell -ExecutionPolicy Bypass -File sft/smoke_multiturn.ps1
powershell -ExecutionPolicy Bypass -File rl/smoke_multiturn.ps1
```

smoke 会构建 8 条 train / 4 条 test 的小数据,训 5 步多轮 SFT,再各跑 1 步
terminal-only 和 turn-level GRPO,最后评测 teacher。

## 正式数据与训练

```powershell
# 默认每个 scenario 30 条;可调 --per-scenario / --test-per-scenario
python data_eng/build_multi_turn.py

# 多轮 SFT cold start: train_all 含 1/2/4/8;30 步、lr 5e-5 是实测较稳的配置
python sft/train_multiturn_sft.py `
  --data-file data/multiturn/sft_train_all.jsonl `
  --max-steps 30 `
  --max-seq-len 768 `
  --output-dir models/sft_multiturn_all_early

# terminal-only baseline(8 轮任务)
python rl/train_multiturn_grpo.py `
  --reward-mode terminal `
  --adapter models/sft_multiturn_all_early `
  --data-file data/multiturn/episodes_train_all.jsonl `
  --curriculum 8 `
  --max-steps 10 `
  --num-generations 4 `
  --temperature 1.0 `
  --lr 1e-5 `
  --output-dir models/multiturn_grpo_terminal_h8_allearly_10

# turn-level return-to-go(稳定版:过程奖励 0.2,终局奖励 2.0)
python rl/train_multiturn_grpo.py `
  --reward-mode turn `
  --adapter models/sft_multiturn_all_early `
  --data-file data/multiturn/episodes_train_all.jsonl `
  --curriculum 8 `
  --max-steps 10 `
  --num-generations 4 `
  --temperature 1.0 `
  --lr 1e-5 `
  --process-scale 0.2 `
  --terminal-scale 2.0 `
  --output-dir models/multiturn_grpo_turn_h8_scaled

# 也可以把 curriculum 换成 1,2,4,8 做完整课程
python rl/train_multiturn_grpo.py `
  --reward-mode turn `
  --adapter models/sft_multiturn_all_early `
  --data-file data/multiturn/episodes_train_all.jsonl `
  --curriculum 1,2,4,8 `
  --max-steps 20 `
  --num-generations 4 `
  --temperature 1.0 `
  --lr 1e-5 `
  --process-scale 0.2 `
  --terminal-scale 2.0 `
  --max-prompt-tokens 640 `
  --max-seq-len 768 `
  --output-dir models/multiturn_grpo_curriculum
```

评测:

```powershell
python rl/eval_multiturn.py --teacher --data-file data/multiturn/episodes_test.jsonl
python rl/eval_multiturn.py `
  --adapters models/sft_multiturn_all_early models/multiturn_grpo_terminal_h8_allearly_10 models/multiturn_grpo_turn_h8_scaled `
  --data-file data/multiturn/episodes_test.jsonl `
  --output-dir runs/multiturn_eval
```

如果只想跑信用分配对比,可以直接用:

```powershell
powershell -ExecutionPolicy Bypass -File rl/run_multiturn_ablation.ps1 `
  -SftAdapter models/sft_multiturn_all_early -Steps 10 -Generations 4 -Curriculum 8
```

## 已跑出的结果(60 条 test,每个 horizon 15 条)

| policy | 2 轮 | 4 轮 | 6 轮 | 8 轮 | 总体 |
|---|---:|---:|---:|---:|---:|
| SFT-100(1/2/4 数据) | 100% | 100% | 100% | 0% | 75% |
| SFT-all-30(1/2/4/8,冷启动) | 100% | 100% | 0% | 0% | 50% |
| terminal-only-10(从 SFT-all) | 100% | 100% | 0% | 100% | 75% |
| turn-level-10(从 SFT-all,scaled) | 100% | 100% | 0% | 100% | 75% |

关键观察:

1. **100 步 SFT 过拟合**:train_loss 降到 8e-5,采样组内几乎完全一致,
   terminal/turn 都出现 `zero_std_fraction=1.0`,无法继续 RL;但它在 2/4/6 轮上 100%。
2. **30 步 SFT-all 保留多样性**:train_loss 0.306,8 轮训练集成功率为 0.25–1.0,
   但 test 的 6/8 轮仍为 0%,适合作为 RL cold start。
3. **terminal-only 并非总是零梯度**:从 SFT-all 出发,训练集 8 轮有成功样本,
   所以 terminal-only 也有组内差异,10 步后 test 8 轮从 0% 到 100%。
4. **turn-level 提供更密的信号**:scaled 版本在 8 轮训练步里 `used_turns=12–31`,
   但最终 test 成功率和 terminal-only 相同(都是 75%),没有额外净收益。
5. **6 轮仍是覆盖问题**:train_all 没有 6 轮 scenario,RL 只提升了它训练过的 8 轮;
   这是数据覆盖/curriculum 的缺口,不是信用分配能凭空解决的。

完整报告:
`runs/multiturn_eval_full/report.md` 与 `runs/multiturn_eval_full_terminal/report.md`。

## 信用分配对比

- `terminal`:所有 turn 共用终局 reward;
- `turn`:每轮有工具正确性/追问正确性/轮数成本,再叠加终局 reward,用
  `R_t = Σ γ^(k-t) r_k` 得到 return-to-go,按 turn 做组内归一化。

实测结论:**credit assignment 的收益取决于 terminal reward 的稀疏程度。**
如果 cold start 完全不会 8 轮(只用 1/2/4 训的 SFT),terminal-only 会
`zero_std_fraction=1.0`、没有梯度;turn-level 有信号,但如果不缩放过程奖励,
10 步内会把策略带崩。若 cold start 已能解一部分 8 轮,terminal-only 也能学,
turn-level 只是信号更密,最终成功率不一定更高。

## 指标

- 任务成功率(按 horizon 2/4/6/8 分桶)
- 平均轮数
- 工具调用准确率
- 每 1k token 成功率
- 提前拒答率
- `zero_std_fraction`(组内无梯度比例)

## 面试口径

这条线的主线是:

> "我把单轮客服扩展成多轮工具环境,核心 ablation 是 terminal-only vs turn-level
> return-to-go;test 覆盖 2/4/6/8 轮,用 horizon 分桶看外推。实验里 100 步 SFT
> 过拟合后采样无差异,terminal/turn 都没有梯度;换成 30 步 SFT-all 后,
> terminal-only 和 turn-level 都能把 8 轮从 0% 提到 100%,但 6 轮因为训练集没有
> 对应 scenario 仍是 0%。结论是 cold start 和数据覆盖比 loss 粒度更关键,
> turn-level 主要改善稀疏终局下的信号密度。"

异步 RL 不放在主线里:它优化 rollout 吞吐,不解决 credit assignment;当前 4GB 单卡也不适合
真正的 actor/trainer 并行。等环境和 reward 稳定后,再做 bounded queue + staleness 的半异步原型。

## 当前边界

- 环境是规则用户 simulator,不是自由 LLM 用户;
- 只有 4 个工具、7 种 scenario,适合做机制验证,不代表线上客服覆盖;
- reward 是确定性的规则分,LLM Judge 只做离线辅助,不做训练奖励;
- GRPO 原型没有显式 KL 项,后续可以加 ref adapter 或 PPO critic。
