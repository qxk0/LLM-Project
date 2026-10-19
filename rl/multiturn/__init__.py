"""多轮工具客服 RL 的最小实现。

模块划分:
- env.py: 环境、工具、用户 simulator、终局判定
- formatting.py: Qwen chat template、assistant token mask、动作解析
- modeling.py: 4bit 模型 / LoRA 加载
- rollout.py: 多轮 rollout
"""

