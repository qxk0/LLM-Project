"""多轮 rollout。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from rl.multiturn.env import (
    CustomerServiceEnv,
    Episode,
    format_action,
    parse_action,
)
from rl.multiturn.formatting import render_chat


@dataclass
class TurnRecord:
    prompt_ids: List[int]
    action_ids: List[int]
    process_reward: float
    terminal_reward: float
    raw_action: str
    parsed_action: Optional[Dict[str, Any]]
    observation: Optional[Dict[str, str]]
    tool_correct: Optional[bool]
    ask_correct: Optional[bool]
    invalid: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    episode_id: str
    horizon: int
    turns: List[TurnRecord]
    terminal_reward: float
    success: bool
    transcript: List[Dict[str, str]]
    tool_calls: int
    tool_correct: int
    invalid_actions: int
    total_tokens: int
    returns: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)


def _stop_ids(tokenizer) -> List[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id and im_end not in ids:
        ids.append(im_end)
    return ids


def rollout_episode(
    model,
    tokenizer,
    episode: Episode,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    do_sample: bool = True,
    max_prompt_tokens: int = 768,
    max_seq_len: int = 1024,
    enable_thinking: bool = False,
    device: Optional[str] = None,
) -> Trajectory:
    """模型与确定性环境交互,返回一条可训练的轨迹。"""
    env = CustomerServiceEnv(episode)
    history = env.reset()
    turns: List[TurnRecord] = []
    stop_ids = _stop_ids(tokenizer)
    target_device = device or str(model.device)

    model.eval()
    model.config.use_cache = True

    while not env.done:
        prompt = render_chat(
            tokenizer,
            history,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(prompt_ids) > max_prompt_tokens:
            # 保留开头的 system/动作 schema,再保留最近的历史,避免截掉最新 tool/user 消息。
            head = min(128, max_prompt_tokens // 4)
            prompt_ids = prompt_ids[:head] + prompt_ids[-(max_prompt_tokens - head):]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
        attention_mask = torch.ones_like(input_ids)
        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": stop_ids,
        }
        if do_sample:
            gen_kwargs.update(
                temperature=max(temperature, 1e-5),
                top_p=0.95,
                top_k=50,
            )
        with torch.no_grad():
            generated = model.generate(**gen_kwargs)
        generated_ids = generated[0][input_ids.shape[1]:].tolist()
        raw_action = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        parsed, _err = parse_action(raw_action)
        result = env.step(parsed if parsed is not None else raw_action)

        action_text = raw_action or "{}"
        train_action_ids = tokenizer(
            action_text, add_special_tokens=False
        ).input_ids
        if tokenizer.eos_token_id is not None:
            train_action_ids = train_action_ids + [tokenizer.eos_token_id]
        if not train_action_ids:
            train_action_ids = [tokenizer.eos_token_id]
        room = max_seq_len - len(prompt_ids)
        if room <= 0:
            prompt_ids = prompt_ids[:max_seq_len - 1]
            room = 1
        train_action_ids = train_action_ids[:room]

        turns.append(
            TurnRecord(
                prompt_ids=prompt_ids,
                action_ids=train_action_ids,
                process_reward=result.turn_reward - result.terminal_reward,
                terminal_reward=result.terminal_reward,
                raw_action=raw_action,
                parsed_action=parsed,
                observation=result.observation,
                tool_correct=result.info.get("tool_correct"),
                ask_correct=result.info.get("ask_correct"),
                invalid=bool(result.info.get("invalid")),
                info=result.info,
            )
        )
        history.append({"role": "assistant", "content": action_text})
        if result.observation:
            history.append(result.observation)

    return Trajectory(
        episode_id=episode.episode_id,
        horizon=episode.horizon,
        turns=turns,
        terminal_reward=1.0 if env.success else -1.0,
        success=env.success,
        transcript=history,
        tool_calls=len(env.tool_calls),
        tool_correct=env.tool_correct,
        invalid_actions=env.invalid_actions,
        total_tokens=sum(len(t.action_ids) for t in turns),
    )


def teacher_trajectory(episode: Episode) -> Trajectory:
    """把脚本化 teacher 包装成与模型 rollout 相同的 Trajectory。"""
    env = CustomerServiceEnv(episode)
    history = env.reset()
    turns: List[TurnRecord] = []
    for action in episode.solution:
        result = env.step(action)
        turns.append(
            TurnRecord(
                prompt_ids=[],
                action_ids=[],
                process_reward=result.turn_reward - result.terminal_reward,
                terminal_reward=result.terminal_reward,
                raw_action=format_action(action),
                parsed_action=action,
                observation=result.observation,
                tool_correct=result.info.get("tool_correct"),
                ask_correct=result.info.get("ask_correct"),
                invalid=False,
                info=result.info,
            )
        )
        history.append({"role": "assistant", "content": format_action(action)})
        if result.observation:
            history.append(result.observation)
    return Trajectory(
        episode_id=episode.episode_id,
        horizon=episode.horizon,
        turns=turns,
        terminal_reward=1.0 if env.success else -1.0,
        success=env.success,
        transcript=history,
        tool_calls=len(env.tool_calls),
        tool_correct=env.tool_correct,
        invalid_actions=env.invalid_actions,
        total_tokens=0,
    )

