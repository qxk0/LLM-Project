"""环境与多轮 tokenization 的最小单元测试。"""

from __future__ import annotations

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl.multiturn.env import (  # noqa: E402
    CustomerServiceEnv,
    make_episode,
    parse_action,
    run_teacher,
)
from rl.multiturn.formatting import tokenize_multiturn_example  # noqa: E402


class MultiTurnEnvTest(unittest.TestCase):
    def test_teacher_success_all_scenarios(self):
        scenarios = [
            "train_faq_coupon",
            "train_pickup_remake",
            "train_delivery_spill",
            "train_ood",
            "val_late_delivery",
            "test_delivery_no_photo",
            "test_multi_issue",
        ]
        for i, scenario in enumerate(scenarios):
            with self.subTest(scenario=scenario):
                episode = make_episode(scenario, 1000 + i)
                messages, env = run_teacher(episode)
                self.assertTrue(env.success)
                self.assertEqual(env.turns, episode.horizon)
                self.assertGreater(len(messages), 2)

    def test_ask_and_tools(self):
        episode = make_episode("train_pickup_remake", 2001)
        env = CustomerServiceEnv(episode)
        env.reset()
        r = env.step(
            {"action": "ask", "slot": "order_id", "question": "订单号?"}
        )
        self.assertIn(str(episode.order.order_id), r.observation["content"])
        r = env.step(
            {"action": "tool", "name": "lookup_order", "args": {"order_id": episode.order.order_id}}
        )
        self.assertTrue(r.info["tool_correct"])
        r = env.step(
            {
                "action": "tool",
                "name": "check_refund_policy",
                "args": {"order_id": episode.order.order_id},
            }
        )
        self.assertTrue(r.info["tool_correct"])
        self.assertEqual(r.info["tool_result"]["decision"], "remake_or_refund")
        r = env.step({"action": "answer", "content": "可以免费重做或退款。"})
        self.assertTrue(r.done)
        self.assertTrue(r.success)

    def test_invalid_action_penalty(self):
        episode = make_episode("train_ood", 2002)
        env = CustomerServiceEnv(episode)
        env.reset()
        r = env.step("{not json")
        self.assertFalse(r.done)
        self.assertLess(r.turn_reward, 0)
        self.assertEqual(env.invalid_actions, 1)

    def test_parse_action(self):
        action, err = parse_action('```json\n{"action":"ask","slot":"order_id"}\n```')
        self.assertIsNone(err)
        self.assertEqual(action["slot"], "order_id")
        action, err = parse_action('{"action":"tool","name":"lookup_order","args":{}}')
        self.assertIsNone(err)
        action, err = parse_action('{"action":"tool","name":"bad_tool","args":{}}')
        self.assertIsNotNone(err)

    def test_assistant_mask(self):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-0.6B", trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:
            self.skipTest(f"Qwen3 tokenizer unavailable: {exc}")
        episode = make_episode("train_pickup_remake", 2003)
        messages, _ = run_teacher(episode)
        enc = tokenize_multiturn_example(tokenizer, messages, max_len=1024)
        self.assertGreater(enc["assistant_tokens"], 0)
        self.assertTrue(any(label != -100 for label in enc["labels"]))
        # system + user 前缀不应该被训练
        self.assertTrue(all(label == -100 for label in enc["labels"][:10]))


if __name__ == "__main__":
    unittest.main()

