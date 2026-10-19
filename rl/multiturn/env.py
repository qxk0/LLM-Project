"""确定性的多轮客服环境。

设计目标:
1. 每轮只接受一个 JSON 动作,便于规则 reward 和可复现评测;
2. 工具和用户 simulator 都是确定性的,不依赖外部 LLM;
3. 支持 1/2/4/6/8 个 assistant action 的 episode,用于 horizon 课程;
4. 同时记录 terminal reward 与 turn-level process reward,供信用分配对比。

动作:
  {"action":"ask","slot":"order_id|issue|photo|member_id|coupon_code","question":"..."}
  {"action":"tool","name":"lookup_order|check_refund_policy|get_member_info|search_kb","args":{...}}
  {"action":"answer","content":"..."}
  {"action":"refuse","content":"..."}
  {"action":"escalate","content":"..."}
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from data_eng.kb import PRODUCTS, REFUSAL_TEMPLATES

SYSTEM_PROMPT = """你是茶语时光客服 agent。你只能通过工具和用户完成客服任务。
每轮只输出一个 JSON 对象,不要输出 Markdown、解释或多余文字。

可用动作:
1. 追问用户:
{"action":"ask","slot":"order_id|issue|photo|member_id|coupon_code","question":"你的问题"}
2. 调用工具:
{"action":"tool","name":"lookup_order|check_refund_policy|get_member_info|search_kb","args":{...}}
3. 最终回答:
{"action":"answer","content":"给用户的回答"}
4. 拒答知识库外问题:
{"action":"refuse","content":"拒答话术"}
5. 转人工:
{"action":"escalate","content":"转人工说明"}

规则:
- 不要编造订单、会员、优惠券或政策事实;不知道就先 ask 或调用工具。
- 退款/重做/转人工必须依据 check_refund_policy 的结果。
- 最终回答要简洁,并在解决后停止。"""

VALID_ACTIONS = {"ask", "tool", "answer", "refuse", "escalate"}
VALID_SLOTS = {"order_id", "issue", "photo", "member_id", "coupon_code"}
VALID_TOOLS = {"lookup_order", "check_refund_policy", "get_member_info", "search_kb"}

REFUSAL_MARKERS = ["抱歉", "不好意思", "无法回答", "超出", "服务范围", "无法处理"]
ESCALATE_MARKERS = ["转人工", "人工客服", "客服专员", "升级处理"]

ISSUE_TEXT = {
    "none": "没有特别的问题。",
    "wrong_flavor": "饮品做错口味了。",
    "spill": "外卖杯盖没盖好,洒出来了。",
    "late": "外卖超时很久才送到。",
    "spill_late_missing": "外卖超时、洒漏,还少了一件。",
}


@dataclass
class Order:
    order_id: str
    user_id: str
    product: str
    price: int
    channel: str  # pickup / delivery
    minutes_since_delivery: int
    issue: str
    photo_uploaded: bool
    member_tier: str
    points: int
    coupon_code: Optional[str] = None
    coupon_desc: Optional[str] = None


@dataclass
class Episode:
    episode_id: str
    family_id: str
    scenario: str
    horizon: int
    initial_message: str
    order: Order
    required_slots: List[str]
    expected_outcome: str  # answer / refuse / escalate
    expected_keywords: List[str]
    solution: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def max_turns(self) -> int:
        return self.horizon + 2


@dataclass
class StepResult:
    observation: Optional[Dict[str, str]]
    done: bool
    turn_reward: float
    terminal_reward: float
    success: bool
    info: Dict[str, Any]


# 每个 scenario 对应一个固定的任务形态。
# train_* 只覆盖 1/2/4 轮;test_* 覆盖 2/4/6/8 轮,其中 6/8 轮是训练未见 horizon。
SCENARIO_CONFIG: Dict[str, Dict[str, Any]] = {
    "faq_coupon": {
        "horizon": 2,
        "initial": "新客有什么优惠?",
        "channel": "pickup",
        "issue": "none",
        "minutes": 0,
        "photo": False,
        "expected_outcome": "answer",
        "expected_keywords": ["3", "新客"],
        "required_slots": [],
    },
    "faq_hours": {
        "horizon": 2,
        "initial": "你们营业到几点?",
        "channel": "pickup",
        "issue": "none",
        "minutes": 0,
        "photo": False,
        "expected_outcome": "answer",
        "expected_keywords": ["10:00", "22:00"],
        "required_slots": [],
    },
    "faq_member": {
        "horizon": 2,
        "initial": "积分怎么用?",
        "channel": "pickup",
        "issue": "none",
        "minutes": 0,
        "photo": False,
        "expected_outcome": "answer",
        "expected_keywords": ["100", "5"],
        "required_slots": [],
    },
    "faq_store": {
        "horizon": 2,
        "initial": "附近有门店吗?",
        "channel": "pickup",
        "issue": "none",
        "minutes": 0,
        "photo": False,
        "expected_outcome": "answer",
        "expected_keywords": ["小程序", "200"],
        "required_slots": [],
    },
    "pickup_remake": {
        "horizon": 4,
        "initial": "我的饮品做错口味了,可以怎么处理?",
        "channel": "pickup",
        "issue": "wrong_flavor",
        "minutes": 15,
        "photo": False,
        "expected_outcome": "answer",
        "expected_keywords": ["重做", "退款"],
        "required_slots": ["order_id"],
    },
    "delivery_spill": {
        "horizon": 4,
        "initial": "外卖杯盖没盖好洒出来了,怎么退?",
        "channel": "delivery",
        "issue": "spill",
        "minutes": 20,
        "photo": True,
        "expected_outcome": "answer",
        "expected_keywords": ["拍照", "退款"],
        "required_slots": ["order_id"],
    },
    "ood": {
        "horizon": 1,
        "initial": "你们能修手机吗?",
        "channel": "pickup",
        "issue": "none",
        "minutes": 0,
        "photo": False,
        "expected_outcome": "refuse",
        "expected_keywords": ["无法回答", "服务范围"],
        "required_slots": [],
    },
    "late_delivery": {
        "horizon": 6,
        "initial": "外卖超时很久了,怎么办?",
        "channel": "delivery",
        "issue": "late",
        "minutes": 55,
        "photo": False,
        "expected_outcome": "escalate",
        "expected_keywords": ["转人工", "客服"],
        "required_slots": ["order_id", "member_id"],
    },
    "delivery_no_photo": {
        "horizon": 6,
        "initial": "外卖洒了,能退款吗?",
        "channel": "delivery",
        "issue": "spill",
        "minutes": 25,
        "photo": False,
        "expected_outcome": "answer",
        "expected_keywords": ["拍照", "退款"],
        "required_slots": ["order_id", "photo"],
    },
    "multi_issue": {
        "horizon": 8,
        "initial": "我的外卖又晚又洒了,还少了一件,怎么处理?",
        "channel": "delivery",
        "issue": "spill_late_missing",
        "minutes": 45,
        "photo": False,
        "expected_outcome": "answer",
        "expected_keywords": ["退款", "优惠券"],
        "required_slots": ["order_id", "photo", "member_id"],
    },
}


def _base_scenario(scenario: str) -> str:
    for prefix in ("train_", "val_", "test_"):
        if scenario.startswith(prefix):
            return scenario[len(prefix):]
    return scenario


def make_episode(scenario: str, index: int, seed: int = 42) -> Episode:
    """按 scenario + index 生成一个确定性 episode。"""
    base = _base_scenario(scenario)
    if base not in SCENARIO_CONFIG:
        raise ValueError(f"unknown scenario: {scenario} (base={base})")
    cfg = SCENARIO_CONFIG[base]
    rng = random.Random(seed * 100_000 + index)
    product = rng.choice(PRODUCTS)
    order_id = f"ORD{index:05d}"
    user_id = f"U{index:05d}"
    coupon_code = "NEW3" if base in {"faq_coupon", "multi_issue"} else None
    coupon_desc = "新客注册立减 3 元" if coupon_code else None

    order = Order(
        order_id=order_id,
        user_id=user_id,
        product=product["name"],
        price=product["price"],
        channel=cfg["channel"],
        minutes_since_delivery=cfg["minutes"],
        issue=cfg["issue"],
        photo_uploaded=cfg["photo"] if base != "delivery_no_photo" else False,
        member_tier=rng.choice(["普通会员", "金卡会员"]),
        points=rng.choice([80, 120, 160]),
        coupon_code=coupon_code,
        coupon_desc=coupon_desc,
    )
    episode = Episode(
        episode_id=f"{scenario}_{index:04d}",
        family_id=f"{scenario}:{product['name']}",
        scenario=scenario,
        horizon=int(cfg["horizon"]),
        initial_message=cfg["initial"],
        order=order,
        required_slots=list(cfg["required_slots"]),
        expected_outcome=cfg["expected_outcome"],
        expected_keywords=list(cfg["expected_keywords"]),
    )
    episode.solution = teacher_plan(episode)
    if len(episode.solution) != episode.horizon:
        raise AssertionError(
            f"{scenario}: solution length {len(episode.solution)} != horizon {episode.horizon}"
        )
    return episode


def teacher_plan(episode: Episode) -> List[Dict[str, Any]]:
    """脚本化 teacher: 用于生成 SFT 轨迹和验证环境可解性。"""
    base = _base_scenario(episode.scenario)
    oid = episode.order.order_id
    uid = episode.order.user_id
    if base == "faq_coupon":
        return [
            {"action": "tool", "name": "search_kb", "args": {"query": "新客优惠"}},
            {"action": "answer", "content": "新客注册立减 3 元。"},
        ]
    if base == "faq_hours":
        return [
            {"action": "tool", "name": "search_kb", "args": {"query": "营业时间"}},
            {"action": "answer", "content": "营业时间 10:00-22:00,外卖 21:30 截止。"},
        ]
    if base == "faq_member":
        return [
            {"action": "tool", "name": "search_kb", "args": {"query": "会员积分"}},
            {"action": "answer", "content": "消费 1 元积 1 分,100 积分可抵 5 元。"},
        ]
    if base == "faq_store":
        return [
            {"action": "tool", "name": "search_kb", "args": {"query": "门店"}},
            {"action": "answer", "content": "全国 200 多家门店,可在小程序查看最近门店。"},
        ]
    if base == "ood":
        return [{"action": "refuse", "content": REFUSAL_TEMPLATES[0]}]
    if base in {"pickup_remake", "delivery_spill"}:
        final = (
            "30 分钟内可以免费重做或退款。"
            if base == "pickup_remake"
            else "外卖洒漏,拍照联系客服可以退款。"
        )
        return [
            {"action": "ask", "slot": "order_id", "question": "请提供订单号。"},
            {"action": "tool", "name": "lookup_order", "args": {"order_id": oid}},
            {"action": "tool", "name": "check_refund_policy", "args": {"order_id": oid}},
            {"action": "answer", "content": final},
        ]
    if base == "late_delivery":
        return [
            {"action": "ask", "slot": "order_id", "question": "请提供订单号。"},
            {"action": "tool", "name": "lookup_order", "args": {"order_id": oid}},
            {"action": "tool", "name": "check_refund_policy", "args": {"order_id": oid}},
            {"action": "ask", "slot": "member_id", "question": "请提供会员号。"},
            {"action": "tool", "name": "get_member_info", "args": {"member_id": uid}},
            {"action": "escalate", "content": "这个问题我帮您转人工客服处理。"},
        ]
    if base == "delivery_no_photo":
        return [
            {"action": "ask", "slot": "order_id", "question": "请提供订单号。"},
            {"action": "tool", "name": "lookup_order", "args": {"order_id": oid}},
            {"action": "tool", "name": "check_refund_policy", "args": {"order_id": oid}},
            {"action": "ask", "slot": "photo", "question": "请拍照上传撒漏情况。"},
            {"action": "tool", "name": "check_refund_policy", "args": {"order_id": oid}},
            {"action": "answer", "content": "需要拍照上传,之后可以退款。"},
        ]
    if base == "multi_issue":
        return [
            {"action": "ask", "slot": "order_id", "question": "请提供订单号。"},
            {"action": "tool", "name": "lookup_order", "args": {"order_id": oid}},
            {"action": "tool", "name": "check_refund_policy", "args": {"order_id": oid}},
            {"action": "ask", "slot": "photo", "question": "请拍照上传撒漏情况。"},
            {"action": "tool", "name": "check_refund_policy", "args": {"order_id": oid}},
            {"action": "ask", "slot": "member_id", "question": "请提供会员号。"},
            {"action": "tool", "name": "get_member_info", "args": {"member_id": uid}},
            {"action": "answer", "content": "可以退款,并为您补发优惠券。"},
        ]
    raise ValueError(f"no teacher plan for {episode.scenario}")


def parse_action(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """从模型输出中提取一个 JSON 动作。"""
    if not text or not text.strip():
        return None, "empty_action"
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        return None, "no_json_object"
    try:
        obj = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    if not isinstance(obj, dict) or obj.get("action") not in VALID_ACTIONS:
        return None, "invalid_action"
    kind = obj["action"]
    if kind == "ask":
        if obj.get("slot") not in VALID_SLOTS:
            return None, "invalid_slot"
        obj.setdefault("question", "请补充相关信息。")
    elif kind == "tool":
        if obj.get("name") not in VALID_TOOLS:
            return None, "invalid_tool"
        if not isinstance(obj.get("args"), dict):
            obj["args"] = {}
    else:
        if not isinstance(obj.get("content"), str):
            return None, "missing_content"
    return obj, None


def format_action(action: Dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


class CustomerServiceEnv:
    """单 episode 的确定性环境。"""

    def __init__(self, episode: Episode):
        self.episode = episode
        self.order = copy.deepcopy(episode.order)
        self.done = False
        self.success = False
        self.turns = 0
        self.known_slots: set[str] = set()
        self.lookup_done = False
        self.asked_slots: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_correct = 0
        self.invalid_actions = 0
        self.total_turn_reward = 0.0
        self.final_action: Optional[Dict[str, Any]] = None
        self.pending_slot: Optional[str] = None
        self.messages = self.reset()

    def reset(self) -> List[Dict[str, str]]:
        self.done = False
        self.success = False
        self.turns = 0
        self.known_slots.clear()
        self.lookup_done = False
        self.asked_slots.clear()
        self.tool_calls.clear()
        self.tool_correct = 0
        self.invalid_actions = 0
        self.total_turn_reward = 0.0
        self.final_action = None
        self.pending_slot = None
        self.order = copy.deepcopy(self.episode.order)
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.episode.initial_message},
        ]
        return self.messages

    def step(self, action: Any, auto_user: bool = True) -> StepResult:
        if self.done:
            return StepResult(None, True, 0.0, 0.0, self.success, {"already_done": True})

        self.turns += 1
        turn_reward = -0.02  # 每多一轮都有成本
        terminal_reward = 0.0
        info: Dict[str, Any] = {"turns": self.turns}
        observation: Optional[Dict[str, str]] = None

        if isinstance(action, str):
            parsed, err = parse_action(action)
        elif isinstance(action, dict):
            parsed, err = action, None
            if parsed.get("action") not in VALID_ACTIONS:
                err = "invalid_action"
            elif parsed.get("action") == "ask" and parsed.get("slot") not in VALID_SLOTS:
                err = "invalid_slot"
            elif parsed.get("action") == "tool" and parsed.get("name") not in VALID_TOOLS:
                err = "invalid_tool"
            elif parsed.get("action") in {"answer", "refuse", "escalate"} and not isinstance(
                parsed.get("content"), str
            ):
                err = "missing_content"
        else:
            parsed, err = None, "unsupported_action_type"

        if err is not None or parsed is None:
            self.invalid_actions += 1
            turn_reward -= 0.2
            info["invalid"] = True
            info["parse_error"] = err
            observation = {
                "role": "user",
                "content": "动作格式错误。请只输出一个合法 JSON 动作,并包含 action 字段。",
            }
        else:
            kind = parsed["action"]
            if kind == "ask":
                slot = parsed["slot"]
                self.asked_slots.append(slot)
                if slot in self.episode.required_slots and slot not in self.known_slots:
                    self.known_slots.add(slot)
                    turn_reward += 0.10
                    info["ask_correct"] = True
                else:
                    turn_reward -= 0.05
                    info["ask_correct"] = False
                if auto_user:
                    observation = {"role": "user", "content": self._user_reply(slot)}
                else:
                    # 等待真人顾客输入,不自动替用户回答。
                    self.pending_slot = slot
                    info["need_user"] = True
                    info["slot"] = slot
                    observation = None
            elif kind == "tool":
                name = parsed["name"]
                args = parsed.get("args", {})
                tool_result = self._call_tool(name, args)
                correct = self._tool_correct(name, args, tool_result)
                repeated = any(
                    c["name"] == name and c["args"] == args for c in self.tool_calls
                )
                self.tool_calls.append({"name": name, "args": args, "correct": correct})
                if correct:
                    self.tool_correct += 1
                    turn_reward += 0.15
                else:
                    turn_reward -= 0.10
                if repeated:
                    turn_reward -= 0.05
                    info["repeated_tool"] = True
                info["tool_correct"] = correct
                info["tool_result"] = tool_result
                observation = {
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False, separators=(",", ":")),
                }
            else:
                self.final_action = parsed
                self.success = self._check_terminal(parsed)
                terminal_reward = 1.0 if self.success else -1.0
                turn_reward += terminal_reward
                self.done = True
                info["terminal"] = True
                info["expected_outcome"] = self.episode.expected_outcome
                info["final_action"] = parsed

        if not self.done and self.turns >= self.episode.max_turns:
            self.done = True
            self.success = False
            terminal_reward = -1.0
            turn_reward += terminal_reward
            info["max_turns"] = True
            observation = {"role": "user", "content": "本轮对话结束。"}

        self.total_turn_reward += turn_reward
        return StepResult(
            observation=observation,
            done=self.done,
            turn_reward=turn_reward,
            terminal_reward=terminal_reward,
            success=self.success,
            info=info,
        )

    def reply_as_user(self, text: str) -> Dict[str, str]:
        """真人顾客回复当前 ask slot。"""
        slot = self.pending_slot
        reply = text
        if slot == "order_id":
            match = re.search(r"ORD\d+", text, flags=re.IGNORECASE)
            if match:
                self.order.order_id = match.group(0).upper()
            self.known_slots.add("order_id")
            reply = f"订单号是 {self.order.order_id}。"
        elif slot == "issue":
            self.known_slots.add("issue")
            reply = ISSUE_TEXT.get(self.order.issue, text)
        elif slot == "photo":
            if any(k in text for k in ("拍", "上传", "照片", "图片", "图")):
                self.order.photo_uploaded = True
            self.known_slots.add("photo")
            reply = "我现在拍照上传了。" if self.order.photo_uploaded else "还没有拍照。"
        elif slot == "member_id":
            match = re.search(r"U\d+", text, flags=re.IGNORECASE)
            if match:
                self.order.user_id = match.group(0).upper()
            self.known_slots.add("member_id")
            reply = f"会员号是 {self.order.user_id}。"
        elif slot == "coupon_code":
            self.known_slots.add("coupon_code")
            if self.order.coupon_code:
                reply = f"我有 {self.order.coupon_code} 优惠券。"
            else:
                reply = "我没有优惠券。"
        self.pending_slot = None
        return {"role": "user", "content": reply}

    def _user_reply(self, slot: str) -> str:
        if slot == "order_id":
            return f"订单号是 {self.order.order_id}。"
        if slot == "issue":
            return ISSUE_TEXT.get(self.order.issue, "我不太清楚具体问题。")
        if slot == "photo":
            if not self.order.photo_uploaded:
                self.order.photo_uploaded = True
                return "我现在拍照上传了。"
            return "已经拍照上传了。"
        if slot == "member_id":
            return f"会员号是 {self.order.user_id}。"
        if slot == "coupon_code":
            if self.order.coupon_code:
                return f"我有 {self.order.coupon_code} 优惠券。"
            return "我没有优惠券。"
        return "我不太清楚,你帮我查一下吧。"

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "lookup_order":
            if args.get("order_id") != self.order.order_id:
                return {"ok": False, "error": "order_not_found"}
            self.lookup_done = True
            return {
                "ok": True,
                "order_id": self.order.order_id,
                "product": self.order.product,
                "price": self.order.price,
                "channel": self.order.channel,
                "minutes_since_delivery": self.order.minutes_since_delivery,
                "issue": self.order.issue,
                "photo_uploaded": self.order.photo_uploaded,
            }
        if name == "check_refund_policy":
            if args.get("order_id") != self.order.order_id:
                return {"ok": False, "error": "order_not_found"}
            if not self.lookup_done:
                return {"ok": False, "error": "lookup_order_first"}
            issue = self.order.issue
            if issue in {"spill", "spill_late_missing"}:
                if not self.order.photo_uploaded:
                    return {
                        "ok": True,
                        "decision": "need_photo",
                        "reason": "外卖洒漏需要拍照后才能退款。",
                    }
                if issue == "spill_late_missing":
                    return {
                        "ok": True,
                        "decision": "refund_plus_coupon",
                        "reason": "洒漏且超时,可退款并补发优惠券。",
                    }
                return {
                    "ok": True,
                    "decision": "refund",
                    "reason": "外卖洒漏且已拍照,可退款。",
                }
            if issue == "wrong_flavor":
                if self.order.minutes_since_delivery <= 30:
                    return {
                        "ok": True,
                        "decision": "remake_or_refund",
                        "reason": "30 分钟内口味问题可重做或退款。",
                    }
                return {
                    "ok": True,
                    "decision": "not_eligible",
                    "reason": "超过 30 分钟,不符合重做/退款条件。",
                }
            if issue == "late":
                return {
                    "ok": True,
                    "decision": "escalate",
                    "reason": "配送超时需转人工核实。",
                }
            return {"ok": True, "decision": "no_issue", "reason": "没有售后问题。"}
        if name == "get_member_info":
            if args.get("member_id") != self.order.user_id:
                return {"ok": False, "error": "member_not_found"}
            return {
                "ok": True,
                "member_id": self.order.user_id,
                "tier": self.order.member_tier,
                "points": self.order.points,
                "coupon_code": self.order.coupon_code,
                "coupon_desc": self.order.coupon_desc,
            }
        if name == "search_kb":
            query = str(args.get("query", ""))
            if "新客" in query:
                return {"ok": True, "found": True, "answer": "新客注册立减 3 元。"}
            if "退款" in query:
                return {
                    "ok": True,
                    "found": True,
                    "answer": "饮品 30 分钟内可重做/退款;外卖洒漏拍照可退款。",
                }
            if "营业" in query or "时间" in query:
                return {"ok": True, "found": True, "answer": "营业时间 10:00-22:00,外卖 21:30 截止。"}
            if "会员" in query:
                return {"ok": True, "found": True, "answer": "1 元积 1 分,100 积分抵 5 元。"}
            if "门店" in query or "wifi" in query.lower():
                return {
                    "ok": True,
                    "found": True,
                    "answer": "全国 200 多家门店,可在小程序查看最近门店;门店提供免费 WiFi。",
                }
            return {"ok": True, "found": False, "answer": "知识库没有找到相关条目。"}
        return {"ok": False, "error": "unknown_tool"}

    def _tool_correct(self, name: str, args: Dict[str, Any], result: Dict[str, Any]) -> bool:
        if name == "lookup_order":
            return args.get("order_id") == self.order.order_id
        if name == "check_refund_policy":
            return args.get("order_id") == self.order.order_id and self.lookup_done
        if name == "get_member_info":
            return args.get("member_id") == self.order.user_id
        if name == "search_kb":
            return bool(args.get("query")) and bool(result.get("found", False))
        return False

    def _check_terminal(self, action: Dict[str, Any]) -> bool:
        kind = action["action"]
        content = str(action.get("content", ""))
        expected = self.episode.expected_outcome
        if expected == "refuse":
            return kind == "refuse" and any(m in content for m in REFUSAL_MARKERS)
        if expected == "escalate":
            return kind == "escalate" and any(m in content for m in ESCALATE_MARKERS)
        if expected == "answer":
            return kind == "answer" and all(k in content for k in self.episode.expected_keywords)
        return False


def run_teacher(episode: Episode) -> Tuple[List[Dict[str, str]], CustomerServiceEnv]:
    """用脚本化 teacher 跑完整 episode,用于生成 SFT 轨迹。"""
    env = CustomerServiceEnv(episode)
    messages = env.reset()
    for action in episode.solution:
        messages.append({"role": "assistant", "content": format_action(action)})
        result = env.step(action)
        if result.observation:
            messages.append(result.observation)
    if not env.done or not env.success:
        raise RuntimeError(
            f"teacher failed: episode={episode.episode_id} turns={env.turns} "
            f"success={env.success} final={env.final_action}"
        )
    return messages, env


def episode_to_dict(episode: Episode) -> Dict[str, Any]:
    return asdict(episode)


def episode_from_dict(data: Dict[str, Any]) -> Episode:
    payload = dict(data)
    order = Order(**payload.pop("order"))
    return Episode(order=order, **payload)

