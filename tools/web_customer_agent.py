"""本地网页客服体验:顾客由你扮演,模型作为多轮工具 Agent。

启动:
  python tools/web_customer_agent.py --port 7860

浏览器打开 http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl.multiturn.env import CustomerServiceEnv, make_episode, parse_action  # noqa: E402
from rl.multiturn.formatting import render_chat  # noqa: E402
from rl.multiturn.modeling import load_model_and_tokenizer  # noqa: E402
if os.path.join(REPO, "sft") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "sft"))
from eval_domain import (  # noqa: E402
    build_model as build_single_turn_model,
    build_prompt as build_single_turn_prompt,
    resolve_stop_ids,
)
from transformers import AutoTokenizer  # noqa: E402


SCENARIOS = {
    "multi_issue": "test_multi_issue",
    "delivery_no_photo": "test_delivery_no_photo",
    "pickup_remake": "test_pickup_remake",
    "faq_coupon": "test_faq_coupon",
    "faq_general": "test_faq_coupon",
    "faq_hours": "faq_hours",
    "faq_member": "faq_member",
    "faq_store": "faq_store",
    "ood": "train_ood",
    "late_delivery": "val_late_delivery",
}
SCENARIO_LABELS = {
    "multi_issue": "外卖洒漏 + 超时 + 少件",
    "delivery_no_photo": "外卖洒漏，尚未拍照",
    "pickup_remake": "自取饮品做错口味",
    "faq_coupon": "新客优惠咨询",
    "faq_general": "商品/价格/推荐咨询",
    "faq_hours": "营业时间咨询",
    "faq_member": "会员积分咨询",
    "faq_store": "门店/WiFi 咨询",
    "ood": "知识库外问题",
    "late_delivery": "外卖超时",
}
SLOT_LABELS = {
    "order_id": "订单号",
    "issue": "具体问题",
    "photo": "撒漏照片",
    "member_id": "会员号",
    "coupon_code": "优惠券信息",
}
ISSUE_LABELS = {
    "none": "无",
    "wrong_flavor": "饮品做错口味",
    "spill": "外卖洒漏",
    "late": "外卖超时",
    "spill_late_missing": "外卖洒漏 + 超时 + 少件",
}

MODEL_LOCK = threading.Lock()
SESSIONS: Dict[str, Any] = {}
SESSIONS_LOCK = threading.Lock()
SESSION_COUNTER = 0
FAQ_SCENARIOS = {"faq_coupon", "faq_general", "faq_hours", "faq_member", "faq_store", "ood"}


def choose_scenario(text: str) -> str:
    q = text.lower()
    if any(k in q for k in ["修手机", "股票", "天气", "论文", "租房", "招人", "话费"]):
        return "ood"
    # 售后/订单优先;这些才应该走多轮工具 Agent。
    if any(k in q for k in ["订单", "外卖", "售后", "退款", "退货", "赔偿", "洒", "漏", "破损", "少件", "做错", "超时", "迟到"]):
        if any(k in q for k in ["做错", "口味", "重做"]):
            return "pickup_remake"
        if any(k in q for k in ["晚", "超时", "迟到"]):
            return "late_delivery"
        return "multi_issue"
    if any(k in q for k in ["营业", "几点", "开门", "关门", "营业时间"]):
        return "faq_hours"
    if any(k in q for k in ["积分", "会员"]):
        return "faq_member"
    if any(k in q for k in ["门店", "wifi", "地址", "附近"]):
        return "faq_store"
    if any(k in q for k in ["优惠", "新客", "券"]):
        return "faq_coupon"
    # 价格/推荐/定制/过敏原等通用商品咨询:走单轮 FAQ 模型,不问订单号。
    return "faq_general"


class AgentEngine:
    def __init__(self, model_name: str, adapter: str, max_new_tokens: int = 64):
        print(f"加载模型 {model_name} + {adapter} ...", flush=True)
        self.model, self.tokenizer, _ = load_model_and_tokenizer(
            model_name,
            adapter=adapter,
            for_training=False,
            device_map="cuda",
        )
        self.model.eval()
        self.model.config.use_cache = True
        self.max_new_tokens = max_new_tokens
        self.stop_ids = [self.tokenizer.eos_token_id]
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end != self.tokenizer.unk_token_id:
            self.stop_ids.append(im_end)

    def generate(self, history: List[Dict[str, str]]) -> str:
        prompt = render_chat(
            self.tokenizer,
            history,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        with MODEL_LOCK:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.stop_ids,
                )
        gen = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


class FaqEngine:
    """单轮 SFT 负责 FAQ/OOD,避免多轮 Agent 对非订单问题误问订单号。"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        adapter: str = "runs/2026-10-12/models/sft",
        max_new_tokens: int = 96,
    ):
        print(f"加载 FAQ 单轮模型 {model_name} + {adapter} ...", flush=True)
        self.model = build_single_turn_model(model_name, adapter)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.stop_ids = resolve_stop_ids(self.tokenizer)
        self.max_new_tokens = max_new_tokens

    def answer(self, question: str) -> str:
        prompt = build_single_turn_prompt(self.tokenizer, question, "chat")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with MODEL_LOCK:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=4,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.stop_ids,
                )
        text = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        return clean_answer(text)


def clean_answer(text: str) -> str:
    """截掉 FAQ 常见复读尾巴,保留前 1–2 句。"""
    parts = re.split(r"(?<=[。!?])", text)
    kept = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        kept.append(part)
        if len(kept) >= 2:
            break
    cleaned = "".join(kept).strip()
    return (cleaned or text.strip())[:180]


class ChatSession:
    def __init__(self, engine: AgentEngine, faq_engine: Any, scenario: str, index: int):
        self.engine = engine
        self.faq_engine = faq_engine
        self.scenario = scenario
        self.index = index
        self.episode = None
        self.env = None
        self.history = []
        self.display: List[Dict[str, str]] = []
        self.debug: List[Dict[str, Any]] = []
        self.started = False
        self.done = False
        self.success = False
        self.pending_slot = None
        self.turns = 0
        self.faq_mode = False
        if scenario != "auto":
            self._init_episode(scenario)

    def _init_episode(self, scenario: str):
        self.scenario = scenario
        self.faq_mode = scenario in FAQ_SCENARIOS and self.faq_engine is not None
        self.episode = make_episode(SCENARIOS[scenario], self.index)
        self.env = CustomerServiceEnv(self.episode)
        self.history = self.env.reset()
        self._apply_system_hint()

    def _apply_system_hint(self):
        if self.env is None:
            return
        if self.scenario in {"faq_coupon", "faq_general", "faq_hours", "faq_member", "faq_store", "ood"}:
            hint = (
                f"当前场景:{SCENARIO_LABELS.get(self.scenario, self.scenario)}。"
                "这类咨询直接查询知识库并给出答案,或直接拒答;不需要索取额外信息。"
            )
        else:
            hint = (
                f"当前场景:{SCENARIO_LABELS.get(self.scenario, self.scenario)}。"
                "售后处理需要核对订单时,先询问订单号。"
            )
        self.env.messages[0]["content"] += "\n\n" + hint

    def context(self) -> Dict[str, Any]:
        if self.faq_mode:
            return {
                "scenario": self.scenario,
                "scenario_label": SCENARIO_LABELS.get(self.scenario, self.scenario),
                "order_id": "无需订单",
                "product": "-",
                "price": "-",
                "channel": "-",
                "minutes": "-",
                "issue": SCENARIO_LABELS.get(self.scenario, self.scenario),
                "photo": "-",
                "member_id": "-",
                "member_tier": "-",
                "points": "-",
                "coupon": "-",
            }
        if self.env is None:
            return {
                "scenario": self.scenario,
                "scenario_label": "自动判断",
                "order_id": "发送第一条消息后生成",
                "product": "-",
                "price": "-",
                "channel": "-",
                "minutes": "-",
                "issue": "-",
                "photo": "-",
                "member_id": "-",
                "member_tier": "-",
                "points": "-",
                "coupon": "-",
            }
        order = self.env.order
        return {
            "scenario": self.scenario,
            "scenario_label": SCENARIO_LABELS.get(self.scenario, self.scenario),
            "order_id": order.order_id,
            "product": order.product,
            "price": order.price,
            "channel": "外卖" if order.channel == "delivery" else "自取",
            "minutes": order.minutes_since_delivery,
            "issue": ISSUE_LABELS.get(order.issue, order.issue),
            "photo": "已上传" if order.photo_uploaded else "未上传",
            "member_id": order.user_id,
            "member_tier": order.member_tier,
            "points": order.points,
            "coupon": order.coupon_desc or "无",
        }

    def start(self, text: str):
        if self.env is None:
            self._init_episode(choose_scenario(text) if self.scenario == "auto" else self.scenario)
        self.started = True
        if self.faq_mode:
            self.display.append({"role": "user", "content": text})
            answer = self.faq_engine.answer(text)
            self.display.append({"role": "assistant", "content": answer})
            self.done = True
            self.success = True
            return
        self.episode.initial_message = text
        self.env.reset()
        self._apply_system_hint()
        # reset 会 deepcopy episode.order;在 reset 后应用顾客在首句里已经给出的事实。
        order_match = re.search(r"ORD\d+", text, flags=re.IGNORECASE)
        if order_match:
            self.env.order.order_id = order_match.group(0).upper()
            self.env.known_slots.add("order_id")
        if any(k in text for k in ("拍照", "上传", "照片", "图片")):
            self.env.order.photo_uploaded = True
            self.env.known_slots.add("photo")
        self.history = list(self.env.messages)
        self.display.append({"role": "user", "content": text})
        self._run_agent()

    def user_message(self, text: str) -> Dict[str, Any]:
        if self.done:
            return self.state()
        if not self.started:
            self.start(text)
            return self.state()
        if self.faq_mode:
            self.start(text)
            return self.state()
        self.display.append({"role": "user", "content": text})
        if self.pending_slot:
            obs = self.env.reply_as_user(text)
            self.history.append(obs)
            self.pending_slot = None
        else:
            self.history.append({"role": "user", "content": text})
        self._run_agent()
        return self.state()

    def _run_agent(self):
        # 一次顾客消息后,模型可能连续调多个工具,直到需要顾客回答或终局。
        for _ in range(6):
            if self.done:
                return
            raw = self.engine.generate(self.history)
            self.history.append({"role": "assistant", "content": raw or "{}"})
            parsed, err = parse_action(raw)
            self.debug.append({"raw": raw, "error": err})
            if parsed is None:
                self.display.append(
                    {"role": "assistant", "content": "抱歉,我这边没有理解,请再说一次。"}
                )
                self.pending_slot = None
                return

            result = self.env.step(parsed, auto_user=False)
            self.turns += 1
            if result.observation:
                self.history.append(result.observation)
                self.display.append(
                    {
                        "role": "tool",
                        "content": tool_summary(parsed, result.observation.get("content", "")),
                    }
                )
                continue

            if result.info.get("need_user"):
                slot = result.info.get("slot")
                self.pending_slot = slot
                question = parsed.get("question") or "请补充相关信息。"
                self.display.append({"role": "assistant", "content": question})
                return

            if result.done:
                self.done = True
                self.success = result.success
                content = parsed.get("content", "")
                self.display.append({"role": "assistant", "content": content})
                return

        self.done = True
        self.display.append(
            {"role": "assistant", "content": "抱歉,这次会话步骤太多了,我先为您转人工客服。"}
        )

    def state(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "display": self.display,
            "context": self.context(),
            "done": self.done,
            "success": self.success,
            "pending_slot": self.pending_slot,
            "pending_label": SLOT_LABELS.get(self.pending_slot, ""),
        }


def tool_summary(action: Dict[str, Any], raw: str) -> str:
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    name = action.get("name", "")
    if name == "lookup_order" and data.get("ok"):
        return (
            f"已查到订单 {data.get('order_id')}:{data.get('product')} / "
            f"{'外卖' if data.get('channel') == 'delivery' else '自取'} / "
            f"{data.get('minutes_since_delivery')} 分钟前 / "
            f"{ISSUE_LABELS.get(data.get('issue'), data.get('issue'))} / "
            f"照片{'已上传' if data.get('photo_uploaded') else '未上传'}"
        )
    if name == "check_refund_policy":
        return f"政策判断:{data.get('reason', data.get('error', raw))}"
    if name == "get_member_info" and data.get("ok"):
        return (
            f"会员信息:{data.get('tier')} / {data.get('points')} 积分 / "
            f"优惠券:{data.get('coupon_desc') or '无'}"
        )
    if name == "search_kb":
        return f"知识库:{data.get('answer', raw)}"
    return data.get("error") or data.get("reason") or raw


def new_session(engine: AgentEngine, faq_engine: Any, scenario: str) -> Dict[str, Any]:
    global SESSION_COUNTER
    SESSION_COUNTER += 1
    sid = uuid.uuid4().hex
    session = ChatSession(engine, faq_engine, scenario, 3000 + SESSION_COUNTER)
    with SESSIONS_LOCK:
        SESSIONS[sid] = session
    payload = session.state()
    payload["session_id"] = sid
    return payload


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>茶语时光 · 客服体验</title>
<style>
:root{
  --bg:#f4efe6; --panel:#fffdf8; --ink:#2c241d; --muted:#8a7d6e;
  --brand:#8a5a3b; --brand2:#c98a5b; --line:#e6dccd; --tool:#eee8dd;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 20% 0%,#fff9ef 0,#f4efe6 45%,#e9dfd0 100%);
     color:var(--ink);font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif}
.app{display:grid;grid-template-columns:320px 1fr;min-height:100vh;max-width:1280px;margin:0 auto;
     background:rgba(255,253,248,.72);box-shadow:0 10px 60px rgba(70,45,25,.12)}
.sidebar{padding:28px 24px;border-right:1px solid var(--line);background:rgba(255,250,242,.8)}
.brand{font-size:24px;font-weight:700;letter-spacing:.06em}
.brand small{display:block;font-size:12px;color:var(--muted);font-weight:400;margin-top:4px;letter-spacing:.12em}
.section-label{font-size:12px;color:var(--muted);margin:24px 0 8px;letter-spacing:.08em}
select,button{font:inherit}
select{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:#fff;color:var(--ink)}
button{border:0;border-radius:12px;padding:11px 14px;cursor:pointer;background:var(--brand);color:#fff}
button.ghost{background:#fff;color:var(--brand);border:1px solid var(--line)}
button:disabled{opacity:.5;cursor:not-allowed}
.new-btn{width:100%;margin-top:12px}
.order-card{margin-top:18px;background:#fff;border:1px solid var(--line);border-radius:18px;padding:16px;
            box-shadow:0 8px 30px rgba(80,50,20,.06)}
.order-card h3{margin:0 0 12px;font-size:15px}
.order-row{display:flex;justify-content:space-between;gap:10px;padding:7px 0;border-top:1px dashed #eee4d6;font-size:13px}
.order-row:first-of-type{border-top:0}
.order-row span:first-child{color:var(--muted);white-space:nowrap}
.order-row span:last-child{text-align:right;font-weight:600}
.hint{margin-top:14px;font-size:12px;line-height:1.7;color:var(--muted);background:#f8f1e7;border-radius:14px;padding:12px}
.main{display:flex;flex-direction:column;min-height:100vh}
.topbar{padding:24px 30px 16px;border-bottom:1px solid var(--line)}
.topbar h2{margin:0;font-size:18px}
.topbar p{margin:6px 0 0;color:var(--muted);font-size:13px}
.chat{flex:1;overflow:auto;padding:26px 30px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:76%;padding:12px 15px;border-radius:16px;line-height:1.65;font-size:14px;white-space:pre-wrap;
     animation:pop .18s ease-out}
@keyframes pop{from{transform:translateY(4px);opacity:.4}to{transform:none;opacity:1}}
.msg.user{align-self:flex-end;background:var(--brand);color:#fff;border-bottom-right-radius:4px}
.msg.assistant{align-self:flex-start;background:#fff;border:1px solid var(--line);border-bottom-left-radius:4px}
.msg.tool{align-self:center;max-width:88%;background:var(--tool);color:#5c5044;border-radius:12px;
          font-size:12.5px;padding:9px 14px;border:1px dashed #d6c9b8}
.tool-label{font-size:10px;letter-spacing:.12em;color:#9a8977;margin-bottom:4px}
.composer{padding:16px 24px 24px;border-top:1px solid var(--line);background:rgba(255,253,248,.9)}
.composer form{display:flex;gap:10px}
.composer input{flex:1;padding:14px 16px;border:1px solid var(--line);border-radius:14px;font-size:14px;outline:none;background:#fff}
.composer input:focus{border-color:var(--brand2);box-shadow:0 0 0 3px rgba(201,138,91,.15)}
.status{min-height:20px;margin-top:9px;font-size:12px;color:var(--muted)}
.badge{display:inline-block;padding:3px 8px;border-radius:999px;background:#e9f5ec;color:#2d6b3b;font-size:11px}
.badge.bad{background:#fbe9e7;color:#9b3b32}
@media(max-width:820px){
  .app{grid-template-columns:1fr}
  .sidebar{border-right:0;border-bottom:1px solid var(--line)}
  .chat{min-height:55vh}
}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">茶语时光<small>CUSTOMER SERVICE LAB</small></div>
    <div class="section-label">演示场景</div>
    <select id="scenario"></select>
    <button class="new-btn ghost" id="newBtn">开始新会话</button>
    <div class="section-label">你的订单</div>
    <div class="order-card" id="orderCard"></div>
    <div class="hint" id="hint">你是顾客,直接输入你的问题。客服可能会向你要订单号或照片,信息在左侧订单卡里。</div>
  </aside>
  <main class="main">
    <div class="topbar">
      <h2>客服对话体验</h2>
      <p>你扮演顾客 · 客服由 Qwen3-0.6B 多轮工具 Agent 扮演</p>
    </div>
    <div class="chat" id="chat"></div>
    <div class="composer">
      <form id="form">
        <input id="input" autocomplete="off" placeholder="输入你的问题,例如:我的外卖洒了,怎么退?">
        <button id="sendBtn" type="submit">发送</button>
      </form>
      <div class="status" id="status">正在连接客服...</div>
    </div>
  </main>
</div>
<script>
const scenarioLabels = {
  auto:"自动判断意图",
  multi_issue:"外卖洒漏 + 超时 + 少件",
  delivery_no_photo:"外卖洒漏,尚未拍照",
  pickup_remake:"自取饮品做错口味",
  faq_coupon:"新客优惠咨询",
  faq_general:"商品/价格/推荐咨询",
  faq_hours:"营业时间咨询",
  faq_member:"会员积分咨询",
  faq_store:"门店/WiFi 咨询",
  ood:"知识库外问题",
  late_delivery:"外卖超时"
};
const scenarioSelect = document.getElementById("scenario");
for (const [key,label] of Object.entries(scenarioLabels)) {
  const opt = document.createElement("option");
  opt.value = key; opt.textContent = label; scenarioSelect.appendChild(opt);
}
const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const statusEl = document.getElementById("status");
const orderEl = document.getElementById("orderCard");
let sessionId = null;
let busy = false;
let done = false;

function esc(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function renderOrder(ctx){
  const rows = [
    ["订单号",ctx.order_id],["商品",ctx.product],["渠道",ctx.channel],
    ["送达",ctx.minutes+" 分钟前"],["问题",ctx.issue],["照片",ctx.photo],
    ["会员号",ctx.member_id],["会员",ctx.member_tier],["积分",ctx.points],["优惠券",ctx.coupon]
  ];
  orderEl.innerHTML = "<h3>顾客信息</h3>" + rows.map(([k,v]) =>
    `<div class="order-row"><span>${k}</span><span>${esc(v)}</span></div>`).join("");
}
function render(data){
  chatEl.innerHTML = "";
  for (const m of data.display) {
    const div = document.createElement("div");
    div.className = "msg " + m.role;
    if (m.role === "tool") {
      div.innerHTML = `<div class="tool-label">系统查询</div>${esc(m.content)}`;
    } else {
      div.textContent = m.content;
    }
    chatEl.appendChild(div);
  }
  chatEl.scrollTop = chatEl.scrollHeight;
  renderOrder(data.context);
  done = data.done;
  inputEl.disabled = done || busy;
  sendBtn.disabled = done || busy;
  if (done) {
    statusEl.innerHTML = data.success
      ? '<span class="badge">会话完成:客服已解决</span>'
      : '<span class="badge bad">会话结束:未完全解决</span>';
    inputEl.placeholder = "本次会话已结束,点击“开始新会话”";
    inputEl.focus();
  } else if (data.pending_slot) {
    statusEl.textContent = "客服正在等待你提供:" + (data.pending_label || data.pending_slot);
    inputEl.placeholder = data.pending_slot === "order_id"
      ? "例如:我的订单号是 " + data.context.order_id
      : (data.pending_slot === "photo" ? "例如:我已经拍照上传了" : "请输入客服需要的信息");
    inputEl.focus();
  } else {
    statusEl.textContent = "可以继续输入";
  }
}
async function newSession(){
  busy = true; inputEl.disabled = true; sendBtn.disabled = true;
  statusEl.textContent = "正在准备新会话...";
  const res = await fetch("/api/new", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({scenario: scenarioSelect.value})
  });
  const data = await res.json();
  sessionId = data.session_id;
  busy = false;
  render(data);
  if(!data.display.length){
    chatEl.innerHTML = '<div class="msg assistant">您好,我是茶语时光客服。请直接描述您遇到的问题。</div>';
  }
  inputEl.focus();
}
async function send(){
  const text = inputEl.value.trim();
  if(!text || busy || done || !sessionId) return;
  inputEl.value = "";
  busy = true; sendBtn.disabled = true; inputEl.disabled = true;
  statusEl.textContent = "客服正在输入...";
  const temp = document.createElement("div");
  temp.className = "msg user"; temp.textContent = text;
  chatEl.appendChild(temp); chatEl.scrollTop = chatEl.scrollHeight;
  try {
    const res = await fetch("/api/chat", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({session_id: sessionId, message: text})
    });
    const data = await res.json();
    if(data.error) throw new Error(data.error);
    busy = false;
    render(data);
  } catch(err) {
    busy = false;
    statusEl.textContent = "请求失败:" + err.message;
    inputEl.disabled = false; sendBtn.disabled = false;
  }
}
document.getElementById("form").addEventListener("submit", e => { e.preventDefault(); send(); });
document.getElementById("newBtn").addEventListener("click", newSession);
scenarioSelect.addEventListener("change", newSession);
newSession();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    engine: AgentEngine = None
    faq_engine: Any = None

    def log_message(self, fmt, *args):
        return

    def _send_json(self, payload: Dict[str, Any], code: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/health":
            self._send_json({"ok": True})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/new":
                scenario = payload.get("scenario", "multi_issue")
                if scenario not in SCENARIOS and scenario != "auto":
                    scenario = "auto"
                self._send_json(new_session(self.engine, self.faq_engine, scenario))
                return
            if self.path == "/api/chat":
                sid = payload.get("session_id")
                message = str(payload.get("message", "")).strip()
                with SESSIONS_LOCK:
                    session = SESSIONS.get(sid)
                if session is None:
                    self._send_json({"error": "session not found"}, 404)
                    return
                if not message:
                    self._send_json({"error": "empty message"}, 400)
                    return
                self._send_json(session.user_message(message))
                return
            self._send_json({"error": "not found"}, 404)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    parser = argparse.ArgumentParser(description="网页客服体验")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", type=str, default="models/multiturn_grpo_turn_h8_scaled")
    parser.add_argument("--faq-adapter", type=str, default="runs/2026-10-12/models/sft")
    parser.add_argument("--no-faq", action="store_true", help="禁用单轮 FAQ 模型")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    engine = AgentEngine(args.model_name, args.adapter, args.max_new_tokens)
    Handler.engine = engine
    if not args.no_faq:
        try:
            Handler.faq_engine = FaqEngine(adapter=args.faq_adapter)
        except Exception as exc:
            print(f"FAQ 单轮模型加载失败,回退到多轮 Agent: {exc}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"客服体验已启动: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

