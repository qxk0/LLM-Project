"""数据血缘(leakage)审计工具:文本归一化、稳定 ID、多层泄漏检查、报告生成。

设计原则:
  1. 只读:本模块不修改任何数据文件,只做计算与报告输出。
  2. 可复现:所有 ID 都由内容 hash 派生,相同输入必得相同 ID / 相同报告。
  3. 可审计:每一条泄漏指标都同时给出"值集合交集大小"和"行级命中数",避免口径歧义。

三层泄漏口径(由细到粗):
  * exact  : normalized question(小写 + 去空白 + 统一中英文标点)完全相同
  * family : family_id 相同(= intent + base_question 相同,prefix 已剥离)
  * group  : split_group_id 相同(= 同一实体 / 同一 seed 的血缘连通分量,最严格)

可单独执行对任意数据目录做审计:
  python data_eng/leakage_audit.py --data-dir data_eng/output_clean --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------- 文本归一化

# 中英文标点统一:全部映射到半角
PUNCT_TABLE = str.maketrans(
    {
        "？": "?",
        "，": ",",
        "。": ".",
        "！": "!",
        "：": ":",
        "；": ";",
        "、": ",",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "～": "~",
        "％": "%",
        "　": " ",
    }
)

# prefix paraphrase 模板产生的前缀(长的放前面,避免"请问"先吃掉"请问哪里")
PARAPHRASE_PREFIXES = (
    "麻烦问一下,",
    "麻烦问一下，",
    "你好,",
    "你好，",
    "请问",
    "亲,",
    "亲，",
)

SPLIT_NAMES = ("train", "val", "test")
MODEL_GENERATED_SOURCES = ("evolved", "onpolicy")

AUDIT_FIELDS = (
    "sample_id",
    "source",
    "variant_type",
    "intent",
    "family_id",
    "split_group_id",
    "base_question",
)


def normalize_text(text) -> str:
    """小写 + 统一中英文标点 + 去掉所有空白。"""
    t = str(text).lower().translate(PUNCT_TABLE)
    t = re.sub(r"\s+", "", t)
    return t.strip()


def normalize(text) -> str:
    """兼容旧接口(build_dataset.normalize / evolve.py 曾使用)。"""
    return normalize_text(text)


def strip_paraphrase_prefix(text, prefixes=PARAPHRASE_PREFIXES) -> str:
    """反复剥离 prefix paraphrase 前缀,直到不再变化。

    必须反复剥离:模板本身可能已含"请问"(例如 "请问{name}的价格是?"),
    paraphrase 之后会出现 "请问请问..."。
    """
    cur = str(text).strip()
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if cur.startswith(p) and len(cur) > len(p) + 1:
                cur = cur[len(p):].strip()
                changed = True
                break
    return cur


def base_question(text) -> str:
    """去掉口语化前缀后的"基础问题",是 family 的判定依据。"""
    return strip_paraphrase_prefix(text)


# ---------------------------------------------------------------- 稳定 ID


def stable_id(prefix: str, *parts, width: int = 12) -> str:
    payload = "\u241f".join(str(p) for p in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:width]}"


def family_id_of(intent: str, base_q: str) -> str:
    """family = (intent, 基础问题);同一基础问题的所有改写共享同一 family。"""
    return stable_id("fam_", intent, normalize_text(base_q))


def sample_id_of(intent: str, q: str, a: str) -> str:
    """sample_id 由内容派生:同一 (intent, q, a) 在任意一次重建中 ID 都不变。"""
    return stable_id("s_", intent, normalize_text(q), normalize_text(a))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- IO


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 审计核心


def _q_set(rows):
    return {normalize_text(r["q"]) for r in rows}


def _base_set(rows):
    return {normalize_text(r.get("base_question") or base_question(r["q"])) for r in rows}


def _family_set(rows):
    return {r["family_id"] for r in rows if r.get("family_id")}


def _group_set(rows):
    return {r["split_group_id"] for r in rows if r.get("split_group_id")}


def _ood_rows(rows):
    return [r for r in rows if r.get("ood")]


def _ood_family_set(rows):
    return {r["family_id"] for r in _ood_rows(rows) if r.get("family_id")}


def _ood_entity_set(rows):
    return {r.get("entity_key") for r in _ood_rows(rows) if r.get("entity_key")}


def _counter(values):
    return dict(sorted(Counter(values).items(), key=lambda kv: (-kv[1], str(kv[0]))))


def _pairwise_overlap(set_fn, splits):
    value_sets = {name: set_fn(rows) for name, rows in splits.items()}
    overlap = {}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            overlap[f"{a}-{b}"] = len(value_sets[a] & value_sets[b])
    return overlap


def _sample_stats(rows):
    ood = _ood_rows(rows)
    return {
        "samples": len(rows),
        "unique_questions": len(_q_set(rows)),
        "unique_base_questions": len(_base_set(rows)),
        "unique_families": len(_family_set(rows)),
        "unique_groups": len(_group_set(rows)),
        "unique_intents": len({r.get("intent") for r in rows}),
        "ood_samples": len(ood),
        "ood_families": len(_ood_family_set(rows)),
        "ood_entities": sorted(x for x in _ood_entity_set(rows) if x),
    }


def audit_splits(
    splits: dict,
    *,
    dataset_name: str = "clean",
    inputs: dict | None = None,
    raw_rows: list | None = None,
    lineage: dict | None = None,
    notes: list | None = None,
) -> dict:
    """对 {train,val,test} 做多层泄漏审计;splits 的值是行 list。"""
    splits = {name: list(splits.get(name, [])) for name in SPLIT_NAMES}

    # ---- 字段完整性
    missing = defaultdict(int)
    for name, rows in splits.items():
        for r in rows:
            for field in AUDIT_FIELDS:
                if r.get(field) in (None, ""):
                    missing[f"{name}.{field}"] += 1

    exact_overlap = _pairwise_overlap(_q_set, splits)
    base_overlap = _pairwise_overlap(_base_set, splits)
    family_overlap = _pairwise_overlap(_family_set, splits)
    group_overlap = _pairwise_overlap(_group_set, splits)
    ood_family_overlap = _pairwise_overlap(_ood_family_set, splits)
    ood_entity_overlap = _pairwise_overlap(_ood_entity_set, splits)

    # ---- 行级污染面:另一个 split 有多少行"撞上"了本 split 的值
    exact_row_hits, base_row_hits, family_row_hits = {}, {}, {}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a == b:
                continue
            other_q = _q_set(splits[b])
            other_b = _base_set(splits[b])
            other_f = _family_set(splits[b])
            exact_row_hits[f"{a}<-{b}"] = sum(1 for r in splits[a] if normalize_text(r["q"]) in other_q)
            base_row_hits[f"{a}<-{b}"] = sum(
                1 for r in splits[a]
                if normalize_text(r.get("base_question") or base_question(r["q"])) in other_b
            )
            family_row_hits[f"{a}<-{b}"] = sum(1 for r in splits[a] if r.get("family_id") in other_f)

    # ---- split 内部重复(信息项,不参与断言)
    dup_within = {}
    for name, rows in splits.items():
        qa = Counter((normalize_text(r["q"]), normalize_text(r["a"])) for r in rows)
        qq = Counter(normalize_text(r["q"]) for r in rows)
        dup_within[name] = {
            "duplicate_qa_pairs": sum(c - 1 for c in qa.values() if c > 1),
            "duplicate_questions": sum(c - 1 for c in qq.values() if c > 1),
        }

    source_dist = {name: _counter(r.get("source", "<missing>") for r in rows) for name, rows in splits.items()}
    intent_dist = {name: _counter(r.get("intent", "<missing>") for r in rows) for name, rows in splits.items()}
    variant_dist = {name: _counter(r.get("variant_type", "<missing>") for r in rows) for name, rows in splits.items()}

    test_model_generated = {src: source_dist["test"].get(src, 0) for src in MODEL_GENERATED_SOURCES}
    val_model_generated = {src: source_dist["val"].get(src, 0) for src in MODEL_GENERATED_SOURCES}

    unknown_rows = [r for rows in splits.values() for r in rows if r.get("lineage_confidence") == "unknown"]
    ambiguous_rows = [
        r for rows in splits.values() for r in rows if r.get("lineage_confidence") == "ambiguous_union"
    ]

    raw_stats = None
    if raw_rows is not None:
        raw_stats = _sample_stats(raw_rows)
        raw_stats["source_distribution"] = _counter(r.get("source", "<missing>") for r in raw_rows)
        raw_stats["intent_distribution"] = _counter(r.get("intent", "<missing>") for r in raw_rows)

    assertions = []

    def _assert(name, passed, detail):
        assertions.append({"name": name, "passed": bool(passed), "detail": detail})

    _assert(
        "exact normalized question never crosses split",
        all(v == 0 for v in exact_overlap.values()),
        f"overlap={exact_overlap}",
    )
    _assert(
        "family never crosses split",
        all(v == 0 for v in family_overlap.values()),
        f"overlap={family_overlap}",
    )
    _assert(
        "base question never crosses split",
        all(v == 0 for v in base_overlap.values()),
        f"overlap={base_overlap}",
    )
    _assert(
        "split group never crosses split",
        all(v == 0 for v in group_overlap.values()),
        f"overlap={group_overlap}",
    )
    _assert(
        "OOD family never crosses split",
        all(v == 0 for v in ood_family_overlap.values()),
        f"overlap={ood_family_overlap} (train={len(_ood_family_set(splits['train']))}, "
        f"val={len(_ood_family_set(splits['val']))}, test={len(_ood_family_set(splits['test']))})",
    )
    _assert(
        "OOD entity/topic never crosses split",
        all(v == 0 for v in ood_entity_overlap.values()),
        f"overlap={ood_entity_overlap}",
    )
    _assert(
        "test contains 0 model-generated samples (onpolicy/evolved)",
        all(v == 0 for v in test_model_generated.values()),
        f"test model-generated = {test_model_generated}",
    )
    _assert(
        "val contains 0 model-generated samples (onpolicy/evolved)",
        all(v == 0 for v in val_model_generated.values()),
        f"val model-generated = {val_model_generated}",
    )
    _assert(
        "all samples carry complete lineage metadata",
        not missing,
        "no missing fields" if not missing else f"missing_fields={dict(missing)}",
    )
    _assert(
        "train/val/test all non-empty",
        all(len(splits[name]) > 0 for name in SPLIT_NAMES),
        f"sizes={{k: len(v) for k, v in splits.items()}}".replace("{k: len(v) for k, v in splits.items()}", str({n: len(r) for n, r in splits.items()})),
    )
    _assert(
        "no unknown-lineage sample in val/test",
        all(r.get("lineage_confidence") != "unknown" for name in ("val", "test") for r in splits[name]),
        f"unknown_lineage_samples_total={len(unknown_rows)}",
    )

    all_intents = sorted({r.get("intent") for rows in splits.values() for r in rows})
    intent_coverage = {name: sorted({r.get("intent") for r in rows}) for name, rows in splits.items()}
    missing_intents = {name: sorted(set(all_intents) - set(intent_coverage[name])) for name in SPLIT_NAMES}

    audit = {
        "dataset": dataset_name,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": inputs or {},
        "raw_dataset": raw_stats,
        "split": {name: _sample_stats(splits[name]) for name in SPLIT_NAMES},
        "source_distribution": source_dist,
        "variant_distribution": variant_dist,
        "intent_distribution": intent_dist,
        "intent_coverage": intent_coverage,
        "missing_intents_per_split": missing_intents,
        "ood": {
            "counts": {name: _sample_stats(splits[name])["ood_samples"] for name in SPLIT_NAMES},
            "families": {name: _sample_stats(splits[name])["ood_families"] for name in SPLIT_NAMES},
            "entities": {name: _sample_stats(splits[name])["ood_entities"] for name in SPLIT_NAMES},
            "family_overlap": ood_family_overlap,
            "entity_overlap": ood_entity_overlap,
        },
        "leakage": {
            "exact_question_overlap": exact_overlap,
            "base_question_overlap": base_overlap,
            "family_overlap": family_overlap,
            "split_group_overlap": group_overlap,
            "row_level": {
                "exact_question_rows": exact_row_hits,
                "base_question_rows": base_row_hits,
                "family_rows": family_row_hits,
            },
            "duplicate_within_split": dup_within,
        },
        "test_purity": {
            "model_generated_sources": test_model_generated,
            "test_onpolicy": source_dist["test"].get("onpolicy", 0),
            "test_evolved": source_dist["test"].get("evolved", 0),
        },
        "unknown_lineage": {
            "unknown_family_count": len(unknown_rows),
            "ambiguous_lineage_count": len(ambiguous_rows),
            "unknown_by_source": _counter(r.get("source", "<missing>") for r in unknown_rows),
            "ambiguous_by_source": _counter(r.get("source", "<missing>") for r in ambiguous_rows),
        },
        "lineage": lineage or {},
        "assertions": assertions,
        "all_passed": all(a["passed"] for a in assertions),
        "notes": notes or [],
    }
    return audit


# ---------------------------------------------------------------- 报告


def _table(counter_dict, key_header="key"):
    lines = [f"| {key_header} | count |", "|---|---|"]
    if not counter_dict:
        lines.append("| (empty) | 0 |")
    for k, v in counter_dict.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def render_markdown(audit: dict) -> str:
    o = []
    o.append("# 数据集划分与泄漏审计报告 (DATASET LEAKAGE AUDIT)")
    o.append("")
    o.append(f"- dataset: `{audit['dataset']}`")
    o.append(f"- generated_at: {audit['generated_at']}")
    for k, v in (audit.get("inputs") or {}).items():
        o.append(f"- input `{k}`: {v}")
    o.append(f"- **all assertions passed: `{audit['all_passed']}`**")
    if audit.get("reused_build_sections"):
        o.append(
            "- 注:本次为独立重跑审计,`raw dataset / lineage / notes / legacy_baseline` "
            "段落沿用最近一次完整构建(`build_dataset.py`)的输出。"
        )
    o.append("")

    if audit.get("notes"):
        o.append("## 0. 说明与已知残余风险")
        o.append("")
        for n in audit["notes"]:
            o.append(f"- {n}")
        o.append("")

    raw = audit.get("raw_dataset")
    if raw:
        o.append("## 1. Raw dataset(去重清洗后、切分前)")
        o.append("")
        o.append("```text")
        o.append(f"total samples           : {raw['samples']}")
        o.append(f"unique questions        : {raw['unique_questions']}")
        o.append(f"unique base questions   : {raw['unique_base_questions']}")
        o.append(f"unique families         : {raw['unique_families']}")
        o.append(f"unique split groups     : {raw['unique_groups']}")
        o.append(f"unique intents          : {raw['unique_intents']}")
        o.append(f"OOD samples / families  : {raw['ood_samples']} / {raw['ood_families']}")
        o.append("```")
        o.append("")
        o.append("来源分布(切分前):")
        o.append("")
        o.append(_table(raw["source_distribution"], "source"))
        o.append("")

    o.append("## 2. Split")
    o.append("")
    o.append("| split | samples | questions | base questions | families | groups | intents | OOD samples | OOD families |")
    o.append("|---|---|---|---|---|---|---|---|---|")
    for name in SPLIT_NAMES:
        s = audit["split"][name]
        o.append(
            f"| {name} | {s['samples']} | {s['unique_questions']} | {s['unique_base_questions']} | "
            f"{s['unique_families']} | {s['unique_groups']} | {s['unique_intents']} | "
            f"{s['ood_samples']} | {s['ood_families']} |"
        )
    o.append("")

    o.append("## 3. Source distribution")
    o.append("")
    for name in SPLIT_NAMES:
        o.append(f"### {name}")
        o.append("")
        o.append(_table(audit["source_distribution"][name], "source"))
        o.append("")

    o.append("## 4. Intent distribution")
    o.append("")
    for name in SPLIT_NAMES:
        o.append(f"### {name}")
        o.append("")
        o.append(_table(audit["intent_distribution"][name], "intent"))
        o.append("")
    o.append("intent 覆盖(缺失项会被显式列出,不做任何数据修补):")
    o.append("")
    for name in SPLIT_NAMES:
        miss = audit["missing_intents_per_split"][name]
        o.append(f"- {name}: covered={audit['intent_coverage'][name]} | **missing={miss}**")
    o.append("")

    o.append("## 5. OOD distribution")
    o.append("")
    o.append("| split | OOD samples | OOD families | OOD entities |")
    o.append("|---|---|---|---|")
    for name in SPLIT_NAMES:
        o.append(
            f"| {name} | {audit['ood']['counts'][name]} | {audit['ood']['families'][name]} | "
            f"{len(audit['ood']['entities'][name])} |"
        )
    o.append("")
    o.append("交集(必须全为 0):")
    o.append("")
    o.append("```text")
    o.append(f"OOD family overlap : {audit['ood']['family_overlap']}")
    o.append(f"OOD entity overlap : {audit['ood']['entity_overlap']}")
    o.append("```")
    o.append("")
    for name in SPLIT_NAMES:
        ents = audit["ood"]["entities"][name]
        o.append(f"- {name} OOD entities({len(ents)}): {', '.join(ents) if ents else '(none)'}")
    o.append("")

    leak = audit["leakage"]
    o.append("## 6. Leakage")
    o.append("")
    o.append("### 6.1 Exact duplicate (normalized question)")
    o.append("")
    o.append("```text")
    for k, v in leak["exact_question_overlap"].items():
        o.append(f"{k}: {v}")
    o.append("```")
    o.append("")
    o.append("### 6.2 Same-family leakage")
    o.append("")
    o.append("```text")
    for k, v in leak["family_overlap"].items():
        o.append(f"{k}: {v}")
    o.append("```")
    o.append("")
    o.append("### 6.3 Base-question leakage (prefix paraphrase 已剥离)")
    o.append("")
    o.append("```text")
    for k, v in leak["base_question_overlap"].items():
        o.append(f"{k}: {v}")
    o.append("```")
    o.append("")
    o.append("### 6.4 Split-group leakage (实体级 / seed 级连通分量)")
    o.append("")
    o.append("```text")
    for k, v in leak["split_group_overlap"].items():
        o.append(f"{k}: {v}")
    o.append("```")
    o.append("")
    o.append("### 6.5 行级污染面")
    o.append("")
    o.append("```text")
    for label, payload in (
        ("exact question rows", leak["row_level"]["exact_question_rows"]),
        ("base question rows", leak["row_level"]["base_question_rows"]),
        ("family rows", leak["row_level"]["family_rows"]),
    ):
        o.append(f"{label}: {payload}")
    o.append("```")
    o.append("")
    o.append("### 6.6 split 内部重复(信息项)")
    o.append("")
    o.append(
        _table(
            {
                f"{k}.{kk}": vv
                for k, v in leak["duplicate_within_split"].items()
                for kk, vv in v.items()
            },
            "metric",
        )
    )
    o.append("")

    o.append("## 7. Test purity")
    o.append("")
    o.append("```text")
    o.append(f"test onpolicy = {audit['test_purity']['test_onpolicy']}")
    o.append(f"test evolved  = {audit['test_purity']['test_evolved']}")
    o.append(f"test model-generated total = {audit['test_purity']['model_generated_sources']}")
    o.append("```")
    o.append("")

    o.append("## 8. Unknown / ambiguous lineage")
    o.append("")
    u = audit["unknown_lineage"]
    o.append("```text")
    o.append(f"unknown_family_count    = {u['unknown_family_count']}")
    o.append(f"ambiguous_lineage_count = {u['ambiguous_lineage_count']}")
    o.append(f"unknown by source       = {u['unknown_by_source']}")
    o.append(f"ambiguous by source     = {u['ambiguous_by_source']}")
    o.append("```")
    o.append("")
    lin = audit.get("lineage") or {}
    if lin:
        o.append("### lineage 恢复统计")
        o.append("")
        o.append("```text")
        for k, v in lin.items():
            if not isinstance(v, (dict, list)):
                o.append(f"{k} = {v}")
        o.append("```")
        o.append("")
        for k, v in lin.items():
            if isinstance(v, dict):
                o.append(f"**{k}**")
                o.append("")
                o.append(_table({str(kk): vv for kk, vv in v.items()}, "key"))
                o.append("")
            elif isinstance(v, list):
                o.append(f"- {k}: {v}")
        o.append("")

    ge = audit.get("gold_eval")
    if ge:
        o.append("## 9. Gold evaluation set(人工 Gold Evaluation Seed)")
        o.append("")
        o.append("```text")
        o.append(f"gold_eval count        = {ge.get('count')} (expected {ge.get('expected_count')})")
        o.append(f"gold_eval in test      = {ge.get('in_test')}")
        o.append(f"by intent              = {ge.get('by_intent')}")
        o.append(f"test intent counts     = {ge.get('test_intent_counts')}")
        o.append(f"test missing intents   = {ge.get('test_missing_intents')}")
        o.append(f"previous test preserved= {ge.get('previous_test_preserved')}")
        o.append(f"previous test sha256    = {ge.get('previous_test_sha256')}")
        o.append("```")
        o.append("")
        o.append("gold_eval 与 train / val / 旧 test 的多层去重(必须全为 0):")
        o.append("")
        o.append("```text")
        for name, d in (ge.get("overlap") or {}).items():
            o.append(f"gold_eval ∩ {name}: {d}")
        o.append("```")
        o.append("")
        o.append("逐题清单见 `GOLD_EVAL_AUDIT.md`(含 knowledge source / novelty reason / leakage check)。")
        o.append("")
    o.append("## 10. Assertions")
    o.append("")
    o.append("| assertion | result | detail |")
    o.append("|---|---|---|")
    for a in audit["assertions"]:
        detail = str(a["detail"]).replace("|", "/")
        o.append(f"| {a['name']} | {'PASS' if a['passed'] else '**FAIL**'} | {detail} |")
    o.append("")
    return "\n".join(o)


def write_reports(audit: dict, out_dir: str) -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "leakage_audit.json")
    md_path = os.path.join(out_dir, "DATASET_LEAKAGE_AUDIT.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(audit))
    return json_path, md_path


def audit_directory(data_dir: str, dataset_name: str | None = None) -> dict:
    splits = {name: load_jsonl(os.path.join(data_dir, f"{name}.jsonl")) for name in SPLIT_NAMES}
    audit = audit_splits(splits, dataset_name=dataset_name or os.path.basename(os.path.abspath(data_dir)))

    # 单独重跑审计时不丢信息:沿用最近一次完整构建写入的 raw / lineage / notes / legacy 段落
    previous_path = os.path.join(data_dir, "leakage_audit.json")
    if os.path.exists(previous_path):
        try:
            with open(previous_path, encoding="utf-8") as f:
                previous = json.load(f)
        except (json.JSONDecodeError, OSError):
            previous = {}
        carried = [k for k in ("raw_dataset", "lineage", "notes", "legacy_baseline") if previous.get(k)]
        for key in carried:
            audit[key] = previous[key]
        if carried:
            audit["reused_build_sections"] = carried
    return audit


def main():
    parser = argparse.ArgumentParser(description="对 train/val/test 做多层泄漏审计")
    parser.add_argument("--data-dir", type=str, default="data_eng/output_clean")
    parser.add_argument("--write", action="store_true", help="把审计结果写回该目录")
    args = parser.parse_args()

    audit = audit_directory(args.data_dir)
    if args.write:
        json_path, md_path = write_reports(audit, args.data_dir)
        print(f"审计报告已写入: {json_path} / {md_path}")
    print(render_markdown(audit))
    print(f"\nALL ASSERTIONS PASSED: {audit['all_passed']}")
    return 0 if audit["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
