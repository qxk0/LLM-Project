#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_context.py —— 把项目代码打包成适合上传给网页版 LLM（ChatGPT / Claude）的 Markdown 文件。

设计目的：
  本项目里 .venv(第三方库) / models(权重) / runs(训练产物) / data(数据集) 占了 99% 的体积，
  但真正需要 LLM 理解的是自研代码(约 29 个 .py, 7.7k 行)。
  本脚本遵守 .gitignore 语义，只抽取"人写的"文本文件，并按模块切分，
  避免单个文件超过网页版 LLM 的上下文/上传限制。

用法：
  python tools/export_context.py                    # 默认：按模块切分输出到 llm_context/
  python tools/export_context.py --mode single      # 输出为一个大文件
  python tools/export_context.py --with-configs     # 额外附带 models/runs 下的小型 json 配置
  python tools/export_context.py --with-docs        # 额外附带项目文档(md)
  python tools/export_context.py --out D:/tmp/ctx   # 指定输出目录

输出：
  llm_context/00_INDEX.md        导航 + 目录树 + token 估算
  llm_context/10_pretrain.md     预训练模块
  llm_context/20_sft.md          SFT 模块
  llm_context/30_rl.md           RL(DPO/GRPO/RFT) 模块
  llm_context/40_data_eng.md     数据工程模块
  llm_context/50_root.md         顶层文件(README/requirements/start.ps1)
"""

import argparse
import os
import re
import sys
from datetime import datetime

# ---------------------------------------------------------------- 配置

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 目录级黑名单：体积大 / 非自研 / 含隐私
EXCLUDE_DIRS = {
    ".venv", "venv", "env",
    ".git", ".svn",
    ".workbuddy", ".cache", ".idea", ".vscode", ".pytest_cache",
    "__pycache__",
    "models", "runs", "data",          # 权重 / 训练产物 / 数据集
    "output", "output_clean", "logs",  # 数据工程中间产物
    "node_modules", "dist", "build",
}

# 文件级黑名单：密钥与含个人信息的文件，绝不外传
EXCLUDE_FILES = {
    ".env", ".env.local", ".env.production",
    "戚潇琨.docx",                     # 简历，含个人信息
}
EXCLUDE_SUFFIX = {".docx", ".doc", ".xlsx", ".pptx", ".pdf"}

# 允许的文本扩展名
INCLUDE_EXT = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".jinja",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".ps1", ".sh", ".bat",
}

# 单个源文件超过该字节数则只保留头部（防止个别巨型文件撑爆上下文）
MAX_FILE_BYTES = 120_000

# 模块划分：目录 -> (输出文件名, 标题)
MODULES = [
    ("pretrain", "10_pretrain.md", "预训练模块 pretrain/ (SwiGLU + RMSNorm + FlashAttention)"),
    ("sft",      "20_sft.md",      "监督微调模块 sft/ (SFT + 评测)"),
    ("rl",       "30_rl.md",       "对齐模块 rl/ (DPO / GRPO / RFT)"),
    ("data_eng", "40_data_eng.md", "数据工程模块 data_eng/ (数据集构建 + 质检 + 评审)"),
]


# ---------------------------------------------------------------- 工具

def is_excluded_dir(name: str) -> bool:
    return name in EXCLUDE_DIRS or name.startswith(".")


def is_excluded_file(name: str) -> bool:
    if name in EXCLUDE_FILES:
        return True
    if os.path.splitext(name)[1].lower() in EXCLUDE_SUFFIX:
        return True
    if name.endswith(".pyc"):
        return True
    return False


def read_text(path: str):
    """读取文本文件，失败返回 None。"""
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            return f.read()
    except (UnicodeDecodeError, OSError, PermissionError):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return None


def estimate_tokens(text: str) -> int:
    """粗估 token：中文按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    rest = len(text) - cjk
    return int(cjk + rest / 4)


def fence_for(text: str) -> str:
    """根据内容里最长的反引号串动态选择 fence，避免代码块被提前闭合。"""
    longest = 0
    cur = 0
    for ch in text:
        if ch == "`":
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return "`" * max(3, longest + 1)


def ext_lang(ext: str) -> str:
    return {
        ".py": "python", ".md": "markdown", ".json": "json", ".jsonl": "jsonl",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ps1": "powershell",
        ".sh": "bash", ".bat": "bat", ".jinja": "jinja", ".txt": "text",
        ".ini": "ini", ".cfg": "ini",
    }.get(ext, "")


def collect_files(base_dirs, only_ext=None, exclude_ext=None, recursive=True):
    """收集 (相对路径, 绝对路径, 扩展名) 列表，按路径排序。"""
    out = []
    for base in base_dirs:
        abs_base = os.path.join(ROOT, base) if base != "." else ROOT
        if not os.path.isdir(abs_base):
            continue

        def keep(ext):
            if ext not in INCLUDE_EXT:
                return False
            if only_ext is not None and ext not in only_ext:
                return False
            if exclude_ext is not None and ext in exclude_ext:
                return False
            return True

        if base == "." or not recursive:
            # 顶层只取文件，不递归
            for name in sorted(os.listdir(abs_base)):
                p = os.path.join(abs_base, name)
                if not os.path.isfile(p):
                    continue
                if is_excluded_file(name) or name.startswith("."):
                    continue
                if keep(os.path.splitext(name)[1].lower()):
                    out.append((name, p, os.path.splitext(name)[1].lower()))
            continue
        for dirpath, dirnames, filenames in os.walk(abs_base):
            dirnames[:] = [d for d in dirnames if not is_excluded_dir(d)]
            for name in sorted(filenames):
                if is_excluded_file(name):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if not keep(ext):
                    continue
                p = os.path.join(dirpath, name)
                rel = os.path.relpath(p, ROOT).replace("\\", "/")
                out.append((rel, p, ext))
    return sorted(set(out))


def build_tree(base_dirs):
    """生成简洁的目录树（只显示 2 层，避免过长）。"""
    lines = []
    for base in base_dirs:
        if base == ".":
            continue
        abs_base = os.path.join(ROOT, base)
        if not os.path.isdir(abs_base):
            continue
        lines.append(f"{base}/")
        for dirpath, dirnames, filenames in os.walk(abs_base):
            dirnames[:] = [d for d in dirnames if not is_excluded_dir(d)]
            depth = dirpath[len(abs_base):].count(os.sep)
            if depth >= 2:
                dirnames[:] = []
                continue
            indent = "  " * (depth + 1)
            for d in sorted(dirnames):
                lines.append(f"{indent}{d}/")
            for f in sorted(filenames):
                if is_excluded_file(f):
                    continue
                if os.path.splitext(f)[1].lower() in INCLUDE_EXT:
                    lines.append(f"{indent}{f}")
    return "\n".join(lines)


def render_file(rel, abs_path, ext):
    text = read_text(abs_path)
    if text is None:
        return f"### `{rel}`\n\n_（无法读取，可能是二进制文件）_\n", 0

    truncated = False
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) > MAX_FILE_BYTES:
        text = raw[:MAX_FILE_BYTES].decode("utf-8", errors="ignore")
        truncated = True
    if len(raw) > 300_000:
        return f"### `{rel}`\n\n_（文件过大 {len(raw)//1024}KB，已跳过；如需分析请单独上传）_\n", 0

    fence = fence_for(text)
    body = [f"### `{rel}`", "", f"{fence}{ext_lang(ext)}", text.rstrip(), fence, ""]
    if truncated:
        body.insert(1, f"> ⚠️ 文件超过 {MAX_FILE_BYTES//1000}KB，此处仅保留前段。")
        body.insert(2, "")
    return "\n".join(body), estimate_tokens(text)


def write_bundle(out_dir, fname, title, base_dirs, note="", only_ext=None, exclude_ext=None):
    files = collect_files(base_dirs, only_ext=only_ext, exclude_ext=exclude_ext)
    if not files:
        return None

    parts = [f"# {title}", ""]
    if note:
        parts += [note, ""]
    parts += [
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 目录结构",
        "",
        "```text",
        build_tree([d for d in base_dirs if d != "."]),
        "```",
        "",
        "## 文件内容",
        "",
    ]

    total_tokens = 0
    for rel, abs_path, ext in files:
        body, tk = render_file(rel, abs_path, ext)
        parts.append(body)
        total_tokens += tk

    content = "\n".join(parts)
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    size_kb = os.path.getsize(path) / 1024
    return {
        "file": fname,
        "title": title,
        "n_files": len(files),
        "tokens": total_tokens,
        "size_kb": size_kb,
    }


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="导出项目代码上下文给网页版 LLM")
    ap.add_argument("--mode", choices=["split", "single"], default="split",
                    help="split=按模块切分（推荐）；single=合并为一个文件")
    ap.add_argument("--out", default=os.path.join(ROOT, "llm_context"),
                    help="输出目录")
    ap.add_argument("--with-configs", action="store_true",
                    help="附带 models/ 与 runs/ 下的小型 json 配置（config.json 等）")
    ap.add_argument("--with-docs", action="store_true",
                    help="附带项目文档（README.md / INTERVIEW_MAP.md 等）")
    args = ap.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    stats = []

    # 代码包：排除 .md，保证体积可控；文档单独成包
    if args.mode == "single":
        bases = ["pretrain", "sft", "rl", "data_eng", "."]
        s = write_bundle(out_dir, "LLM_Project_full.md", "LLM-Project 全量代码上下文",
                         bases, exclude_ext={".md"})
        if s:
            stats.append(s)
    else:
        for d, fname, title in MODULES:
            s = write_bundle(out_dir, fname, title, [d], exclude_ext={".md"})
            if s:
                stats.append(s)
        s = write_bundle(out_dir, "50_root.md", "顶层文件 (requirements / 启动脚本)",
                         ["."], exclude_ext={".md"})
        if s:
            stats.append(s)

    # 核心文档包：README + INTERVIEW_MAP（GPT 最该先读的项目说明书）
    core_docs = [n for n in ["README.md", "INTERVIEW_MAP.md", "requirements.txt", "start.ps1"]
                 if os.path.isfile(os.path.join(ROOT, n))]
    if core_docs:
        parts = ["# 项目核心文档（建议第一个上传 / 先读）", ""]
        for name in core_docs:
            body, _ = render_file(name, os.path.join(ROOT, name),
                                  os.path.splitext(name)[1].lower())
            parts.append(body)
        path = os.path.join(out_dir, "60_docs_core.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        stats.append({"file": "60_docs_core.md", "title": "核心文档 (README/面试地图)",
                      "n_files": len(core_docs),
                      "tokens": estimate_tokens("\n".join(parts)),
                      "size_kb": os.path.getsize(path) / 1024})

    # 子模块文档：默认只打包 data_eng 的审计文档（可选上传）
    if args.with_docs:
        sub_docs = collect_files(["pretrain", "sft", "rl", "data_eng"], only_ext={".md"})
        if sub_docs:
            parts = ["# 各模块文档（审计 / 实验记录，按需上传）", ""]
            for rel, p, ext in sub_docs:
                body, _ = render_file(rel, p, ext)
                parts.append(body)
            path = os.path.join(out_dir, "65_module_docs.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(parts))
            stats.append({"file": "65_module_docs.md", "title": "模块文档 (审计/实验记录)",
                          "n_files": len(sub_docs),
                          "tokens": estimate_tokens("\n".join(parts)),
                          "size_kb": os.path.getsize(path) / 1024})

    if args.with_configs:
        parts = ["# 模型与实验配置（小型 json 摘录）", "",
                 "> 仅收录 < 50KB 的 json 配置，权重/分词器大文件已跳过。", ""]
        cnt = 0
        for base in ["models", "runs"]:
            abs_base = os.path.join(ROOT, base)
            if not os.path.isdir(abs_base):
                continue
            for dirpath, dirnames, filenames in os.walk(abs_base):
                for name in sorted(filenames):
                    if not name.endswith(".json"):
                        continue
                    p = os.path.join(dirpath, name)
                    try:
                        if os.path.getsize(p) > 50_000:
                            continue
                    except OSError:
                        continue
                    text = read_text(p)
                    if text is None:
                        continue
                    rel = os.path.relpath(p, ROOT).replace("\\", "/")
                    fence = fence_for(text)
                    parts += [f"### `{rel}`", "", f"{fence}json", text.rstrip(), fence, ""]
                    cnt += 1
        if cnt:
            path = os.path.join(out_dir, "70_configs.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(parts))
            stats.append({"file": "70_configs.md", "title": "模型/实验配置",
                          "n_files": cnt,
                          "tokens": estimate_tokens("\n".join(parts)),
                          "size_kb": os.path.getsize(path) / 1024})

    # 索引文件
    total_tokens = sum(s["tokens"] for s in stats)
    idx = [
        "# LLM-Project 代码上下文包（用于上传网页版 LLM）",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"项目根目录：`{ROOT}`",
        "",
        "## 包含哪些文件",
        "",
        "| 文件 | 内容 | 源文件数 | 估算 tokens | 体积 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for s in stats:
        idx.append(f"| `{s['file']}` | {s['title']} | {s['n_files']} | "
                   f"~{s['tokens']:,} | {s['size_kb']:.0f} KB |")
    idx += [
        f"| **合计** | | {sum(s['n_files'] for s in stats)} | **~{total_tokens:,}** | "
        f"{sum(s['size_kb'] for s in stats):.0f} KB |",
        "",
        "## 已排除的内容（重要）",
        "",
        "- `.venv/` —— Python 虚拟环境（第三方库，几万个文件）",
        "- `models/` —— 模型权重（5.6GB，.safetensors / .pt）",
        "- `runs/` —— 训练产物与 checkpoint（1.6GB）",
        "- `data/` —— 训练数据集（43MB）",
        "- `data_eng/output/`、`output_clean/` —— 数据工程中间产物",
        "- `.env` —— **密钥文件，绝不外传**",
        "- `*.docx` —— **简历，含个人信息，绝不外传**",
        "- `__pycache__/`、`.git/`、`.workbuddy/`",
        "",
        "## 上传建议",
        "",
        "1. 单文件 token 超过 ~60k 的，优先只传你当前要讨论的那个模块；",
        "2. 推荐用 ChatGPT **Projects** 功能建一个「LLM-Project」项目，把这些文件一次性加进项目知识库；",
        "3. 首次提问时附上 `README.md` 内容 + 你的具体问题，不要问「帮我看看整个项目」。",
        "",
        "## 目录树（自研代码部分）",
        "",
        "```text",
        build_tree(["pretrain", "sft", "rl", "data_eng"]),
        "```",
        "",
    ]
    with open(os.path.join(out_dir, "00_INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(idx))

    print(f"✅ 已输出到 {out_dir}")
    print(f"{'文件':<24}{'源文件数':>8}{'估算 tokens':>14}{'体积':>10}")
    for s in stats:
        print(f"{s['file']:<24}{s['n_files']:>8}{s['tokens']:>14,}{s['size_kb']:>9.0f}KB")
    print(f"{'合计':<24}{sum(s['n_files'] for s in stats):>8}{total_tokens:>14,}"
          f"{sum(s['size_kb'] for s in stats):>9.0f}KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
