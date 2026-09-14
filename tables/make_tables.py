# -*- coding: utf-8 -*-
# =====================================================================
# make_tables.py —— 从 evidence/results/*.json 生成「科研格式表格」
#
# 为什么用脚本生成而不是手写：
#   数字全部从原始评估 JSON 读取 → 不会抄错；以后重跑评测，表格自动更新
#
# 产出（tables/ 目录）：
#   main_results.{tex,md}       三层能力总表
#   gsm8k.{tex,md}              推理层（含 9B 参照）
#   humaneval.{tex,md}          执行层
#   bfcl_skill.{tex,md}         选择层（含 4 条基线 + 双臂）
#   bfcl_per_category.{tex,md}  选择层分类别
#   transfer.{tex,md}           跨层迁移分析
#   ALL_TABLES.md               以上 Markdown 汇总（可直接贴主页/README）
#
# 用法：python tables/make_tables.py
# =====================================================================

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence" / "results"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# 读取工具
# ---------------------------------------------------------------
def _tag_of(path: str, j: dict) -> str:
    if j.get("tag"):
        return j["tag"]
    m = re.match(r"^[^_]+_(.+)_\d{8}-\d{4}\.json$", os.path.basename(path))
    return m.group(1) if m else (j.get("strategy") or os.path.basename(path))


def load(prefix: str) -> dict:
    """读某前缀（如 gsm8k / humaneval / bfcl_skill）的所有汇总 JSON，tag -> 指标。"""
    out = {}
    for f in glob.glob(str(EVID / f"{prefix}_*.json")):
        if ".samples." in f:
            continue
        j = json.load(open(f, encoding="utf-8"))
        if not isinstance(j, dict) or "accuracy" not in j and "overall" not in j:
            continue
        t = _tag_of(f, j)
        # 同名 tag 保留最新（文件时间戳大者）
        if t not in out or f > out[t]["_file"]:
            out[t] = {**j, "_file": f}
    return out


def pct(x: float) -> str:
    return f"{x*100:.1f}\\%"


def sec(x: float) -> str:
    return f"{x:.3f}"


# ---------------------------------------------------------------
# LaTeX / Markdown 渲染
# ---------------------------------------------------------------
def latex_table(caption: str, label: str, header: list, rows: list,
                align: str = "l" + "c" * 0) -> str:
    align = align if len(align) == len(header) else "l" + "c" * (len(header) - 1)
    lines = [
        "\\begin{table}[t]", "  \\centering",
        f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{align}}}", "    \\toprule",
        "    " + " & ".join(header) + " \\\\", "    \\midrule",
    ]
    for r in rows:
        lines.append("    " + " & ".join(str(c) for c in r) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def md_table(header: list, rows: list) -> str:
    out = ["| " + " | ".join(_md_clean(str(h)) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_md_clean(str(c)) for c in r) + " |")
    return "\n".join(out) + "\n"


def _md_clean(s: str) -> str:
    """把 LaTeX 写法换成 Markdown 可读写法（LaTeX 版仍保留原样）。"""
    return (s.replace("\\%", "%").replace("$\\downarrow$", "↓")
             .replace("$\\Delta_{mathRL}-\\Delta_{base}$", "Δ(mathRL) − Δ(base)")
             .replace("$\\rightarrow$", "→")
             .replace("\\textbf{", "").replace("\\Delta", "Δ"))


def write(name: str, latex: str, md: str):
    (OUT / f"{name}.tex").write_text(latex, encoding="utf-8")
    (OUT / f"{name}.md").write_text(md, encoding="utf-8")


TEX_NOTE = ("% 由 tables/make_tables.py 自动生成（数字来自 evidence/results/*.json）\n"
            "% 需要 \\usepackage{booktabs}\n\n")


def main() -> None:
    g = load("gsm8k")
    h = load("humaneval")
    b = load("bfcl_skill")

    all_md = ["# 科研格式表格（自动生成）\n",
              "> 由 `tables/make_tables.py` 从 `evidence/results/*.json` 生成；"
              "重跑评测后重新执行即可更新。\n"]

    # ---------------- 表 1：三层能力总表 ----------------
    def acc_of(d, *tags, default=None):
        """取主指标：GSM8K/HumanEval 是顶层 accuracy；BFCL 嵌在 overall.exact_acc。"""
        for t in tags:
            if t in d:
                j = d[t]
                if isinstance(j, dict):
                    if "accuracy" in j:
                        return j["accuracy"]
                    if "overall" in j:
                        return j["overall"]["exact_acc"]
        return default

    rows = [
        ("Reasoning (GSM8K, 50)", pct(acc_of(g, "Base-full", "Base")), pct(acc_of(g, "grpo-v2-full", "grpo-v2")),
         pct(acc_of(g, "9B-full", "9B")), "verifier: numeric match"),
        ("Execution (HumanEval, 164)", pct(acc_of(h, "Base-full", "Base")),
         pct(acc_of(h, "grpo-v2-full", "grpo-v2")), "--", "verifier: sandboxed unit tests"),
        ("Selection (BFCL v3, 400)", sec(acc_of(b, "Base-full", "Base")),
         sec(acc_of(b, "skillrl-from-mathrl-full", "skillrl-from-mathrl")),
         sec(acc_of(b, "strat-rule")), "verifier: skill-set match + refusal"),
    ]
    header = ["Layer (task, n)", "Base", "Our RL", "Strong ref.", "Verifiable reward"]
    tex_rows = [[r[0], r[1], "\\textbf{" + r[2] + "}", r[3], r[4]] for r in rows]
    latex = TEX_NOTE + latex_table(
        "One pipeline, three capability layers. 'Our RL' is the best RL checkpoint per layer; "
        "'Strong ref.' is the 9B baseline (reasoning) or the best non-RL heuristic (selection).",
        "tab:main", header, tex_rows, align="lcccc")
    md = md_table(header, [list(r) for r in rows])
    write("main_results", latex, md)
    all_md += ["## Table 1. Three-layer summary\n", md]

    # ---------------- 表 2：GSM8K ----------------
    order = [("Base", "4B-Base (no RL)"), ("sft", "4B-SFT (cold start)"),
             ("grpo", "4B-GRPO v1 (G=4, binary reward)"),
             ("grpo-v2", "4B-GRPO v2 (G=8, partial credit)"),
             ("9B", "9B-Base (reference)")]
    rows, tex_rows = [], []
    for key, name in order:
        for sfx in ("-full", ""):
            if key + sfx in g:
                d = g[key + sfx]
                rows.append([name, pct(d["accuracy"]), f"{d['correct']}/{d['n']}"])
                tex_rows.append([name, pct(d["accuracy"]), f"{d['correct']}/{d['n']}"])
                break
    header = ["Model", "Accuracy", "Correct"]
    latex = TEX_NOTE + latex_table(
        "Reasoning layer: GSM8K. GRPO v2 fixes the zero-variance-group failure mode of v1 "
        "(group size 4$\\rightarrow$8 + partial-credit reward) and surpasses the 9B baseline.",
        "tab:gsm8k", header, tex_rows, align="lcc")
    md = md_table(header, rows)
    write("gsm8k", latex, md)
    all_md += ["## Table 2. Reasoning layer (GSM8K)\n", md]

    # ---------------- 表 3：HumanEval ----------------
    order = [("Base", "4B-Base"), ("sft", "4B-SFT (math cold start)"),
             ("grpo", "4B-Code-GRPO v1"), ("grpo-v2", "4B-Code-GRPO v2")]
    rows, tex_rows = [], []
    for key, name in order:
        for sfx in ("-full", ""):
            if key + sfx in h:
                d = h[key + sfx]
                rows.append([name, pct(d["accuracy"]), f"{d['correct']}/{d['n']}"])
                tex_rows.append([name, pct(d["accuracy"]), f"{d['correct']}/{d['n']}"])
                break
    header = ["Model", "pass@1", "Passed"]
    latex = TEX_NOTE + latex_table(
        "Execution layer: HumanEval pass@1 with sandboxed unit-test reward. "
        "Math cold-start SFT transfers to code (+7.3 pt over Base).",
        "tab:humaneval", header, tex_rows, align="lcc")
    md = md_table(header, rows)
    write("humaneval", latex, md)
    all_md += ["## Table 3. Execution layer (HumanEval)\n", md]

    # ---------------- 表 4：BFCL 技能选择 ----------------
    order = [("strat-load_all", "Load all candidates"), ("strat-random", "Random pick"),
             ("strat-similarity", "Similarity retrieval (common practice)"),
             ("strat-rule", "Retrieval + threshold rule"),
             ("Base", "4B-Base (no RL)"), ("start-grpo-v2", "Math-RL only (no skill RL)"),
             ("skillrl-from-base", "Skill-RL from Base"),
             ("skillrl-from-mathrl", "Skill-RL from math-RL checkpoint")]
    rows, tex_rows = [], []
    for key, name in order:
        tag = key
        for sfx in ("-full", ""):
            if tag + sfx in b:
                d = b[tag + sfx]["overall"]
                r = [name, sec(d["exact_acc"]), sec(d["micro_f1"]),
                     sec(d["false_load_rate"]), sec(d["miss_rate"]), f"{d['avg_pred_size']:.2f}"]
                rows.append(r)
                tex_rows.append(r[:1] + ["\\textbf{" + r[1] + "}"] +
                                (["\\textbf{" + r[2] + "}"] if key.startswith("skillrl") else r[2:]))
                break
    header = ["Strategy / model", "Exact", "F1", "False-load$\\downarrow$", "Miss$\\downarrow$", "Avg. loads"]
    latex = TEX_NOTE + latex_table(
        "Selection layer: BFCL v3 (stratified 400; 80 per category). Similarity retrieval never refuses "
        "(false-load = 1.000); RL cuts it to 0.225 while improving exact match.",
        "tab:bfcl", header, tex_rows, align="lccccc")
    md = md_table(header, rows)
    write("bfcl_skill", latex, md)
    all_md += ["## Table 4. Selection layer (BFCL v3, stratified 400)\n", md]

    # ---------------- 表 5：BFCL 分类别 ----------------
    cats = ["multiple", "live_multiple", "simple", "irrelevance", "live_irrelevance"]
    cols = [("strat-similarity", "Retrieval"), ("strat-rule", "Rule"),
            ("Base", "Base"), ("skillrl-from-mathrl", "Skill-RL (ours)")]
    if all(tag in b for tag, _ in cols):
        rows, tex_rows = [], []
        for c in cats:
            r = [c] + [sec(b[tag]["per_category"].get(c, {}).get("exact_acc", float("nan"))) for tag, _ in cols]
            rows.append(r)
            tex_rows.append([r[0]] + ["\\textbf{" + v + "}" if i == len(r) - 1 else v
                                      for i, v in enumerate(r[1:])])
        header = ["Category", "Retrieval", "Rule", "Base", "Skill-RL (ours)"]
        latex = TEX_NOTE + latex_table(
            "Per-category exact match on BFCL. Retrieval scores 0.000 on both refusal categories "
            "(irrelevance / live_irrelevance); RL reaches 0.650 / 0.900.",
            "tab:bfcl-cat", header, tex_rows, align="lcccc")
        md = md_table(header, rows)
        write("bfcl_per_category", latex, md)
        all_md += ["## Table 5. Selection layer, per category\n", md]

    # ---------------- 表 6：迁移分析 ----------------
    def A(d, *tags):
        for t in tags:
            if t in d:
                return d[t]["overall"]["exact_acc"] if "overall" in d[t] else d[t]["accuracy"]
        return None
    base = A(b, "Base-full", "Base")
    mathrl = A(b, "start-grpo-v2-full", "start-grpo-v2")
    armA = A(b, "skillrl-from-base-full", "skillrl-from-base")
    armB = A(b, "skillrl-from-mathrl-full", "skillrl-from-mathrl")
    if None not in (base, mathrl, armA, armB):
        rows = [
            ["Start = Base", sec(base), sec(armA), f"{armA-base:+.3f}"],
            ["Start = math-RL checkpoint", sec(mathrl), sec(armB), f"{armB-mathrl:+.3f}"],
            ["Transfer effect $\\Delta_{mathRL}-\\Delta_{base}$", "--", "--",
             f"\\textbf{{{((armB-mathrl)-(armA-base)):+.3f}}}"],
        ]
        header = ["Setting", "Before skill RL", "After skill RL", "Gain"]
        latex = TEX_NOTE + latex_table(
            "Cross-task transfer: starting skill-selection RL from a math-RL checkpoint yields a much "
            "larger gain than starting from Base (learnability transfer), although math RL alone does "
            "not improve selection (0.160 vs 0.207).",
            "tab:transfer", header, rows, align="lccc")
        md = md_table(header, [[r[0], r[1], r[2], r[3].replace("\\textbf{", "").replace("}", "")] for r in rows])
        write("transfer", latex, md)
        all_md += ["## Table 6. Cross-layer transfer\n", md]

    (OUT / "ALL_TABLES.md").write_text("\n".join(all_md), encoding="utf-8")
    print("generated in", OUT)
    for f in sorted(OUT.glob("*")):
        if f.suffix in (".tex", ".md"):
            print("  ", f.name)


if __name__ == "__main__":
    main()
