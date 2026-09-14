# -*- coding: utf-8 -*-
# =====================================================================
# method_overview.py —— 论文风格「方法总览图」生成脚本
#
# 产出：figures/method_overview.png（300dpi，用于主页/README）
#       figures/method_overview.pdf（矢量，插入论文/报告不糊）
#
# 设计（三栏式，模仿会议论文的 Figure 1）：
#   (a) 共享流水线：数据 → 冷启动 SFT → GRPO rollout → 可验证奖励 → 组内相对优势 → LoRA 更新
#   (b) 三层能力与结果：推理 / 执行 / 选择
#   (c) 跨层迁移与关键诊断：数学SFT→代码 +7.3pt；数学RL→技能选择 迁移 +48.3pt；零方差步 0.50→0.00
#
# 说明：全英文标注（论文风格，也避免中文字体缺失导致的方块）
# 用法：python figures/method_overview.py
# =====================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

# ---- 配色（低饱和、印刷友好）----
C_BOX   = "#F5F7FA"; C_EDGE = "#3B4A5A"
C_ACC   = "#DCE9F7"; C_ACC_E = "#2D6DB5"      # 主流程
C_TASK  = "#EAF3E8"; C_TASK_E = "#3F7A3A"     # 三个任务
C_RES   = "#FDF1DC"; C_RES_E = "#B57A16"      # 结果
C_WARN  = "#FBE9E7"; C_WARN_E = "#B3372C"     # 诊断/修复


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_EDGE, fs=9, weight="normal", fold=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                linewidth=1.1, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, zorder=3, linespacing=1.35)
    if fold:      # 右上角小标（如 (a)/(b)）
        ax.text(x + 0.004, y + h - 0.006, fold, ha="left", va="top",
                fontsize=10, fontweight="bold", color=ec, zorder=4)


def arrow(ax, p1, p2, color=C_EDGE, style="-|>", lw=1.3, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def main() -> None:
    fig = plt.figure(figsize=(13.6, 7.8))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ============================================================
    # (a) 共享流水线
    # ============================================================
    ax.text(0.012, 0.975, "(a) One shared pipeline over three verifiable tasks",
            fontsize=11.5, fontweight="bold", va="top")

    y, h = 0.800, 0.110
    stages = [
        (0.015, 0.145, "Data pools\nGSM8K · MBPP/HumanEval\nself-built skills + BFCL", C_TASK, C_TASK_E),
        (0.172, 0.145, "Cold-start SFT\n(reasoning traces)\noptional", C_BOX, C_EDGE),
        (0.329, 0.160, "GRPO rollout\nG = 8 samples / prompt\nLoRA + frozen vision", C_ACC, C_ACC_E),
        (0.501, 0.170, "Verifiable reward\n(1) numeric match\n(2) sandbox unit tests\n(3) skill-set + refusal", C_ACC, C_ACC_E),
        (0.683, 0.150, "Group-relative\nadvantage\n(no critic)", C_ACC, C_ACC_E),
        (0.845, 0.140, "Policy update\nLoRA only\n(0.07% params)", C_ACC, C_ACC_E),
    ]
    centers = []
    for x, w, t, fc, ec in stages:
        box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=8.6)
        centers.append((x + w / 2, y + h / 2))
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + stages[i][1]
        arrow(ax, (x1 + 0.002, y + h / 2), (stages[i + 1][0] - 0.002, y + h / 2))
    # 反馈回路
    arrow(ax, (0.915, y - 0.004), (0.409, y - 0.004), color=C_ACC_E, rad=0.16, ls=(0, (4, 3)))
    ax.text(0.66, 0.718, "on-policy loop (resumable, checkpointed)",
            fontsize=8.4, color=C_ACC_E, ha="center")

    ax.text(0.015, 0.685, "Single GPU: 12 GB consumer (4-bit/LoRA) or 32 GB V100  ·  "
                          "SFT and RL question pools are disjoint (no leakage)  ·  "
                          "training pools self-built, evaluation on public benchmarks",
            fontsize=8.6, color="#444444")

    # ============================================================
    # (b) 三层能力与结果
    # ============================================================
    ax.text(0.012, 0.650, "(b) Three capability layers (same recipe, different verifier)",
            fontsize=11.5, fontweight="bold", va="top")

    rows = [
        (0.530, "Reasoning", "GSM8K (answer matching)",
         "34.0% → 50.0%   (base 30.0%)", "surpasses the 9B baseline (40.0%)"),
        (0.420, "Execution", "HumanEval / MBPP (sandboxed tests)",
         "21.9% → 25.0%   (base 14.6%)", "custom sandbox: timeout · no-net · rlimits"),
        (0.310, "Selection", "BFCL v3 (skill set + refusal)",
         "0.795 exact   (retrieval 0.485 / rule 0.590)", "false-load rate 100% → 22.5%"),
    ]
    for yrow, name, task, res, note in rows:
        box(ax, 0.015, yrow, 0.105, 0.090, name, fc=C_TASK, ec=C_TASK_E, fs=10.5, weight="bold")
        box(ax, 0.130, yrow, 0.300, 0.090, task, fc=C_BOX, ec=C_EDGE, fs=9)
        box(ax, 0.440, yrow, 0.330, 0.090, res, fc=C_RES, ec=C_RES_E, fs=9.4, weight="bold")
        box(ax, 0.780, yrow, 0.205, 0.090, note, fc=C_BOX, ec=C_EDGE, fs=8.4)
        arrow(ax, (0.121, yrow + 0.045), (0.129, yrow + 0.045), color=C_TASK_E)
        arrow(ax, (0.431, yrow + 0.045), (0.439, yrow + 0.045), color=C_RES_E)

    # ============================================================
    # (c) 迁移与诊断
    # ============================================================
    ax.text(0.012, 0.270, "(c) Cross-layer transfer and the key diagnosis",
            fontsize=11.5, fontweight="bold", va="top")

    box(ax, 0.015, 0.120, 0.300, 0.115,
        "Transfer 1: math cold-start SFT → code\nHumanEval 14.6% → 21.9%  (+7.3 pt)\n"
        "structured reasoning transfers across tasks",
        fc=C_RES, ec=C_RES_E, fs=8.3)
    box(ax, 0.330, 0.120, 0.330, 0.115,
        "Transfer 2: math-RL checkpoint → skill RL\nΔ_base = +15.2 pt   Δ_mathRL = +63.5 pt\n"
        "transfer effect = +48.3 pt (positive)",
        fc=C_RES, ec=C_RES_E, fs=8.3)
    box(ax, 0.675, 0.120, 0.310, 0.115,
        "Diagnosis: zero-variance groups block RL\n"
        "frac(zero-variance steps) 0.50 → 0.00\nafter group size 4→8 + partial-credit reward",
        fc=C_WARN, ec=C_WARN_E, fs=8.3)

    ax.text(0.012, 0.075,
            "Metrics: GSM8K = exact accuracy on 50 test items (full-suite run pending) · HumanEval = pass@1 over 164 tasks · "
            "BFCL v3 = stratified 400 items (80 per category); false-load = loading a skill when none is required.",
            fontsize=7.8, color="#555555")
    ax.text(0.012, 0.040,
            "All results are reproducible from this repository (see FINAL_REPORT.md §6); "
            "curated raw evidence (result JSONs, training logs) is in evidence/.",
            fontsize=7.8, color="#555555")

    for ext in ("png", "pdf"):
        p = OUT / f"method_overview.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print("saved:", p)


if __name__ == "__main__":
    main()
