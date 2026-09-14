# -*- coding: utf-8 -*-
# =====================================================================
# method_overview.py —— 论文风格 Figure 1 生成脚本（v2：图表混排版）
#
# 与 v1（纯方框+文字）的区别：
#   · 流程图标（文件/课本/骰子/对勾/柱状/齿轮）用矢量图元绘制，不依赖字体图标
#   · 每层能力右侧内嵌【迷你柱状图】——before/after 柱 + 参考线（9B / 规则基线）
#   · 底部两个科研小图：迁移斜率图（Base vs math-RL 起点）与 零方差修复条形图
#   · 统一图例、面板标号 (a)(b)(c)、论文式 caption
#
# 产出：figures/method_overview.png（300dpi）· figures/method_overview.pdf（矢量）
# 用法：python figures/method_overview.py
# =====================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Polygon,
                                Circle, Rectangle, Wedge)

OUT = Path(__file__).resolve().parent

# ---------------- 论文风配色（低饱和、印刷友好）----------------
INK      = "#22303C"      # 主文字/描边
SUBINK   = "#5A6B7B"      # 次级文字
BAND     = "#F2F6FA"      # 流程底带
BEFORE   = "#B9C4CE"      # “RL 前”柱
AFTER    = "#2E6DA4"      # “RL 后”柱（主色）
REF      = "#C0392B"      # 参考线（红）
ACCENT2  = "#3F7A46"      # 迁移/正增益
WARN     = "#B3541E"      # 诊断
PANEL_EC = "#D6DEE6"

FS_TITLE, FS_BODY, FS_SMALL = 11.5, 9.0, 7.6


# =====================================================================
# 图元：小型矢量图标（每个画在 (x, y) 处，尺寸 s）
# =====================================================================
def ico_document(ax, x, y, s, color=INK):
    ax.add_patch(Rectangle((x, y), s * 0.72, s, fc="white", ec=color, lw=1.1, zorder=4))
    for i, dy in enumerate((0.66, 0.44, 0.22)):
        ax.plot([x + s * 0.12, x + s * 0.60], [y + s * dy, y + s * dy],
                color=color, lw=0.9, zorder=5)


def ico_book(ax, x, y, s, color=INK):
    ax.add_patch(Rectangle((x, y), s * 0.78, s, fc="white", ec=color, lw=1.1, zorder=4))
    ax.plot([x + s * 0.39, x + s * 0.39], [y, y + s], color=color, lw=1.0, zorder=5)
    ax.plot([x + s * 0.10, x + s * 0.32], [y + s * 0.70, y + s * 0.70],
            color=color, lw=0.8, zorder=5)


def ico_dice(ax, x, y, s, color=INK):
    ax.add_patch(Rectangle((x, y), s, s, fc="white", ec=color, lw=1.1,
                           joinstyle="round", zorder=4))
    for dx, dy in ((0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72), (0.5, 0.5)):
        ax.add_patch(Circle((x + s * dx, y + s * dy), s * 0.085, fc=color, zorder=5))


def ico_check(ax, x, y, s, color=ACCENT2):
    ax.plot([x + s * 0.10, x + s * 0.42, x + s * 0.92],
            [y + s * 0.50, y + s * 0.14, y + s * 0.88],
            color=color, lw=2.0, solid_capstyle="round", zorder=5)


def ico_bars(ax, x, y, s, color=INK):
    for i, h in enumerate((0.45, 0.75, 1.0)):
        ax.add_patch(Rectangle((x + i * s * 0.34, y), s * 0.22, s * h,
                               fc=color, ec="none", alpha=0.85, zorder=5))


def ico_gear(ax, x, y, s, color=INK):
    c = (x + s / 2, y + s / 2)
    ax.add_patch(Circle(c, s * 0.44, fc="white", ec=color, lw=1.1, zorder=4))
    ax.add_patch(Circle(c, s * 0.15, fc=color, ec="none", zorder=5))
    for k in range(8):
        a = k * np.pi / 4
        ax.plot([c[0] + np.cos(a) * s * 0.44, c[0] + np.cos(a) * s * 0.58],
                [c[1] + np.sin(a) * s * 0.44, c[1] + np.sin(a) * s * 0.58],
                color=color, lw=1.4, zorder=5)


def ico_code(ax, x, y, s, color=INK):
    ax.plot([x + s * 0.30, x + s * 0.06, x + s * 0.30],
            [y + s * 0.85, y + s * 0.50, y + s * 0.15], color=color, lw=1.6, zorder=5)
    ax.plot([x + s * 0.62, x + s * 0.86, x + s * 0.62],
            [y + s * 0.85, y + s * 0.50, y + s * 0.15], color=color, lw=1.6, zorder=5)


def ico_skill(ax, x, y, s, color=INK):
    """技能/工具选择：三个候选方块，其中一个被勾选。"""
    for i in range(3):
        ax.add_patch(Rectangle((x + i * s * 0.36, y + s * 0.30), s * 0.26, s * 0.26,
                               fc="white", ec=color, lw=1.0, zorder=4))
    ico_check(ax, x, y + s * 0.22, s * 0.30, color=color)


def shadow_box(ax, x, y, w, h, fc="white", ec=PANEL_EC, lw=1.1, rad=0.014, z=2):
    ax.add_patch(FancyBboxPatch((x + 0.004, y - 0.006), w, h,
                                boxstyle=f"round,pad=0.006,rounding_size={rad}",
                                fc="#E9EEF3", ec="none", zorder=z))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.006,rounding_size={rad}",
                                fc=fc, ec=ec, lw=lw, zorder=z + 1))


def chevron(ax, x, y, w, h, fc, ec, notch=0.012):
    pts = [(x, y), (x + w - notch, y), (x + w, y + h / 2), (x + w - notch, y + h),
           (x, y + h), (x + notch, y + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.1, zorder=2))


# =====================================================================
# 内嵌迷你图：三个能力的 before/after 柱 + 参考线
# =====================================================================
def mini_bars(ax, base, ours, ref, ref_label, ylim, ylabel, fmt="{:.1f}"):
    ax.bar([0, 1], [base, ours], width=0.52, color=[BEFORE, AFTER],
           edgecolor=INK, linewidth=0.6, zorder=3)
    ax.axhline(ref, ls=(0, (3, 2)), lw=1.2, color=REF, zorder=4)
    ax.text(1.52, ref, ref_label, color=REF, fontsize=6.6, va="center", ha="left")
    for xi, v in ((0, base), (1, ours)):
        ax.text(xi, v + ylim * 0.035, fmt.format(v), ha="center",
                fontsize=7.2, color=INK, fontweight="bold")
    ax.set_xlim(-0.55, 2.05); ax.set_ylim(0, ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(PANEL_EC)
    ax.set_ylabel(ylabel, fontsize=6.8, color=SUBINK)
    ax.tick_params(length=0)


# =====================================================================
def main() -> None:
    fig = plt.figure(figsize=(14.0, 9.3))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------------- (a) 共享流水线 ----------------
    ax.text(0.010, 0.985, "(a) One GRPO + verifiable-reward pipeline, three verifiable tasks",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")
    ax.add_patch(Rectangle((0.008, 0.735), 0.984, 0.175, fc=BAND, ec="none", zorder=0))

    y, h = 0.775, 0.095
    stages = [
        (0.022, 0.150, "Data pools", "GSM8K · MBPP/HumanEval\nself-built skills + BFCL", ico_document),
        (0.186, 0.140, "Cold-start SFT", "reasoning traces\n(optional)", ico_book),
        (0.340, 0.158, "GRPO rollout", "G = 8 on-policy samples\nLoRA + frozen vision", ico_dice),
        (0.512, 0.176, "Verifiable reward", "numeric match · sandbox tests\nskill-set + refusal", ico_check),
        (0.702, 0.142, "Group-relative adv.", "no critic network", ico_bars),
        (0.858, 0.128, "LoRA update", "0.07% params", ico_gear),
    ]
    for x, w, title, sub, icon in stages:
        chevron(ax, x, y, w, h, fc="white", ec=INK)
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color=INK, zorder=4)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=6.8, color=SUBINK, zorder=4, linespacing=1.25)
        icon(ax, x + w / 2 - 0.010, y + h + 0.012, 0.020, color=INK)

    # 反馈回路（on-policy loop）
    ax.add_patch(FancyArrowPatch((0.916, y - 0.022), (0.412, y - 0.022),
                                 arrowstyle="-|>", mutation_scale=10, lw=1.3,
                                 color=AFTER, ls=(0, (4, 2.5)),
                                 connectionstyle="arc3,rad=0.12", zorder=3))
    ax.text(0.664, y - 0.010, "on-policy rollout loop  (resumable · checkpointed)",
            fontsize=7.2, color=AFTER, ha="center")


    # ---------------- (b) 三层能力（左：图标+任务/奖励；右：迷你柱状图）----------------
    ax.text(0.010, 0.712, "(b) Three capability layers — same recipe, different verifier",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")

    rows = [
        dict(y=0.545, name="Reasoning", icon=ico_book,
             task="GSM8K  ·  numeric answer match",
             extra="cold-start SFT → GRPO (2000 steps)",
             base=30.0, ours=50.0, ref=40.0, ref_label="9B 40.0", ylim=62,
             ylab="acc (%)"),
        dict(y=0.400, name="Execution", icon=ico_code,
             task="HumanEval / MBPP  ·  sandboxed unit tests",
             extra="T1 sandbox: timeout · no-net · rlimits",
             base=14.6, ours=25.0, ref=21.9, ref_label="SFT 21.9", ylim=32,
             ylab="pass@1 (%)"),
        dict(y=0.255, name="Selection", icon=ico_skill,
             task="BFCL v3  ·  skill-set match + refusal",
             extra="self-built pool → public-benchmark eval",
             base=0.485, ours=0.795, ref=0.590, ref_label="rule 0.590", ylim=0.92,
             ylab="exact"),
    ]
    for r in rows:
        shadow_box(ax, 0.022, r["y"], 0.108, 0.108, fc="white")
        r["icon"](ax, 0.048, r["y"] + 0.052, 0.030, color=INK)
        ax.text(0.076, r["y"] + 0.026, r["name"], fontsize=10.2, fontweight="bold",
                color=INK, ha="center", va="center")

        shadow_box(ax, 0.140, r["y"], 0.360, 0.108, fc="white")
        ax.text(0.152, r["y"] + 0.070, r["task"], fontsize=8.8, color=INK, va="center")
        ax.text(0.152, r["y"] + 0.032, r["extra"], fontsize=7.2, color=SUBINK, va="center")

        # 迷你柱状图（论文里常见的 inline chart）
        ax_in = fig.add_axes([0.545, r["y"] + 0.012, 0.135, 0.088])
        mini_bars(ax_in, r["base"], r["ours"], r["ref"], r["ref_label"],
                  r["ylim"], r["ylab"], fmt="{:.3f}" if r["ylim"] < 1 else "{:.1f}")

        # 结果结论
        arrow = "→"
        gain = r["ours"] - r["base"]
        gtxt = f"+{gain:.1f} pt" if r["ylim"] > 1 else f"+{gain:.3f}"
        ax.text(0.700, r["y"] + 0.062, f"{r['base']:g} {arrow} {r['ours']:g}", fontsize=9.4,
                fontweight="bold", color=INK, va="center")
        ax.text(0.700, r["y"] + 0.028, f"{gtxt} after RL", fontsize=7.6,
                color=ACCENT2, va="center")

    # 每行结论批注
    ax.text(0.845, 0.590, "surpasses the\n9B baseline", fontsize=7.4, color=SUBINK, va="center")
    ax.text(0.845, 0.445, "sandbox makes code\nrewards safe & stable", fontsize=7.4, color=SUBINK, va="center")
    ax.text(0.845, 0.300, "false-load rate\n1.000 → 0.225", fontsize=7.4, color=ACCENT2,
            va="center", fontweight="bold")

    # ---------------- (c) 迁移与诊断 ----------------
    ax.text(0.010, 0.232, "(c) Cross-layer transfer and the failure mode we fixed",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")

    # c1: 迁移斜率图
    ax1 = fig.add_axes([0.055, 0.062, 0.225, 0.130])
    x = [0, 1]
    ax1.plot(x, [0.207, 0.360], "-o", color=BEFORE, lw=1.8, ms=4.5, label="start = Base")
    ax1.plot(x, [0.160, 0.795], "-o", color=AFTER, lw=2.2, ms=4.5, label="start = math-RL")
    ax1.annotate("", xy=(1, 0.795), xytext=(1, 0.360),
                 arrowprops=dict(arrowstyle="<->", color=ACCENT2, lw=1.1))
    ax1.text(1.04, 0.575, "+0.483\ntransfer", fontsize=6.8, color=ACCENT2, va="center")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["before", "after skill-RL"], fontsize=6.6)
    ax1.set_ylim(0, 0.95); ax1.set_xlim(-0.15, 1.55)
    ax1.set_ylabel("BFCL exact", fontsize=6.8, color=SUBINK)
    ax1.tick_params(labelsize=6.4); ax1.legend(fontsize=6.2, frameon=False, loc="upper left")
    for s in ax1.spines.values():
        s.set_color(PANEL_EC)
    ax1.set_title("transfer effect", fontsize=7.6, color=INK, pad=3)

    # c2: 零方差修复
    ax2 = fig.add_axes([0.325, 0.062, 0.185, 0.130])
    ax2.bar([0, 1], [0.50, 0.00], width=0.5, color=[WARN, AFTER], edgecolor=INK, lw=0.6)
    ax2.text(0, 0.52, "0.50", ha="center", fontsize=7.6, fontweight="bold", color=WARN)
    ax2.text(1, 0.03, "0.00", ha="center", fontsize=7.6, fontweight="bold", color=AFTER)
    ax2.annotate("G=4 → G=8\n+ partial credit", xy=(0.5, 0.42), xytext=(0.5, 0.60),
                 ha="center", fontsize=6.6, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["v1", "v2"], fontsize=6.8)
    ax2.set_ylim(0, 0.78); ax2.set_yticks([])
    ax2.set_title("zero-variance GRPO steps", fontsize=7.6, color=INK, pad=3)
    for s in ax2.spines.values():
        s.set_color(PANEL_EC)

    # c3: 误加载率
    ax3 = fig.add_axes([0.565, 0.062, 0.185, 0.130])
    ax3.bar([0, 1], [1.000, 0.225], width=0.5, color=[WARN, AFTER], edgecolor=INK, lw=0.6)
    for xi, v in ((0, 1.000), (1, 0.225)):
        ax3.text(xi, v + 0.04, f"{v:.3f}", ha="center", fontsize=7.4,
                 fontweight="bold", color=INK)
    ax3.set_xticks([0, 1]); ax3.set_xticklabels(["retrieval", "ours"], fontsize=6.8)
    ax3.set_ylim(0, 1.25); ax3.set_yticks([])
    ax3.set_title("false-load rate (refusal cases)", fontsize=7.6, color=INK, pad=3)
    for s in ax3.spines.values():
        s.set_color(PANEL_EC)

    # 图例
    ax.add_patch(Rectangle((0.800, 0.162), 0.016, 0.016, fc=BEFORE, ec=INK, lw=0.5))
    ax.text(0.822, 0.170, "before RL", fontsize=6.8, color=SUBINK, va="center")
    ax.add_patch(Rectangle((0.800, 0.132), 0.016, 0.016, fc=AFTER, ec=INK, lw=0.5))
    ax.text(0.822, 0.140, "after RL (ours)", fontsize=6.8, color=SUBINK, va="center")
    ax.plot([0.800, 0.816], [0.109, 0.109], ls=(0, (3, 2)), color=REF, lw=1.3)
    ax.text(0.822, 0.109, "reference baseline", fontsize=6.8, color=SUBINK, va="center")

    # caption
    ax.text(0.010, 0.012,
            "Metrics — GSM8K: exact accuracy on a fixed 50-item test subset; HumanEval: pass@1 over 164 tasks; "
            "BFCL v3: stratified 400 items (80 per category), exact set match; false-load = loading a skill when none is required.\n"
            "Single GPU throughout: 12 GB consumer card or 32 GB V100 · SFT and RL question pools are disjoint (no leakage) · "
            "training pools are self-built; evaluation uses public benchmarks. All numbers are reproducible from the repository.",
            fontsize=6.8, color=SUBINK, va="bottom", linespacing=1.5)

    for ext, dpi in (("png", 300), ("pdf", None)):
        p = OUT / f"method_overview.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("saved:", p)


if __name__ == "__main__":
    main()
