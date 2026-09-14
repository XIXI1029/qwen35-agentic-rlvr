# -*- coding: utf-8 -*-
# =====================================================================
# make_figures.py —— 生成三张独立科研图（字号更大、可分别插入论文/主页）
#
#   figures/fig_a_pipeline.png|pdf   共享流水线（数据→SFT→GRPO→奖励→优势→更新）
#   figures/fig_b_layers.png|pdf     三层能力 + 内嵌迷你柱状图（含参考线）
#   figures/fig_c_transfer.png|pdf   跨层迁移 + 失败模式修复
#
# 设计原则：每张图单独成图 → 字号可以放大（正文 11–13pt），打印/网页都清晰
# 图元全部用 matplotlib 图元绘制（图标不依赖字体），配色低饱和、印刷友好
#
# 用法：python figures/make_figures.py
# =====================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Polygon,
                                Circle, Rectangle)

OUT = Path(__file__).resolve().parent

# ---------------- 配色 ----------------
INK, SUBINK = "#22303C", "#5A6B7B"
BAND = "#F2F6FA"
BEFORE, AFTER = "#B9C4CE", "#2E6DA4"      # RL 前 / RL 后
REF, ACCENT2, WARN = "#C0392B", "#3F7A46", "#B3541E"
PANEL_EC = "#D6DEE6"

# ---------------- 字号（放大版）----------------
FS_TITLE, FS_HEAD, FS_BODY, FS_SMALL = 14, 12, 11, 9.5


# =====================================================================
# 矢量图标
# =====================================================================
def ico_document(ax, x, y, s, color=INK):
    ax.add_patch(Rectangle((x, y), s * 0.72, s, fc="white", ec=color, lw=1.3, zorder=4))
    for dy in (0.66, 0.44, 0.22):
        ax.plot([x + s * 0.12, x + s * 0.60], [y + s * dy, y + s * dy], color=color, lw=1.1, zorder=5)


def ico_book(ax, x, y, s, color=INK):
    ax.add_patch(Rectangle((x, y), s * 0.78, s, fc="white", ec=color, lw=1.3, zorder=4))
    ax.plot([x + s * 0.39, x + s * 0.39], [y, y + s], color=color, lw=1.1, zorder=5)
    ax.plot([x + s * 0.10, x + s * 0.32], [y + s * 0.70, y + s * 0.70], color=color, lw=1.0, zorder=5)


def ico_dice(ax, x, y, s, color=INK):
    ax.add_patch(Rectangle((x, y), s, s, fc="white", ec=color, lw=1.3, zorder=4))
    for dx, dy in ((0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72), (0.5, 0.5)):
        ax.add_patch(Circle((x + s * dx, y + s * dy), s * 0.09, fc=color, zorder=5))


def ico_check(ax, x, y, s, color=ACCENT2):
    ax.plot([x + s * 0.10, x + s * 0.42, x + s * 0.92], [y + s * 0.50, y + s * 0.14, y + s * 0.88],
            color=color, lw=2.4, solid_capstyle="round", zorder=5)


def ico_bars(ax, x, y, s, color=INK):
    for i, h in enumerate((0.45, 0.75, 1.0)):
        ax.add_patch(Rectangle((x + i * s * 0.34, y), s * 0.22, s * h, fc=color, ec="none", zorder=5))


def ico_gear(ax, x, y, s, color=INK):
    c = (x + s / 2, y + s / 2)
    ax.add_patch(Circle(c, s * 0.44, fc="white", ec=color, lw=1.3, zorder=4))
    ax.add_patch(Circle(c, s * 0.15, fc=color, ec="none", zorder=5))
    for k in range(8):
        a = k * np.pi / 4
        ax.plot([c[0] + np.cos(a) * s * 0.44, c[0] + np.cos(a) * s * 0.58],
                [c[1] + np.sin(a) * s * 0.44, c[1] + np.sin(a) * s * 0.58], color=color, lw=1.6, zorder=5)


def ico_code(ax, x, y, s, color=INK):
    ax.plot([x + s * 0.30, x + s * 0.06, x + s * 0.30], [y + s * 0.85, y + s * 0.50, y + s * 0.15],
            color=color, lw=2.0, zorder=5)
    ax.plot([x + s * 0.62, x + s * 0.86, x + s * 0.62], [y + s * 0.85, y + s * 0.50, y + s * 0.15],
            color=color, lw=2.0, zorder=5)


def ico_skill(ax, x, y, s, color=INK):
    for i in range(3):
        ax.add_patch(Rectangle((x + i * s * 0.36, y + s * 0.34), s * 0.26, s * 0.26,
                               fc="white", ec=color, lw=1.1, zorder=4))
    ico_check(ax, x, y + s * 0.20, s * 0.32, color=color)


def shadow_box(ax, x, y, w, h, fc="white", ec=PANEL_EC, lw=1.2, rad=0.016, z=2):
    ax.add_patch(FancyBboxPatch((x + 0.005, y - 0.008), w, h,
                                boxstyle=f"round,pad=0.006,rounding_size={rad}",
                                fc="#E9EEF3", ec="none", zorder=z))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.006,rounding_size={rad}",
                                fc=fc, ec=ec, lw=lw, zorder=z + 1))


def chevron(ax, x, y, w, h, fc="white", ec=INK, notch=0.014):
    pts = [(x, y), (x + w - notch, y), (x + w, y + h / 2), (x + w - notch, y + h),
           (x, y + h), (x + notch, y + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.4, zorder=2))


def save(fig, stem: str):
    for ext, dpi in (("png", 300), ("pdf", None)):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("saved:", p)
    plt.close(fig)


def mini_bars(ax, base, ours, ref, ref_label, ylim, ylabel, fmt="{:.1f}"):
    """内嵌迷你柱状图：before/after 双柱 + 参考线（论文里常见的 inline chart）。"""
    ax.bar([0, 1], [base, ours], width=0.55, color=[BEFORE, AFTER],
           edgecolor=INK, linewidth=0.8, zorder=3)
    ax.axhline(ref, ls=(0, (3, 2)), lw=1.6, color=REF, zorder=4)
    ax.text(0.5, ref + ylim * 0.025, ref_label, color=REF, fontsize=FS_SMALL,
            va="bottom", ha="center", fontweight="bold")
    for xi, v in ((0, base), (1, ours)):
        ax.text(xi, v + ylim * 0.04, fmt.format(v), ha="center",
                fontsize=FS_BODY - 1.5, color=INK, fontweight="bold")
    ax.set_xlim(-0.6, 2.2); ax.set_ylim(0, ylim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(ylabel, fontsize=FS_SMALL, color=SUBINK)
    for s in ax.spines.values():
        s.set_color(PANEL_EC)
    ax.tick_params(length=0)


# =====================================================================
# 图 (a)：共享流水线
# =====================================================================
def fig_a() -> None:
    fig = plt.figure(figsize=(13.5, 3.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.012, 0.965, "(a) One GRPO + verifiable-reward pipeline, three verifiable tasks",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")

    ax.add_patch(Rectangle((0.006, 0.28), 0.988, 0.52, fc=BAND, ec="none", zorder=0))
    y, h = 0.40, 0.30
    stages = [
        (0.020, 0.150, "Data pools", "GSM8K · MBPP/HumanEval\nself-built skills + BFCL", ico_document),
        (0.182, 0.140, "Cold-start SFT", "reasoning traces\n(optional)", ico_book),
        (0.334, 0.158, "GRPO rollout", "G = 8 on-policy samples\nLoRA + frozen vision", ico_dice),
        (0.504, 0.178, "Verifiable reward", "numeric match · sandbox tests\nskill-set + refusal", ico_check),
        (0.694, 0.142, "Group-relative adv.", "no critic network", ico_bars),
        (0.848, 0.140, "LoRA update", "0.07% of params", ico_gear),
    ]
    for x, w, title, sub, icon in stages:
        chevron(ax, x, y, w, h)
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=FS_BODY + 0.5, fontweight="bold", color=INK, zorder=4)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=FS_SMALL, color=SUBINK, zorder=4, linespacing=1.3)
        icon(ax, x + w / 2 - 0.011, y + h + 0.035, 0.026)

    ax.add_patch(FancyArrowPatch((0.918, 0.245), (0.415, 0.245), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.8, color=AFTER,
                                 ls=(0, (5, 3)), connectionstyle="arc3,rad=0.0", zorder=3))
    ax.text(0.666, 0.155, "on-policy rollout loop   (resumable · checkpointed)",
            fontsize=FS_SMALL + 0.5, color=AFTER, ha="center")

    ax.text(0.012, 0.055, "Single GPU (12 GB consumer / 32 GB V100)   ·   SFT and RL question pools are disjoint "
                          "(no leakage)   ·   training pools self-built; evaluation on public benchmarks",
            fontsize=FS_SMALL, color=SUBINK, va="bottom")
    save(fig, "fig_a_pipeline")


# =====================================================================
# 图 (b)：三层能力 + 迷你柱状图
# =====================================================================
def fig_b() -> None:
    fig = plt.figure(figsize=(12.0, 6.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.012, 0.975, "(b) Three capability layers — one recipe, three verifiers",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")

    rows = [
        dict(y=0.665, name="Reasoning", icon=ico_book,
             task="GSM8K  ·  numeric answer match", sub="cold-start SFT → GRPO (2000 steps)",
             base=30.0, ours=50.0, ref=40.0, ref_label="9B 40.0", ylim=64, ylab="accuracy (%)",
             note="surpasses the 9B baseline"),
        dict(y=0.375, name="Execution", icon=ico_code,
             task="HumanEval / MBPP  ·  sandboxed unit tests",
             sub="T1 sandbox: hard timeout · no-net · rlimits",
             base=14.6, ours=25.0, ref=21.9, ref_label="SFT 21.9", ylim=34, ylab="pass@1 (%)",
             note="sandbox makes code rewards\nsafe and stable"),
        dict(y=0.085, name="Selection", icon=ico_skill,
             task="BFCL v3  ·  skill-set match + refusal",
             sub="self-built training pool → public-benchmark evaluation",
             base=0.485, ours=0.795, ref=0.590, ref_label="rule 0.590", ylim=0.95, ylab="exact match",
             note="false-load rate\n1.000 → 0.225"),
    ]
    for r in rows:
        shadow_box(ax, 0.020, r["y"], 0.128, 0.235)
        r["icon"](ax, 0.055, r["y"] + 0.130, 0.055)
        ax.text(0.084, r["y"] + 0.045, r["name"], fontsize=FS_HEAD, fontweight="bold",
                color=INK, ha="center", va="center")

        shadow_box(ax, 0.160, r["y"], 0.400, 0.235)
        ax.text(0.176, r["y"] + 0.170, r["task"], fontsize=FS_BODY, color=INK, va="center")
        ax.text(0.176, r["y"] + 0.115, r["sub"], fontsize=FS_SMALL, color=SUBINK, va="center")
        gain = r["ours"] - r["base"]
        gtxt = f"+{gain:.1f} pt" if r["ylim"] > 1.2 else f"+{gain:.3f}"
        ax.text(0.176, r["y"] + 0.055, f"{r['base']:g}  →  {r['ours']:g}", fontsize=FS_BODY + 1.5,
                fontweight="bold", color=INK, va="center")
        ax.text(0.400, r["y"] + 0.055, f"({gtxt})", fontsize=FS_BODY, color=ACCENT2, va="center")

        ax_in = fig.add_axes([0.585, r["y"] + 0.045, 0.145, 0.175])
        mini_bars(ax_in, r["base"], r["ours"], r["ref"], r["ref_label"], r["ylim"], r["ylab"],
                  fmt="{:.3f}" if r["ylim"] < 1.2 else "{:.1f}")
        ax.text(0.762, r["y"] + 0.115, r["note"], fontsize=FS_SMALL, color=SUBINK, va="center",
                linespacing=1.4)

    ax.add_patch(Rectangle((0.762, 0.010), 0.020, 0.022, fc=BEFORE, ec=INK, lw=0.7))
    ax.text(0.790, 0.021, "before RL", fontsize=FS_SMALL, color=SUBINK, va="center")
    ax.add_patch(Rectangle((0.880, 0.010), 0.020, 0.022, fc=AFTER, ec=INK, lw=0.7))
    ax.text(0.908, 0.021, "after RL", fontsize=FS_SMALL, color=SUBINK, va="center")
    ax.plot([0.205, 0.240], [0.021, 0.021], ls=(0, (3, 2)), color=REF, lw=1.8)
    ax.text(0.250, 0.021, "reference baseline", fontsize=FS_SMALL, color=SUBINK, va="center")
    save(fig, "fig_b_layers")


# =====================================================================
# 图 (c)：迁移 + 失败模式修复
# =====================================================================
def fig_c() -> None:
    fig = plt.figure(figsize=(12.0, 3.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.012, 0.965, "(c) Cross-task transfer and the failure mode we fixed",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")

    # c1 迁移斜率图
    ax1 = fig.add_axes([0.065, 0.24, 0.225, 0.55])
    ax1.plot([0, 1], [0.207, 0.360], "-o", color=BEFORE, lw=2.4, ms=6.5, label="start = Base")
    ax1.plot([0, 1], [0.160, 0.795], "-o", color=AFTER, lw=3.0, ms=6.5, label="start = math-RL")
    ax1.annotate("", xy=(1, 0.795), xytext=(1, 0.360),
                 arrowprops=dict(arrowstyle="<->", color=ACCENT2, lw=1.6))
    ax1.text(1.05, 0.575, "+0.483\ntransfer", fontsize=FS_BODY - 1, color=ACCENT2, va="center")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["before", "after\nskill-RL"], fontsize=FS_SMALL)
    ax1.set_ylim(0, 1.0); ax1.set_xlim(-0.15, 1.62)
    ax1.set_ylabel("BFCL exact", fontsize=FS_SMALL, color=SUBINK)
    ax1.tick_params(labelsize=FS_SMALL)
    ax1.legend(fontsize=FS_SMALL - 0.5, frameon=False, loc="upper left")
    for s in ax1.spines.values():
        s.set_color(PANEL_EC)
    ax1.set_title("cross-task transfer", fontsize=FS_HEAD - 1, color=INK, pad=6)

    # c2 零方差修复
    ax2 = fig.add_axes([0.395, 0.24, 0.185, 0.55])
    ax2.bar([0, 1], [0.50, 0.00], width=0.55, color=[WARN, AFTER], edgecolor=INK, lw=0.8)
    ax2.text(0, 0.53, "0.50", ha="center", fontsize=FS_BODY, fontweight="bold", color=WARN)
    ax2.text(1, 0.04, "0.00", ha="center", fontsize=FS_BODY, fontweight="bold", color=AFTER)
    ax2.annotate("G = 4 → G = 8\n+ partial credit", xy=(0.5, 0.40), xytext=(0.5, 0.62),
                 ha="center", fontsize=FS_SMALL, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["v1", "v2"], fontsize=FS_SMALL)
    ax2.set_ylim(0, 0.82); ax2.set_yticks([])
    ax2.set_title("zero-variance GRPO steps", fontsize=FS_HEAD - 1, color=INK, pad=6)
    for s in ax2.spines.values():
        s.set_color(PANEL_EC)

    # c3 误加载率
    ax3 = fig.add_axes([0.700, 0.24, 0.185, 0.55])
    ax3.bar([0, 1], [1.000, 0.225], width=0.55, color=[WARN, AFTER], edgecolor=INK, lw=0.8)
    for xi, v in ((0, 1.000), (1, 0.225)):
        ax3.text(xi, v + 0.045, f"{v:.3f}", ha="center", fontsize=FS_BODY - 0.5,
                 fontweight="bold", color=INK)
    ax3.set_xticks([0, 1]); ax3.set_xticklabels(["retrieval", "ours"], fontsize=FS_SMALL)
    ax3.set_ylim(0, 1.3); ax3.set_yticks([])
    ax3.set_title("false-load rate (refusal cases)", fontsize=FS_HEAD - 1, color=INK, pad=6)
    for s in ax3.spines.values():
        s.set_color(PANEL_EC)

    ax.text(0.012, 0.045,
            "Metrics — GSM8K: exact accuracy on a fixed 50-item test subset; HumanEval: pass@1 over 164 tasks; "
            "BFCL v3: stratified 400 items (80 per category), exact set match; false-load = loading a skill when none is required.",
            fontsize=FS_SMALL - 1.5, color=SUBINK, va="bottom")
    save(fig, "fig_c_transfer")


if __name__ == "__main__":
    fig_a(); fig_b(); fig_c()
