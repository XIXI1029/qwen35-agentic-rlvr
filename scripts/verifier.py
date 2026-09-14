# -*- coding: utf-8 -*-
# =====================================================================
# verifier.py —— 「可验证奖励」的统一答案解析与判定（RLVR 的心脏）
#
# 被三处共用，保证同一套"对/错"标准：
#   - build_sft_coldstart.py  筛选"模型自生成且答对"的轨迹
#   - run_grpo.py 的 reward_fn  给 GRPO 打分
#   - evaluate.py / compare     算最终正确率
#
# 设计要点：Open-AgentRL 数学题的标准答案（ground_truth）有几种形态，
#   整数 "16"、小数、分数 "3/4"、\frac{3}{4}、\boxed{...} 等。
# 将其统一解析成一个 float 再比大小，任何"数字/分数"答案都能对。
# 像集合、不等式等非数字答案解析失败 -> 返回 None -> 判"不可验证"，自动跳过。
# =====================================================================

from __future__ import annotations

import re
from typing import Optional

# latex 分子分母：\frac{3}{4} 或 \dfrac{3}{4}
_FRAC_RE = re.compile(r"\\[d]?frac\{(-?\d+)\}\{(-?\d+)\}")
# 普通分数：3/4（含负号），也兼容整数后带斜杠
_SLASH_RE = re.compile(r"(-?\d+)\s*/\s*(-?\d+)")


def answer_value(text: str) -> Optional[float]:
    """把一段文本解析成答案数值；解析不了返回 None。

    解析顺序：
      1) 剥掉 LaTeX 花括号 \boxed{...}（先取其中内容）
      2) 找 \frac{a}{b} 分子分母 -> 返回 a/b
      3) 找普通分数 a/b
      4) 取最后一个数字（含小数、负数、千分位逗号）
    说明：取"最后一个数字"而不是第一个，因为模型常先推理（含很多数字）再给答案。
    """
    if not text:
        return None
    t = text.strip()
    # (1) 若答案被 \boxed{...} 包着，先掏出内容
    m = re.search(r"\\boxed\s*\{([^}]*)\}", t)
    if m:
        t = m.group(1)
    # 去掉所有空白与千分位逗号，方便统一解析
    t = t.replace(",", "").replace(" ", "").replace("，", "")
    # (2) LaTeX 分数
    m = _FRAC_RE.search(t)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return num / den if den else None
    # (3) 普通分数
    m = _SLASH_RE.search(t)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return num / den if den else None
    # (4) 最后出现的数字
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", t)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def verify(text: str, ground_truth: str) -> bool:
    """模型输出 text 是否答对 ground_truth。"""
    pred = answer_value(text)
    if pred is None:
        return False
    gt = answer_value(ground_truth)
    if gt is None:
        return False        # 标准答案本身不可数字解析 -> 判不了，算错
    # 数值相等（浮点直接比；同数不同写法如 16 与 16.0 相等）
    return abs(pred - gt) < 1e-6
