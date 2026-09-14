# -*- coding: utf-8 -*-
# =====================================================================
# skill_verifier.py —— Skill 选择的奖励函数（RLVR 的"裁判"）
#
# 奖励设计
#   模型输出要加载的技能集合 S_pred，标准答案 S_gt（可以为空=不需要技能）：
#     - 完全一致            -> 1.0
#     - 部分一致            -> Jaccard * partial（给部分分，避免组内全同分→零优势）
#     - 该拒绝却加载/该加载却拒绝 -> 0.0（"加载不准"的两个方向都要罚）
#     - 结果可解析(JSON)     -> +format 小奖励
#     - 多加载了技能          -> 每个多余技能 -over_load（鼓励"少而准"）
# =====================================================================

from __future__ import annotations

import json
import re
from typing import List, Optional

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_LIST_RE = re.compile(r"(?:SKILLS|skills)\s*[:：]\s*\[?(.*?)\]?\s*$", re.MULTILINE | re.DOTALL)


def parse_selection(text: str, candidate_names: List[str]) -> Optional[List[str]]:
    """从模型输出里解析出"要加载的技能名集合"。

    解析顺序：① JSON {"skills": [...]} ② "SKILLS: a, b" ③ 在文本里出现过的候选名
    返回 None 表示无法解析。空列表代表"不需要任何技能"。
    """
    if not text:
        return None
    # ① JSON
    for m in _JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and "skills" in obj:
            v = obj["skills"] or []
            if isinstance(v, str):
                v = [x.strip() for x in re.split(r"[,，;；]", v) if x.strip()]
            return [str(x).strip().strip("'\"`") for x in v]
    # ② SKILLS: a, b / 技能：a、b
    m = _LIST_RE.search(text) or re.search(r"(?:技能|skill)\s*[:：]\s*(.+)", text, re.I)
    if m:
        body = m.group(1)
        if re.search(r"(none|无|空|no skill|不需要)", body, re.I) or not body.strip():
            return []
        return [x.strip().strip("'\"`") for x in re.split(r"[,，、;；\s]+", body) if x.strip()]
    # ③ 文本中命中的候选名（保守：只在明确提到时算）
    hits = [n for n in candidate_names
            if re.search(r"(?<![\w.])" + re.escape(n) + r"(?![\w.])", text)]
    if hits:
        return hits
    if re.search(r"(none|不需要任何技能|无可用技能)", text, re.I):
        return []
    return None


def parse_gt(gt) -> List[str]:
    """标准答案（JSON 字符串或 dict）-> 技能名列表。"""
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except Exception:
            return []
    if isinstance(gt, dict):
        return [str(x) for x in (gt.get("skills") or [])]
    return []


def score_one(pred: Optional[List[str]], gt: List[str], weights: dict) -> float:
    """单样本打分（技能名大小写/空格不敏感）。"""
    w_exact = float(weights.get("exact", 1.0))
    w_partial = float(weights.get("partial", 0.5))
    w_fmt = float(weights.get("format", 0.05))
    w_over = float(weights.get("over_load", 0.1))

    if pred is None:                     # 解析失败：只有格式分也没有
        return 0.0
    P = {x.lower().strip() for x in pred}
    G = {x.lower().strip() for x in gt}
    bonus = w_fmt                        # 能解析出结构 -> 给一点格式分

    if P == G:
        return w_exact + bonus
    # "加载不准"的两个方向都判 0（不给部分分，让信号明确）
    if not G and P:
        return 0.0                       # 该拒绝却加载
    if G and not P:
        return 0.0                       # 该加载却拒绝
    # 部分重叠：Jaccard 部分分，并对多余技能小惩罚
    inter = len(P & G)
    union = len(P | G) or 1
    jac = inter / union
    extra = len(P - G)
    return max(0.0, w_partial * jac + bonus - w_over * extra)


def make_skill_reward(weights: dict | None = None, debug_every: int = 50):
    """构造 TRL 可用的奖励函数（闭包持有权重与统计）。"""
    state = {"calls": 0, "exact": 0, "total": 0}

    def skill_reward(prompts, completions, ground_truth=None, **kwargs):
        rewards = []
        for pr, comp, gt in zip(prompts, completions,
                                ground_truth or [None] * len(completions)):
            # 候选技能名从 prompt 里抠（render_prompt 生成的格式："- name: desc"）
            cands = re.findall(r"^- ([^\s:]+):", pr or "", re.MULTILINE)
            g = parse_gt(gt)
            pred = parse_selection(comp, cands)       # 传候选名，解析兜底更准
            r = score_one(pred, g, weights or {})
            rewards.append(r)
            state["total"] += 1
            state["exact"] += int(pred is not None and
                                  {x.lower() for x in pred} == {x.lower() for x in g})
        state["calls"] += 1
        if state["calls"] % debug_every == 0:
            print(f"[skill_reward] 调用 {state['calls']} 次，"
                  f"累计精确命中率 {state['exact']/max(1,state['total']):.1%}", flush=True)
        return rewards

    return skill_reward


if __name__ == "__main__":
    W = {"exact": 1.0, "partial": 0.5, "format": 0.05, "over_load": 0.1}
    cases = [
        ('{"skills": ["a"]}', ["a"], "完全一致"),
        ('{"skills": ["a","b"]}', ["a"], "多加载了一个"),
        ('{"skills": []}', [], "正确拒绝"),
        ('{"skills": ["x"]}', [], "该拒绝却加载"),
        ('{"skills": []}', ["a"], "该加载却拒绝"),
        ('SKILLS: a, b', ["a", "c"], "文本格式+部分重叠"),
        ('胡说八道', ["a"], "无法解析"),
    ]
    for comp, gt, note in cases:
        pred = parse_selection(comp, [])
        print(f"{note:14s} pred={pred} gt={gt} -> {score_one(pred, gt, W):.3f}")
