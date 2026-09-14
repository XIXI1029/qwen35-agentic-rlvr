# -*- coding: utf-8 -*-
# =====================================================================
# build_bfcl_pool.py —— 下载 BFCL v3 并转成"技能选择"评测池（held-out 基准）
#
# BFCL 数据格式（实测）：
#   题目文件 BFCL_v3_<cat>.json  : JSONL，每行 {id, question(对话), function(候选函数列表)}
#   答案文件 possible_answer/BFCL_v3_<cat>.json : JSONL，每行 {id, ground_truth}
#     ground_truth 形如 [{"triangle_properties.get": {...}}, ...] → 抽函数名即可
#   irrelevance 类没有答案文件：正确行为=不调用任何函数 → ground_truth = []
#
# 产出：data/processed/skill_bfcl_eval.jsonl
#   每行 {"prompt", "ground_truth"(JSON), "meta":{category, id, n_candidates}}
#
# 用法： python memoryRL/build_bfcl_pool.py
# =====================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")      # 镜像下禁用 Xet（否则 401）
from _bootstrap import DATA_RAW, ROOT, logger          # noqa: E402
import skills_lib as SL                                # noqa: E402

REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
# 类别 -> 是否"应该拒绝"（无答案文件）
CATEGORIES = {
    "BFCL_v3_multiple.json": False,          # 多候选里选对（核心）
    "BFCL_v3_live_multiple.json": False,     # live 版（更难/更真实）
    "BFCL_v3_simple.json": False,            # 单候选（简单对照）
    "BFCL_v3_irrelevance.json": True,        # 不该调用 → 必须拒绝
    "BFCL_v3_live_irrelevance.json": True,
}
LOCAL = DATA_RAW / "bfcl"


def _download(fn: str) -> Path | None:
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset",
                            local_dir=str(LOCAL))
        return Path(p)
    except Exception as e:
        logger.warning(f"下载失败 {fn}: {str(e)[:100]}")
        return None


def _read_jsonl(p: Path) -> list:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _flatten_question(q) -> str:
    """BFCL question 是 list[list[msg]]（可有多轮），拍平成一段文本。"""
    turns = q[0] if isinstance(q, list) and q and isinstance(q[0], list) else q
    parts = []
    for m in turns:
        if isinstance(m, dict):
            parts.append(f"{m.get('role','user')}: {m.get('content','')}")
    return "\n".join(parts) if parts else str(q)


def _names_from_gt(gt) -> list:
    """从 BFCL 的 ground_truth 里抽函数名（去重、保序）。"""
    names = []
    if isinstance(gt, list):
        for item in gt:
            if isinstance(item, dict):
                for k in item.keys():
                    if k not in names:
                        names.append(k)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/skill_bfcl_eval.jsonl")
    args = ap.parse_args()

    rows = []
    for fn, is_irrelevance in CATEGORIES.items():
        qp = _download(fn)
        if qp is None:
            continue
        questions = _read_jsonl(qp)
        answers = {}
        if not is_irrelevance:                  # irrelevance 无答案文件
            apth = _download("possible_answer/" + fn)
            if apth is not None:
                answers = {r["id"]: r.get("ground_truth") for r in _read_jsonl(apth)}

        kept = 0
        for q in questions:
            qid = q.get("id")
            funcs = q.get("function") or []
            if not funcs:
                continue
            # 候选函数 -> 本项目的"技能"结构（name/desc/args）
            cands = [{"name": f.get("name", ""),
                      "desc": (f.get("description") or "").strip()[:300],
                      "args": (f.get("parameters", {}) or {}).get("properties", {})}
                     for f in funcs]
            gt_names = [] if is_irrelevance else _names_from_gt(answers.get(qid))
            rows.append({
                "prompt": SL.render_prompt(_flatten_question(q.get("question")), cands),
                "ground_truth": json.dumps({"skills": gt_names}, ensure_ascii=False),
                "meta": {"category": fn.replace("BFCL_v3_", "").replace(".json", ""),
                         "id": qid, "n_candidates": len(cands),
                         "kind": "none" if not gt_names else "select"},
            })
            kept += 1
        logger.info(f"{fn}: 取 {kept} 题（答案 {len(answers)} 条）")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_cat = {}
    for r in rows:
        by_cat[r["meta"]["category"]] = by_cat.get(r["meta"]["category"], 0) + 1
    logger.info(f"BFCL 评测池已写出: {out}  共 {len(rows)} 题  {by_cat}")


if __name__ == "__main__":
    main()
