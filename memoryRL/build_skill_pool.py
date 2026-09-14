# -*- coding: utf-8 -*-
# =====================================================================
# build_skill_pool.py —— 生成 Skill-RLVR 训练题库（自建，near-miss 干扰）
#
# 产出两份：
#   data/processed/skill_train.jsonl    训练池（只用非 held-out 技能）
#   data/processed/skill_heldout.jsonl  泛化评测池（held-out 技能，训练中从未出现）
#   每行：{"prompt", "ground_truth"(JSON), "meta"}
#
# 用法：
#   python memoryRL/build_skill_pool.py --per-skill 40 --none-ratio 0.12
# =====================================================================

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import ROOT, logger                       # noqa: E402
import skills_lib as SL                                   # noqa: E402

OUT_DIR = ROOT / "data" / "processed"


def _make_row(skill, request, rng, near: int, far: int) -> dict:
    cands = SL.sample_candidates(skill, rng, n_near=near, n_far=far)
    return {
        "prompt": SL.render_prompt(request, cands),
        "ground_truth": json.dumps({"skills": [skill["name"]]}, ensure_ascii=False),
        "meta": {"skill": skill["name"], "domain": skill["domain"],
                 "n_candidates": len(cands), "kind": "select"},
    }


def build(skills, per_skill: int, none_ratio: float, seed: int,
          near: int = 3, far: int = 1) -> list:
    rng = random.Random(seed)
    rows = []
    for skill in skills:
        # 每个技能生成 per_skill 条，循环使用它的 3 条措辞模板（保证措辞多样）
        for i in range(per_skill):
            req = skill["requests"][i % len(skill["requests"])]
            rows.append(_make_row(skill, req, rng, near, far))
    # 注入"无需技能"样本（对应 BFCL irrelevance）
    n_none = int(len(rows) * none_ratio / max(1e-6, 1 - none_ratio))
    for _ in range(n_none):
        req = rng.choice(SL.NO_SKILL_REQUESTS)
        cands = SL.sample_no_skill_candidates(rng, k=rng.randint(4, 6))
        rows.append({
            "prompt": SL.render_prompt(req, cands),
            "ground_truth": json.dumps({"skills": []}, ensure_ascii=False),
            "meta": {"skill": None, "domain": "none",
                     "n_candidates": len(cands), "kind": "none"},
        })
    rng.shuffle(rows)
    return rows


def _write(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"已写出 {path}  ({len(rows)} 条)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-skill", type=int, default=40)
    ap.add_argument("--none-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--near", type=int, default=3, help="同域 near-miss 干扰数")
    ap.add_argument("--far", type=int, default=1, help="跨域干扰数")
    args = ap.parse_args()

    # 训练池：只用非 held-out 技能（held-out 技能在训练中完全不出现）
    train_rows = build(SL.TRAINABLE, args.per_skill, args.none_ratio, args.seed,
                       args.near, args.far)
    _write(train_rows, OUT_DIR / "skill_train.jsonl")

    # 泛化池：held-out 技能（干扰项来自同域的可训练技能）——测"没见过的技能"能不能选对
    held_skills = [s for s in SL.SKILLS if s["name"] in SL.HELDOUT]
    held_rows = build(held_skills, max(10, args.per_skill // 4), 0.0,
                      args.seed + 1, args.near, args.far)
    _write(held_rows, OUT_DIR / "skill_heldout.jsonl")

    logger.info(f"技能库统计：共 {len(SL.SKILLS)} 个技能 / "
                f"可训练 {len(SL.TRAINABLE)} / held-out {len(SL.HELDOUT)} {sorted(SL.HELDOUT)}")


if __name__ == "__main__":
    main()
