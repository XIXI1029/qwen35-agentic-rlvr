# -*- coding: utf-8 -*-
# =====================================================================
# evaluate_skill.py —— Skill 选择评测（BFCL held-out）+ 基线对比
#
# 指标
#   exact_acc       技能集合完全匹配率（主指标）
#   micro P/R/F1    技能级微观精确率/召回率/F1（对应 BFCL 的选取口径）
#   false_load_rate "该拒绝却加载"率（BFCL irrelevance 类，正是"加载不准"）
#   miss_rate       "该加载却拒绝"率
#   avg_pred_size   平均加载技能数（越少越省上下文）
#
# 策略
#   --model <path|hf>  用模型生成（本项目的 RL 策略 / Base / SFT 对比）
#   --model load_all   基线 B2：把所有候选都加载
#   --model similarity 基线 B3：按 请求↔技能描述 的词重叠选 top-1（业界现状做法）
#   --model rule       基线 B4：B3 + 阈值，低于阈值则拒绝
#   --model random     基线 B1':随机选一个
#
# 用法：
#   python memoryRL/evaluate_skill.py --model similarity --tag B3-similarity
#   python memoryRL/evaluate_skill.py --model outputs/qwen3.5-4b-grpo-skill --tag skill-rl
# =====================================================================

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import OUTPUTS_DIR, ROOT, logger        # noqa: E402
from skill_verifier import parse_gt, parse_selection    # noqa: E402

DEFAULT_POOL = ROOT / "data" / "processed" / "skill_bfcl_eval.jsonl"


# ---------------------------------------------------------------
# 工具：从 prompt 里还原"请求"与"候选技能"
# ---------------------------------------------------------------
_CAND_RE = re.compile(r"^- ([^\s:]+): (.*?)(?: \(参数:|$)", re.MULTILINE)


def parse_prompt(prompt: str):
    """返回 (request, candidates[list[(name, desc)]])。"""
    req = ""
    m = re.search(r"用户请求[：:]\s*(.*)", prompt)
    if m:
        req = m.group(1).strip()
    else:                     # BFCL 的对话会被拍平成 "user: ..." 行
        req = "\n".join(l for l in prompt.splitlines() if l.startswith("user:")) or prompt
    cands = [(m.group(1), m.group(2)) for m in _CAND_RE.finditer(prompt)]
    return req, cands


def _tokens(s: str):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _load_model_any(path_or_id: str):
    """加载模型：支持 ①普通权重目录/模型ID（复用主项目策略）
    ②LoRA 适配器目录（读 base_model_hint.txt 找到基座后拼装）。
    这样训练用 --save-mode adapter 省内存，评估也能直接吃适配器目录。
    """
    p = Path(path_or_id)
    if p.is_dir() and (p / "adapter_config.json").exists():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        hint = p / "base_model_hint.txt"
        base = hint.read_text(encoding="utf-8").strip() if hint.exists() else None
        if not base:
            raise SystemExit(f"{p} 是适配器目录但缺少 base_model_hint.txt，"
                             f"请手动指定基座或在训练时用默认 adapter 模式重新保存")
        logger.info(f"检测到 LoRA 适配器，基座={base}")
        m = AutoModelForImageTextToText.from_pretrained(
            base, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa", trust_remote_code=False, local_files_only=True)
        m = PeftModel.from_pretrained(m, str(p))
        t = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
        m.eval()
        return m, t
    import evaluate as EVAL                     # 普通路径/模型ID：复用主项目加载策略
    return EVAL._load_with_strategy(path_or_id, "auto")


# ---------------------------------------------------------------
# 离线基线策略
# ---------------------------------------------------------------
def predict_load_all(req, cands, rng): return [n for n, _ in cands]


def predict_random(req, cands, rng):
    return [rng.choice(cands)[0]] if cands else []


def _sim_scores(req, cands):
    rq = _tokens(req)
    out = []
    for name, desc in cands:
        dt = _tokens(name.replace("_", " ").replace(".", " ")) | _tokens(desc)
        out.append((len(rq & dt) / (len(dt) + 1), name))
    return sorted(out, reverse=True)


def predict_similarity(req, cands, rng):
    if not cands:
        return []
    return [_sim_scores(req, cands)[0][1]]


def predict_rule(req, cands, rng, thr: float = 0.12):
    if not cands:
        return []
    best_s, best_n = _sim_scores(req, cands)[0]
    return [best_n] if best_s >= thr else []


# ---------------------------------------------------------------
# 指标
# ---------------------------------------------------------------
def metrics(records) -> dict:
    n = len(records)
    exact = sum(1 for r in records if r["exact"])
    # 注意：records 里 P/G 存的是 list（便于写 JSON），这里统一转 set 再算
    tp = sum(len(set(r["P"]) & set(r["G"])) for r in records)
    pp = sum(len(set(r["P"])) for r in records)
    gg = sum(len(set(r["G"])) for r in records)
    prec = tp / pp if pp else 1.0
    rec = tp / gg if gg else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    none_rows = [r for r in records if not r["G"]]
    sel_rows = [r for r in records if r["G"]]
    false_load = sum(1 for r in none_rows if r["P"]) / len(none_rows) if none_rows else 0.0
    miss = sum(1 for r in sel_rows if not r["P"]) / len(sel_rows) if sel_rows else 0.0
    return {
        "n": n, "exact_acc": round(exact / n, 4),
        "micro_precision": round(prec, 4), "micro_recall": round(rec, 4),
        "micro_f1": round(f1, 4),
        "false_load_rate": round(false_load, 4), "miss_rate": round(miss, 4),
        "avg_pred_size": round(sum(len(r["P"]) for r in records) / n, 2),
        "avg_gt_size": round(gg / n, 2),
        "avg_prompt_chars": round(sum(r["chars"] for r in records) / n, 1),
    }


def run(strategy: str, rows: list, max_new: int = 128, limit: int | None = None) -> dict:
    """按策略跑评测，返回 (指标, 逐题记录)。"""
    rng = random.Random(0)
    records = []
    model = tok = None
    if strategy not in ("load_all", "random", "similarity", "rule"):
        model, tok = _load_model_any(strategy)

    rows = rows[: limit] if limit else rows
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        prompt = r["prompt"]
        req, cands = parse_prompt(prompt)
        G = parse_gt(r["ground_truth"])
        if strategy == "load_all":
            P = predict_load_all(req, cands, rng)
        elif strategy == "random":
            P = predict_random(req, cands, rng)
        elif strategy == "similarity":
            P = predict_similarity(req, cands, rng)
        elif strategy == "rule":
            P = predict_rule(req, cands, rng)
        else:
            import evaluate as EVAL
            out = EVAL.generate_answer(model, tok, prompt, max_new_tokens=max_new)
            P = parse_selection(out, [n for n, _ in cands]) or []
        records.append({
            "id": r["meta"].get("id"), "category": r["meta"].get("category"),
            "P": sorted(set(P)), "G": sorted(set(G)),
            "exact": {x.lower() for x in P} == {x.lower() for x in G},
            "chars": len(prompt),
        })
        if i % 200 == 0:
            logger.info(f"  ...{i}/{len(rows)}")
    return metrics(records), records, round(time.time() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="模型路径/ID，或基线名：load_all/random/similarity/rule")
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--categories", default=None,
                    help="只评测这些类别，逗号分隔（默认全部）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    pool = Path(args.pool)
    rows = [json.loads(l) for l in open(pool, encoding="utf-8") if l.strip()]
    if args.categories:
        keep = set(args.categories.split(","))
        rows = [r for r in rows if r["meta"]["category"] in keep]

    # 分层抽样：限制样本数时，必须**按类别比例抽样**，否则会只抽到某一类
    # （BFCL 题库是按类别顺序排列的，直接 rows[:N] 会漏掉 irrelevance 等类）
    if args.limit and args.limit < len(rows):
        by_cat = {}
        for r in rows:
            by_cat.setdefault(r["meta"]["category"], []).append(r)
        picked, i = [], 0
        while len(picked) < args.limit:
            added = False
            for cat, items in by_cat.items():
                if i < len(items):
                    picked.append(items[i]); added = True
                    if len(picked) >= args.limit:
                        break
            if not added:
                break
            i += 1
        rows = picked
        logger.info(f"分层抽样后：{ {c: sum(1 for r in rows if r['meta']['category']==c) for c in by_cat} }")

    logger.info(f"评测集: {pool.name}  {len(rows)} 题  策略={args.model}")

    m, records, secs = run(args.model, rows, args.max_new, args.limit)

    # 分类别看
    per_cat = {}
    for cat in sorted({r["category"] for r in records}):
        sub = [r for r in records if r["category"] == cat]
        per_cat[cat] = metrics(sub)

    tag = args.tag or args.model.replace("/", "-").split("-")[-1]
    out_dir = OUTPUTS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bfcl_skill_{tag}_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({"tag": tag, "strategy": args.model, "overall": m,
                               "per_category": per_cat, "seconds": secs},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("=== 总指标 ===")
    for k, v in m.items():
        logger.info(f"  {k:18s}: {v}")
    logger.info(f"=== 分类别 exact_acc ===")
    for c, mm in per_cat.items():
        logger.info(f"  {c:18s}: exact={mm['exact_acc']:.3f} "
                    f"false_load={mm['false_load_rate']:.3f} miss={mm['miss_rate']:.3f} n={mm['n']}")
    logger.info(f"结果: {out}")


if __name__ == "__main__":
    main()
