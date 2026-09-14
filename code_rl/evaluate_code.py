# -*- coding: utf-8 -*-
# =====================================================================
# evaluate_code.py —— HumanEval pass@1 评估（代码实验的评测）
#
# 结果写入 outputs/results/humaneval_<tag>_<ts>.json，schema 与数学实验一致，
# 因此可以直接被 scripts/compare_models.py 汇总。
#
# 【用法】
#   python code_rl/evaluate_code.py --model Qwen/Qwen3.5-4B-Base --tag Base
#   python code_rl/evaluate_code.py --model outputs/qwen3.5-4b-grpo-code --tag grpo
# =====================================================================

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import OUTPUTS_DIR, logger                 # noqa: E402
import evaluate as EVAL                                     # 复用其加载策略/生成  # noqa: E402
from code_sandbox import extract_python, run_tests          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "code_rl" / "data" / "humaneval.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-samples", type=int, default=164)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--strategy", choices=["auto", "gpu", "4bit", "cpu"], default="auto")
    ap.add_argument("--timeout", type=float, default=3.0)
    args = ap.parse_args()

    if not POOL.exists():
        raise SystemExit(f"缺少 HumanEval 题库: {POOL}\n先跑: python code_rl/build_code_pool.py --which eval")

    rows = [json.loads(l) for l in open(POOL, encoding="utf-8") if l.strip()]
    rows = rows[: args.max_samples]
    logger.info(f"HumanEval 评估集: {len(rows)} 题")

    model, tok = EVAL._load_with_strategy(args.model, args.strategy)

    correct, records = 0, []
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        out = EVAL.generate_answer(model, tok, r["prompt"], max_new_tokens=args.max_new)
        code = extract_python(out)
        meta = json.loads(r["ground_truth"])
        ok, info = run_tests(code, meta["tests"], timeout=args.timeout)
        correct += int(ok)
        records.append({"task_id": r["task_id"], "passed": ok, "info": info,
                        "code": code[:1500]})
        if n % 10 == 0 or n == len(rows):
            logger.info(f"[humaneval] {n}/{len(rows)}  pass@1 = {correct/n*100:.1f}%")

    res = {"model": args.model, "task": "humaneval", "n": len(rows),
           "correct": correct, "accuracy": round(correct / max(1, len(rows)), 4),
           "seconds": round(time.time() - t0, 1),
           "generated_at": datetime.now().isoformat(timespec="seconds")}
    results_dir = OUTPUTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.model.replace("/", "-").split("-")[-1]
    out_path = results_dir / f"humaneval_{tag}_{datetime.now():%Y%m%d-%H%M}.json"
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path.with_suffix(".samples.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"HumanEval pass@1 = {res['accuracy']*100:.1f}% "
                f"({correct}/{len(rows)})  结果: {out_path}")


if __name__ == "__main__":
    main()
