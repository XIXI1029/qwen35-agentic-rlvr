# -*- coding: utf-8 -*-
# =====================================================================
# build_code_pool.py —— 构建代码 RL 题库（Code RLVR 实验用）
#
# 和数学实验的对应关系
#   数学：GSM8K 题面+数字答案  -> verifier 比数字
#   代码：MBPP 题面+隐藏测试   -> 沙箱跑测试
#   两者都产出统一格式 {"prompt", "ground_truth"}，共用 run_grpo 训练回路。
#
# 数据集选择
#   训练题库：MBPP(sanitized, 427 题) —— 题目短、函数级、自带 assert 测试
#   评估题库：HumanEval(164 题)       —— 业界标准代码基准，pass@1
#   （不把测试写进 prompt，避免模型"背答案"；只给函数名和题意）
#
# 用法
#   python code_rl/build_code_pool.py --which train   # MBPP -> code_rl/data/mbpp_train.jsonl
#   python code_rl/build_code_pool.py --which eval    # HumanEval -> code_rl/data/humaneval.jsonl
# =====================================================================

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 引入父目录 scripts/ 里的公共模块（_bootstrap 负责镜像/缓存/日志）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import DATA_RAW, ROOT, logger  # noqa: E402

OUT_DIR = ROOT / "code_rl" / "data"


def _mbpp_train(max_n: int | None, mbpp_config: str = "full",
                splits=("train", "validation")):
    """MBPP：题面 + 参考解 + assert 测试，转成统一格式（隐藏测试）。

    说明：上游数据集现在 'sanitized' 只剩 120 条，'full' 才有 train(374)/
    validation(90)。这里默认用 full 的 train+validation 拼训练池（约 464 条），
    **不碰 test 集**（避免与可能的 MBPP 评测混淆）。
    """
    from datasets import load_dataset
    logger.info(f"加载 MBPP ({mbpp_config}, splits={list(splits)}) ...")
    rows, seen = [], set()
    for sp in splits:
        ds = load_dataset("google-research-datasets/mbpp", mbpp_config,
                          split=sp, cache_dir=str(DATA_RAW))
        for i, it in enumerate(ds):
            if max_n and len(rows) >= max_n:
                break
            if it.get("task_id") in seen:        # 跨 split 去重
                continue
            seen.add(it.get("task_id"))

            tests = "\n".join(it["test_list"])   # 隐藏测试（不进 prompt）
            problem = it.get("text") or it.get("prompt") or ""  # 字段名兼容
            m = re.search(r"def\s+(\w+)\s*\(", it["code"])      # 参考解里的函数名
            if not m:
                continue
            fname = m.group(1)
            prompt = (
                "You are an expert Python programmer. Write a Python function "
                f"named `{fname}` to solve the following task.\n\n"
                f"{problem}\n\n"
                "Return ONLY the complete function code inside a single "
                "```python code block. Do not include tests or explanations."
            )
            gt = json.dumps({"tests": tests, "entry_point": fname},
                            ensure_ascii=False)
            rows.append({"prompt": prompt, "ground_truth": gt,
                         "task_id": str(it.get("task_id", i)), "source": "mbpp"})
    return rows


def _humaneval(max_n: int | None):
    """HumanEval：prompt=函数签名+docstring, test=含 check() 的测试。"""
    from datasets import load_dataset
    logger.info("加载 HumanEval (openai/openai_humaneval) ...")
    ds = load_dataset("openai/openai_humaneval", split="test",
                      cache_dir=str(DATA_RAW))
    rows = []
    for i, it in enumerate(ds):
        if max_n and len(rows) >= max_n:
            break
        prompt = (
            "Complete the following Python function. Return ONLY the full "
            "function code inside a single ```python code block.\n\n"
            f"{it['prompt']}"
        )
        # HumanEval 的 test 字段是 "def check(candidate): ...  check(entry_point)"
        gt = json.dumps({"tests": it["test"], "entry_point": it["entry_point"]},
                        ensure_ascii=False)
        rows.append({"prompt": prompt, "ground_truth": gt,
                     "task_id": it["task_id"], "source": "humaneval"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["train", "eval"], required=True)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.which == "train":
        rows = _mbpp_train(args.max_n)
        out = Path(args.out) if args.out else OUT_DIR / "mbpp_train.jsonl"
    else:
        rows = _humaneval(args.max_n)
        out = Path(args.out) if args.out else OUT_DIR / "humaneval.jsonl"

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"代码题库已写出: {out}  ({len(rows)} 条)")


if __name__ == "__main__":
    main()
