# -*- coding: utf-8 -*-
# =====================================================================
# build_sft_from_gsm8k.py —— 从 GSM8K 官方 CoT 构建冷启动 SFT 数据
#
# 【为什么】（详见 PROJECT_LOG Phase3 pivot）
#   自生成冷启动在 AIME 级难题上成功率太低(~12%)；GSM8K train 的 answer
#   字段自带人写逐步推理 + "#### 数字"，是现成的高质量 CoT SFT 语料。
#
# 【产物】data/processed/sft_coldstart.jsonl
#   每行: {"question","ground_truth","solution","messages":[...]}
#   messages 与 run_sft.py / GRPO / 评估的 chat 格式一致。
#
# 【用法】 python scripts/build_sft_from_gsm8k.py --n 1200 --seed 0
# =====================================================================

from __future__ import annotations

import argparse
import json
import random

from _bootstrap import DATA_RAW, DATA_PROCESSED, logger


def split_answer(answer: str):
    """把 gsm8k 的 answer "推理... #### 42" 拆成 (推理, 数字)。"""
    if "####" in answer:
        reasoning, val = answer.split("####", 1)
        return reasoning.strip(), val.strip()
    return answer, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1200, help="取多少条")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/processed/sft_coldstart.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    logger.info("加载 GSM8K train ...")
    ds = load_dataset("openai/gsm8k", "main", split="train",
                      cache_dir=str(DATA_RAW))
    rng = random.Random(args.seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:args.n]

    rows = []
    skipped = 0
    for i in idxs:
        q = ds[i]["question"]
        reasoning, val = split_answer(ds[i]["answer"])
        if not val:
            skipped += 1
            continue
        # 让最终答案以纯数字收尾（奖励函数只取最后出现的数字）
        assistant = f"{reasoning}\n\nFinal Answer: {val}"
        rows.append({
            "question": q,
            "ground_truth": val,
            "solution": assistant,
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": assistant},
            ],
        })

    out_path = DATA_PROCESSED / "sft_coldstart.jsonl" \
        if args.out.startswith("data") else __import__("pathlib").Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"冷启动 SFT 数据已生成: {out_path}  ({len(rows)} 条, 跳过 {skipped})")


if __name__ == "__main__":
    main()
