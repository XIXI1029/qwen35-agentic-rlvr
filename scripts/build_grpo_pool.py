# -*- coding: utf-8 -*-
# =====================================================================
# build_grpo_pool.py —— 构建 GRPO 在线训练题库（与 SFT 数据不重叠）
#
# 【要点】
#   - 从 GSM8K train 取一批题作为 RLVR 训练池：题相对简单 -> 模型采样容易
#     命中正确答案 -> 奖励稠密 -> GRPO 学得快（12GB 单卡预算下能见效）
#   - 关键设计：**与 SFT 冷启动数据完全不相交**。
#     否则模型等于把看过的题再 RL 一遍，SFT 与 RL 的增益混在一起说不清。
#     SFT 取 seed0 洗牌后的前 1500 题；本脚本取【下一段】1500~2500。
#   - 数据行格式满足 TRL GRPO：{"prompt", "ground_truth"}，ground_truth 会
#     作为额外列自动传给 reward 函数。
#
# 【用法】 python scripts/build_grpo_pool.py --n 1000 --sft_seen 1500
# =====================================================================

from __future__ import annotations

import argparse
import json
import random

from _bootstrap import DATA_RAW, DATA_PROCESSED, logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="GRPO 题库规模")
    parser.add_argument("--sft_seen", type=int, default=1500,
                        help="SFT 冷启动已占用的题数（与 build_sft_from_gsm8k 的 --n 对齐）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/processed/grpo_train.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    logger.info("加载 GSM8K train ...")
    ds = load_dataset("openai/gsm8k", "main", split="train",
                      cache_dir=str(DATA_RAW))
    # 复现与 SFT 相同的洗牌顺序，保证两批题不相交
    idxs = list(range(len(ds)))
    rng = random.Random(args.seed)
    rng.shuffle(idxs)

    # 跳过 SFT 已用段，取下一段
    start = args.sft_seen
    pick = idxs[start:start + args.n]
    logger.info(f"取题区间: [{start}, {start + args.n})  （跳过 SFT 段 [0,{args.sft_seen})）")

    rows = []
    for i in pick:
        q = ds[i]["question"]
        ans = ds[i]["answer"]
        val = ans.split("####", 1)[1].strip() if "####" in ans else ""
        if not val:
            continue
        rows.append({"prompt": q, "ground_truth": val, "source": "gsm8k_train"})

    out_path = DATA_PROCESSED / "grpo_train.jsonl" \
        if args.out.startswith("data") else __import__("pathlib").Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"GRPO 题库已写出: {out_path}  ({len(rows)} 题)")


if __name__ == "__main__":
    main()
