# -*- coding: utf-8 -*-
# =====================================================================
# inspect_data.py —— 快速探查数据集结构（写 prepare_data.py 前必用）
#
# 不依赖猜测 Open-AgentRL 数据集有哪些列，先跑这个脚本看真实 schema：
#   - 字段名 / 类型
#   - 前几行样本（截断展示）
#   - 总行数（非 streaming 时）
#
# 用法
#   python scripts/inspect_data.py --dataset Gen-Verse/Open-AgentRL-30K
#   python scripts/inspect_data.py --dataset Gen-Verse/Open-AgentRL-SFT-3K --n 2
# =====================================================================

from __future__ import annotations

import argparse
import json

from _bootstrap import DATA_RAW, logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HF 数据集 id")
    parser.add_argument("--n", type=int, default=2, help="展示前几条样本")
    parser.add_argument("--stream", action="store_true",
                        help="流式读取（只取头几条，不整包下载）")
    args = parser.parse_args()

    from datasets import load_dataset

    logger.info(f"数据集: {args.dataset}   stream={args.stream}")
    if args.stream:
        # streaming：只拉取需要的前几条，适合先看 schema
        ds = load_dataset(args.dataset, split="train", streaming=True,
                          cache_dir=str(DATA_RAW))
        rows = [next(iter(ds)) for _ in range(args.n)]
        # 已知列名
        cols = list(rows[0].keys())
        logger.info(f"列名({len(cols)}): {cols}")
        logger.info(f"前 {args.n} 条样本字段类型推断: "
                    f"{ {c: type(rows[0][c]).__name__ for c in cols} }")
        for i, r in enumerate(rows):
            logger.info(f"---- sample[{i}] ----")
            for c in cols:
                v = r[c]
                s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                if len(s) > 600:           # 截断长轨迹，只看结构
                    s = s[:600] + " ...(截断)"
                logger.info(f"  {c}: {s}")
    else:
        # 非 streaming：整包下载（30K 较大，慎用），能拿到 len
        ds = load_dataset(args.dataset, split="train", cache_dir=str(DATA_RAW))
        logger.info(f"总行数: {len(ds)}")
        cols = list(ds.column_names)
        logger.info(f"列名({len(cols)}): {cols}")
        logger.info(f"features: {ds.features}")
        for i in range(args.n):
            logger.info(f"---- sample[{i}] ----")
            for c in cols:
                s = str(ds[i][c])
                logger.info(f"  {c}: {s[:300]}")


if __name__ == "__main__":
    main()
