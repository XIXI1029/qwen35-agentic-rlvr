# -*- coding: utf-8 -*-
# =====================================================================
# evaluate.py —— 单模型评测（Phase 1 基线 & Phase 5 正式对比 & 服务器跑 9B 共用）
#
# 【服务器场景说明】
#   本机 12GB 卡跑不动 9B（加载即 segfault）。把项目复制到大显存 Linux 服务器后：
#     python scripts/evaluate.py --model Qwen/Qwen3.5-9B-Base --task gsm8k --max-samples 50 --strategy auto
#   --strategy auto 会依次尝试：
#     ① bf16 全上 GPU（大显存卡，如 24GB A100/4090）  -> 最快最准
#     ② 4-bit 量化上 GPU（12~16GB 卡，Linux 下 bnb 正常）
#     ③ 纯 CPU float16（都没有 GPU 时的兜底，很慢）
#   模型不存在时会自动下载（见 model_utils.ensure_model）。
#
# 【用法】
#   python scripts/evaluate.py --model Qwen/Qwen3.5-4B-Base --task gsm8k --max-samples 50
#   python scripts/evaluate.py --model Qwen/Qwen3.5-9B-Base --task aime2024 --strategy auto
#   python scripts/evaluate.py --model outputs/qwen3.5-4b-grpo --task gsm8k --max-samples 50 --tag grpo
# =====================================================================

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from _bootstrap import DATA_RAW, OUTPUTS_DIR, logger
from model_utils import (
    ensure_model,
    extract_last_number,
    generate_answer,
    load_qwen35_model,
    normalize_answer,
)


# ---------------------------------------------------------------
# 任务定义：每种 benchmark 提供"取样函数" -> [{"prompt","ground_truth"}]
# ---------------------------------------------------------------
def _gsm8k_samples(max_samples: int, seed: int = 0):
    from datasets import load_dataset
    logger.info("加载 GSM8K test (openai/gsm8k) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test",
                      cache_dir=str(DATA_RAW))
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    out = []
    for i in idxs[:max_samples]:
        ans = ds[i]["answer"]
        m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", ans)
        out.append({"prompt": ds[i]["question"],
                    "ground_truth": m.group(1) if m else ""})
    return out


def _aime_samples(max_samples: int, seed: int = 0):
    """AIME2024：每题答案是 0~999 的整数。

    数据集做容错：依次尝试几个公开 id；字段名自动探测（problem/question/prompt
    与 answer/ground_truth）；有 year 列时优先筛 2024。若 2024 不足则用全部可用。
    AIME 全年题量 < 50，实际取样数以题量为准。
    """
    from datasets import load_dataset
    cands = ["AI-MO/aimo-validation-aime",
             "Hothan/AIME_2024",
             "Maxwell-Jia/AIME_2024"]
    ds, last_err = None, None
    for dsid in cands:
        try:
            logger.info(f"尝试加载 AIME 数据集: {dsid}")
            ds = load_dataset(dsid, split="train", cache_dir=str(DATA_RAW))
            logger.info(f"成功: {dsid} ({len(ds)} 条)")
            break
        except Exception as e:            # 该 id 不存在/被墙则试下一个
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(f"AIME 数据集都加载失败: {last_err}")

    cols = {c.lower(): c for c in ds.column_names}
    pcol = next((cols[k] for k in ("problem", "question", "prompt") if k in cols), None)
    acol = next((cols[k] for k in ("answer", "ground_truth", "final_answer")
                 if k in cols), None)
    if pcol is None or acol is None:
        raise RuntimeError(f"不认识 AIME 数据列: {ds.column_names}")

    rows = []
    for i in range(len(ds)):
        yr = ds[i]["year"] if "year" in cols else None
        if yr is not None and str(yr).strip().isdigit() and int(yr) != 2024:
            continue                     # 优先只要 2024 的题
        # 答案规范化成纯整数
        gt = re.sub(r"[^0-9]", "", str(ds[i][acol]))
        if not gt:
            continue
        rows.append({"prompt": str(ds[i][pcol]), "ground_truth": gt})
        if len(rows) >= max_samples:
            break
    if len(rows) == 0 and any("year" in cols for _ in [1]):
        # 上面年份过滤过头则退化为不限年份重取
        for i in range(min(max_samples, len(ds))):
            gt = re.sub(r"[^0-9]", "", str(ds[i][acol]))
            if gt:
                rows.append({"prompt": str(ds[i][pcol]), "ground_truth": gt})
    logger.info(f"AIME 取样 {len(rows)} 题")
    return rows


TASKS = {
    "gsm8k": _gsm8k_samples,
    "aime2024": _aime_samples,
}


# ---------------------------------------------------------------
# 按策略加载（auto: bf16 GPU -> 4bit GPU -> CPU fp16）
# ---------------------------------------------------------------
def _load_with_strategy(model_id: str, strategy: str):
    """根据 strategy 返回 (model, tokenizer)。异常自动降级。"""
    path = ensure_model(model_id)          # 不存在自动下载
    orders = {
        # 每项: (load 参数, 描述)
        "gpu": [dict(load_in_4bit=False, cpu=False, key="bf16-GPU"),
                dict(load_in_4bit=True, cpu=False, key="4bit-GPU"),
                dict(load_in_4bit=False, cpu=True, key="CPU-fp16")],
        "auto": [dict(load_in_4bit=False, cpu=False, key="bf16-GPU"),
                 dict(load_in_4bit=True, cpu=False, key="4bit-GPU"),
                 dict(load_in_4bit=False, cpu=True, key="CPU-fp16")],
        "4bit": [dict(load_in_4bit=True, cpu=False, key="4bit-GPU")],
        "cpu": [dict(load_in_4bit=False, cpu=True, key="CPU-fp16")],
    }
    last = None
    for opt in orders.get(strategy, orders["auto"]):
        key = opt.pop("key")
        try:
            logger.info(f"[加载策略] 尝试 {key} ...")
            model, tok = load_qwen35_model(path, **opt)
            return model, tok
        except Exception as e:             # OOM / segfault前兆 / bnb缺失都降级
            logger.warning(f"[加载策略] {key} 失败({type(e).__name__}: {str(e)[:120]})，尝试下一档")
            last = e
            torch.cuda.empty_cache()
    raise RuntimeError(f"模型 {model_id} 所有加载策略都失败。最后错误: {last}")


def run_eval(model_id: str, task: str, max_samples: int,
             strategy: str = "auto", max_new: int = 256) -> dict:
    """加载 -> 逐题生成 -> 比对 -> 汇总。"""
    model, tokenizer = _load_with_strategy(model_id, strategy)
    samples = TASKS[task](max_samples)

    correct = 0
    records = []
    t0 = time.time()
    for n, s in enumerate(samples, 1):
        out = generate_answer(model, tokenizer, s["prompt"],
                              max_new_tokens=max_new)
        pred = extract_last_number(out)
        pred_f, gt_f = normalize_answer(pred), normalize_answer(s["ground_truth"])
        ok = (pred_f is not None and gt_f is not None and pred_f == gt_f)
        correct += int(ok)
        records.append({"idx": n, "prompt": s["prompt"], "gt": s["ground_truth"],
                        "pred": pred, "output": out[:200], "correct": ok})
        if n % 10 == 0 or n == len(samples):
            logger.info(f"[{task}] {n}/{len(samples)}  当前正确率 "
                        f"{correct}/{n} = {correct/n*100:.1f}%")

    result = {
        "model": model_id, "task": task,
        "n": len(samples), "correct": correct,
        "accuracy": round(correct / max(1, len(samples)), 4),
        "seconds": round(time.time() - t0, 1),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return result, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="模型 id 或本地目录")
    parser.add_argument("--task", choices=list(TASKS), default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new", type=int, default=256,
                        help="生成最长 token（CPU 推理建议调小）")
    parser.add_argument("--strategy", choices=["auto", "gpu", "4bit", "cpu"],
                        default="auto",
                        help="auto=bf16 GPU->4bit->CPU 自动降级(默认)")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    result, records = run_eval(args.model, args.task, args.max_samples,
                               strategy=args.strategy, max_new=args.max_new)

    results_dir = OUTPUTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.model.replace("/", "-").split("-")[-1]
    out_path = results_dir / f"{args.task}_{tag}_{datetime.now():%Y%m%d-%H%M}.json"
    # 主文件不含逐题明细；明细单独存
    summary = {k: v for k, v in result.items()}
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    out_path.with_suffix(".samples.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"评测完成: {args.task} 正确率 = {result['accuracy']*100:.2f}% "
                f"({result['correct']}/{result['n']})")
    logger.info(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
