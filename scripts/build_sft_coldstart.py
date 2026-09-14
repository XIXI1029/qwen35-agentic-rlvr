# -*- coding: utf-8 -*-
# =====================================================================
# build_sft_coldstart.py —— 生成冷启动 SFT 数据（Rejection-Sampling SFT）
#
# 背景/为什么
#   Open-AgentRL-SFT-3K 的 91% 带"代码工具调用"，和本项目的纯数学 RLVR
#   （无工具沙箱）格式不符。因此冷启动改用**自蒸馏**：
#   - 从 RL 数学池抽 N 道题
#   - 用 4B-Base 自己采样若干次解题轨迹（Chain-of-Thought）
#   - 用可验证奖励（verifier.verify）筛出答对的轨迹作为 SFT 正样本
#
# 为什么这在方法上站得住（写/讲）
#   "Rejection-Sampling SFT 冷启动"是 RL 训练的标准前置：
#   让策略先把自己的"正确且格式受控"的行为看几遍，再进 GRPO 在线探索，
#   能显著提升 RL 稳定性、缩短收敛（和 9B 这类对齐模型无关，纯自举）。
#
# 用法
#   python scripts/build_sft_coldstart.py --questions 600 --attempts 3
# =====================================================================

from __future__ import annotations

import argparse
import json
import random
import time

import torch
from _bootstrap import DATA_PROCESSED, logger
from model_utils import load_qwen35_model
from verifier import verify


def make_chat_prompt(tokenizer, question: str) -> str:
    """把题目包成单轮对话 prompt（Base tokenizer 自带 chat_template）。"""
    messages = [{
        "role": "user",
        "content": question
        + "\n请一步步推理（Let's think step by step），"
          "最终用一个数字给出答案。",
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def sample_solution(model, tokenizer, prompt_text: str, max_new: int) -> str:
    """采样一条解题轨迹，返回纯文本（已去掉 prompt 部分）。"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.8,          # 采样 = 探索多种解法，提高"至少一次答对"概率
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=600,
                        help="要处理的题目数（越多 SFT 样本越足但越耗时）")
    parser.add_argument("--attempts", type=int, default=3,
                        help="每题最多采样几次，直到答对")
    parser.add_argument("--max-new", type=int, default=512,
                        help="每条解题轨迹最长 token")
    parser.add_argument("--out", default="data/processed/sft_coldstart.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # ---- 1. 载入数学题池 ----
    rl_path = DATA_PROCESSED / "rl.jsonl"
    rows = [json.loads(l) for l in
            open(rl_path, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    questions = rows[:args.questions]
    logger.info(f"数学题池共 {len(rows)}，本次抽样 {len(questions)} 道")

    # ---- 2. 载入模型（4B-Base, bf16）----
    model, tokenizer = load_qwen35_model("Qwen/Qwen3.5-4B-Base")
    model.eval()

    # ---- 3. 逐题采样 + 可验证筛选 ----
    kept, attempted_q = [], 0
    t0 = time.time()
    out_path = DATA_PROCESSED / "sft_coldstart.jsonl" if args.out.startswith("data") \
        else __import__("pathlib").Path(args.out)

    for i, item in enumerate(questions):
        q, gt = item["prompt"], item["ground_truth"]
        prompt_text = make_chat_prompt(tokenizer, q)
        solution = None
        # 采样直到答对或达到 attempts 上限
        for _ in range(args.attempts):
            sol = sample_solution(model, tokenizer, prompt_text, args.max_new)
            if verify(sol, gt):
                solution = sol
                break
        attempted_q += 1
        if solution is not None:
            rec = {
                "question": q,
                "ground_truth": gt,
                "solution": solution,
                # 标准 messages 结构，SFT 阶段直接应用 chat_template 得到训练文本
                "messages": [
                    {"role": "user", "content": q
                     + "\n请一步步推理（Let's think step by step），"
                       "最终用一个数字给出答案。"},
                    {"role": "assistant", "content": solution},
                ],
            }
            kept.append(rec)
        if (i + 1) % 25 == 0:
            logger.info(f"进度 {i+1}/{len(questions)}  已保留 {len(kept)} 条"
                        f"  耗时 {time.time()-t0:.0f}s")

    # ---- 4. 写出 ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    succ = len(kept) / max(1, attempted_q)
    logger.info(f"冷启动 SFT 数据已生成: {out_path}")
    logger.info(f"保留 {len(kept)}/{attempted_q} 条正确轨迹"
                f" (成功率 {succ*100:.1f}%)，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
