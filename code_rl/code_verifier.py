# -*- coding: utf-8 -*-
# =====================================================================
# code_verifier.py —— Code RLVR 的奖励函数（对应数学实验里的 verifier.py）
#
# 奖励定义模型输出代码 -> 抠出函数 -> 在受限沙箱里跑隐藏测试：
#     全部通过 => 奖励 1.0，否则 0.0
#   并通过并行执行把"每步要跑 num_generations×batch 段代码"的开销压下去。
#
# 与 TRL 的接口GRPOTrainer 会调用 reward_funcs 里的可调用对象：
#     reward(prompts=..., completions=..., ground_truth=[...], **kwargs)
#   其中 ground_truth 来自数据集列（每行一个 JSON 字符串：
#     {"tests": "...", "entry_point": "..."}）
# =====================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))          # 同级模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import logger                                     # noqa: E402
from code_sandbox import extract_python, run_many                 # noqa: E402


def make_code_reward(timeout: float = 3.0,
                     mem_mb: int = 1024,
                     max_workers: int = 8,
                     debug_every: int = 50,
                     weights: dict | None = None):
    """构造代码奖励函数（v2：分档部分分，避免组内全同分导致无梯度）。

    奖励分档（权重可在 yaml 的 reward.weights 覆盖）：
      pass            全部测试通过       -> 1.0
      wrong_answer    跑通但断言失败     -> 0.3   （学会"代码能跑"这一层）
      error           语法/名称等异常    -> 0.1   （学会"能编译"这一层）
      timeout/空代码  完全无效           -> 0.0
      + format_correct 是否给出代码块     -> +0.1
      + length_penalty 长度惩罚（负系数）

    Args:
        timeout/mem_mb/max_workers: 沙箱配置
        debug_every: 每多少次调用打印一次统计
        weights:     部分分权重（来自 code_config.yaml 的 reward.weights）
    """
    w = weights or {}
    w_pass = float(w.get("tests_passed", 1.0))
    w_wrong = float(w.get("wrong_answer", 0.3))
    w_err = float(w.get("error", 0.1))
    w_fmt = float(w.get("format_correct", 0.1))
    w_len = float(w.get("length_penalty", -0.0001))
    state = {"calls": 0, "passed": 0, "total": 0}

    def code_reward(prompts, completions, ground_truth=None, **kwargs):
        # 1) 准备 (candidate_code, test_code) 列表
        items: List[tuple] = []
        for comp, gt in zip(completions, ground_truth or [None] * len(completions)):
            code = extract_python(comp)
            try:
                meta = json.loads(gt) if isinstance(gt, str) else (gt or {})
                tests = meta.get("tests", "")
            except Exception:
                tests = ""
            items.append((code, tests))

        # 2) 并行跑沙箱
        results = run_many(items, timeout=timeout, mem_mb=mem_mb,
                           max_workers=max_workers)

        # 3) 按"失败类型"给部分分 + 格式分 + 长度惩罚
        rewards = []
        for (ok, info), comp in zip(results, completions):
            if ok:
                r = w_pass
            elif info.startswith("timeout") or info.startswith("empty"):
                r = 0.0
            elif info.startswith("AssertionError"):
                r = w_wrong                     # 能跑、但结果不对
            else:
                r = w_err                       # 语法/名称错误等
            if "```" in comp or "def " in comp:
                r += w_fmt
            r += w_len * len(comp.split())
            rewards.append(r)

        state["calls"] += 1
        state["total"] += len(items)
        state["passed"] += sum(1 for (ok, _i) in results if ok)
        if state["calls"] % debug_every == 0:
            logger.info(f"[code_reward] 调用 {state['calls']} 次，"
                        f"累计通过率 {state['passed']/max(1,state['total']):.1%}")
        return rewards

    return code_reward
