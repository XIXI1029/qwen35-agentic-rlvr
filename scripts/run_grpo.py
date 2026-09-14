# -*- coding: utf-8 -*-
# =====================================================================
# run_grpo.py —— GRPO 强化学习（本项目核心）
#
# 输入 data/processed/grpo_train.jsonl（build_grpo_pool.py 产物）
#         outputs/qwen3.5-4b-sft/（SFT 起点，也可回退到 Base）
# 输出 outputs/qwen3.5-4b-grpo/（merge 后完整权重）+ 训练日志曲线
#
# 原理一句话每个问题采样 N 条轨迹组成一组，先各自算"可验证奖励"，
# 再组内归一化得到相对优势：答对的轨迹概率被推高、答错的被压低；
# 用 KL 惩罚拉住策略别跑飞（由 SFT/Base 当 ref 锚点）。
# RLVR 奖励verifier.verify(completion, ground_truth) —— 数字答案比对。
#   数据集里的 ground_truth 列由 TRL 1.12 自动作为 kwargs 传进来（已按组展开）。
#
# 用法
#   python scripts/run_grpo.py --config configs/grpo_config.yaml
#   python scripts/run_grpo.py --max-steps 120      # 覆盖步数上限
# =====================================================================

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path

import torch
from _bootstrap import DATA_PROCESSED, OUTPUTS_DIR, load_yaml, logger
from model_utils import require_gpu
from verifier import verify

# 先配好环境
from transformers import AutoModelForImageTextToText, AutoTokenizer


# ---------------------------------------------------------------
# 1. 奖励函数（可验证奖励 / RLVR + 部分分）
# ---------------------------------------------------------------
def make_math_reward(weights: dict | None = None):
    """构造数学奖励函数（v2：从"纯 0/1"升级为"正确性 + 格式 + 长度"）。

    为什么加部分分：组大小有限时，若全组同分（都错/都对），优势=0 → 该步不产生有效梯度。
    给"格式正确"一点分，能让同组内产生差异，梯度信号更密。

    权重来自 configs/*.yaml 的 reward.weights：
      answer_correct  正确性（主信号）
      format_correct  是否在结尾给出数值/答案标记（辅助信号）
      length_penalty  长度惩罚（负系数，鼓励简洁）
    """
    import re
    w = weights or {}
    w_corr = float(w.get("answer_correct", 1.0))
    w_fmt = float(w.get("format_correct", 0.0))
    w_len = float(w.get("length_penalty", 0.0))
    _num_tail = re.compile(r"[-+]?\d")

    def math_reward(prompts, completions, ground_truth=None, **kwargs):
        rewards = []
        for comp, gt in zip(completions, ground_truth or [None] * len(completions)):
            correct = 1.0 if (gt is not None and verify(comp, gt)) else 0.0
            # 格式分：结尾 60 字符内出现数字，或含 boxed 标记
            tail = comp[-60:]
            fmt = 1.0 if (_num_tail.search(tail) or "boxed" in comp.lower()) else 0.0
            r = w_corr * correct + w_fmt * fmt + w_len * len(comp.split())
            rewards.append(r)
        return rewards

    return math_reward


# ---------------------------------------------------------------
# 2. 加载模型：SFT checkpoint 优先，否则 Base；冻结视觉 + 挂 LoRA
# ---------------------------------------------------------------
def load_trainable_model(model_path_or_id: str):
    from model_utils import ensure_model
    # SFT checkpoint 本地路径直接用；是模型 id 且本地没有则自动下载
    local = model_path_or_id if os.path.isdir(model_path_or_id) \
        else ensure_model(model_path_or_id)
    logger.info(f"GRPO 起点: {local}")
    model = AutoModelForImageTextToText.from_pretrained(
        local, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=False,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for name, p in model.named_parameters():
        if "visual" in name.lower() or "vision" in name.lower():
            p.requires_grad_(False)

    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    for name, p in model.named_parameters():
        if ("visual" in name.lower() or "vision" in name.lower()) and p.requires_grad:
            p.requires_grad_(False)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="grpo_config.yaml")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="覆盖最大训练步数（预算控制的关键）")
    args = parser.parse_args()
    require_gpu()    # 明确自检 GPU
    cfg = load_yaml(args.config)
    tr = cfg["training"]

    # ---- 3.1 模型 + 数据 ----
    start_model = cfg["model"]["sft_model_dir"]
    if not os.path.isdir(start_model):
        logger.warning(f"SFT checkpoint 不存在({start_model})，回退到 Base 模型")
        start_model = cfg["model"]["base_model_id"]
    model, tokenizer = load_trainable_model(start_model)

    from datasets import Dataset
    rl_path = DATA_PROCESSED / "grpo_train.jsonl"
    rows = [json.loads(l) for l in open(rl_path, encoding="utf-8") if l.strip()]
    logger.info(f"GRPO 训练题数: {len(rows)}")
    train_ds = Dataset.from_list(rows)

    # ---- 3.2 GRPOConfig：按 trl 1.12 实际字段名过滤，防漂移 ----
    from trl import GRPOConfig, GRPOTrainer
    valid = set(inspect.signature(GRPOConfig.__init__).parameters)

    # 采样超参来自 configs/grpo_config.yaml 的 generation 段
    gen = cfg.get("generation", {})
    # 输出目录：尊重配置（相对路径按项目根解析），默认 outputs/qwen3.5-4b-grpo
    from _bootstrap import ROOT
    _out = cfg.get("training", {}).get("output_dir") or "outputs/qwen3.5-4b-grpo"
    out_dir = _out if os.path.isabs(_out) else str((ROOT / _out).resolve())
    cfg_kwargs = {
        "output_dir": out_dir,
        "temperature": gen.get("temperature", 0.9),
        "top_p": gen.get("top_p", 0.95),
    }
    cfg_kwargs.update({k: v for k, v in tr.items() if k in valid and k != "output_dir"})
    if args.max_steps is not None:
        cfg_kwargs["max_steps"] = args.max_steps
    # 前置校验：TRL 要求 generation_batch_size(=batch×累积) 能被 num_generations 整除
    _bs = cfg_kwargs.get("per_device_train_batch_size", 1) * \
        cfg_kwargs.get("gradient_accumulation_steps", 1)
    _ng = cfg_kwargs.get("num_generations", 1)
    if _bs % _ng != 0:
        raise SystemExit(
            f"配置错误：per_device_train_batch_size × gradient_accumulation_steps "
            f"= {_bs}，必须能被 num_generations = {_ng} 整除。\n"
            f"改法：让 batch×累积 成为 {_ng} 的倍数（例如 batch=1 时 accum={_ng}，"
            f"或 batch=2 时 accum={_ng // 2 if _ng % 2 == 0 else _ng}）。")
    logger.info(f"GRPOConfig 使用参数: {sorted(cfg_kwargs)}")
    grpo_cfg = GRPOConfig(**cfg_kwargs)

    # ---- 3.3 训练 ----
    # 奖励：可验证性(RLVR) + 格式 + 长度（权重来自 yaml，v2 起启用）
    reward_fn = make_math_reward(cfg.get("reward", {}).get("weights"))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    # ---- 3.4 自动断点续训：output_dir 里若有 checkpoint-*，从最新一个接着跑 ----
    import glob as _glob
    ckpts = sorted(
        _glob.glob(os.path.join(cfg_kwargs["output_dir"], "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),   # 按步数排序
    )
    resume = ckpts[-1] if ckpts else None
    if resume:
        logger.info(f"检测到 checkpoint，自动续训: {resume}")
    else:
        logger.info("未发现 checkpoint，从头训练")

    logger.info("开始 GRPO 训练（RLVR）...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    logger.info(f"GRPO 训练结束，用时 {(time.time()-t0)/60:.1f} 分钟")

    # ---- 3.4 merge + 保存 ----
    from _bootstrap import ROOT as _ROOT
    _sd = cfg.get("output", {}).get("save_dir") or "outputs/qwen3.5-4b-grpo"
    save_dir = _ROOT / _sd if not os.path.isabs(_sd) else Path(_sd)
    save_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged = merged.to(torch.bfloat16)   # merge 会上转 fp32，务必转回 bf16 再存
    merged.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"GRPO 模型已保存: {save_dir}")


if __name__ == "__main__":
    main()
