# -*- coding: utf-8 -*-
# =====================================================================
# run_grpo_code.py —— Code RLVR 训练（数学实验 run_grpo.py 的"代码版"）
#
# 与 run_grpo.py 的唯一区别：
#   数据 = 代码题库（MBPP），奖励 = 沙箱跑测试（code_verifier）
#   其余（模型加载/LoRA/冻结视觉/GRPOConfig/断点续训/保存）完全一致
#   => 这就是"同一套 RLVR 流水线，换可验证任务即可复用"的证明
#
# 用法
#   python code_rl/run_grpo_code.py --config code_rl/code_config.yaml --max-steps 300
# =====================================================================

from __future__ import annotations

import argparse
import glob as _glob
import inspect
import json
import os
import sys
import time
from pathlib import Path

import torch

# 引入父目录公共模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import DATA_PROCESSED, OUTPUTS_DIR, load_yaml, logger  # noqa: E402
from model_utils import ensure_model, require_gpu                      # noqa: E402
from code_verifier import make_code_reward                            # noqa: E402

from transformers import AutoModelForImageTextToText, AutoTokenizer   # noqa: E402


def load_trainable_model(model_path_or_id: str):
    """与 run_grpo.py 相同的加载逻辑：本地路径优先，缺失自动下载。"""
    from model_utils import ensure_model as _ensure
    local = model_path_or_id if os.path.isdir(model_path_or_id) else _ensure(model_path_or_id)
    logger.info(f"Code-GRPO 起点: {local}")
    model = AutoModelForImageTextToText.from_pretrained(
        local, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=False, local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for name, p in model.named_parameters():
        if "visual" in name.lower() or "vision" in name.lower():
            p.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    for name, p in model.named_parameters():
        if ("visual" in name.lower() or "vision" in name.lower()) and p.requires_grad:
            p.requires_grad_(False)
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="code_config.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    require_gpu()
    cfg = load_yaml(args.config)
    tr = cfg["training"]
    sb = cfg.get("sandbox", {})

    # ---- 起点：默认复用数学实验训好的 SFT 模型（一个起点，两条 RL 实验）----
    start = cfg["model"]["sft_model_dir"]
    if not os.path.isdir(start):
        logger.warning(f"SFT 起点不存在({start})，回退 Base")
        start = cfg["model"]["base_model_id"]
    model, tokenizer = load_trainable_model(start)

    # ---- 数据：代码题库 ----
    from datasets import Dataset
    pool = Path(cfg["data"]["rl_jsonl"])
    if not pool.is_absolute():
        pool = Path(__file__).resolve().parents[1] / pool
    rows = [json.loads(l) for l in open(pool, encoding="utf-8") if l.strip()]
    logger.info(f"Code-RL 题数: {len(rows)}  (from {pool})")
    train_ds = Dataset.from_list(rows)

    # ---- GRPOConfig（字段名过滤，防 trl 版本漂移）----
    from trl import GRPOConfig, GRPOTrainer
    valid = set(inspect.signature(GRPOConfig.__init__).parameters)
    gen = cfg.get("generation", {})
    from _bootstrap import ROOT
    _out = cfg.get("training", {}).get("output_dir") or "outputs/qwen3.5-4b-grpo-code"
    out_dir = _out if os.path.isabs(_out) else str((ROOT / _out).resolve())
    cfg_kwargs = {
        "output_dir": out_dir,
        "temperature": gen.get("temperature", 0.9),
        "top_p": gen.get("top_p", 0.95),
    }
    cfg_kwargs.update({k: v for k, v in tr.items() if k in valid and k != "output_dir"})
    if args.max_steps is not None:
        cfg_kwargs["max_steps"] = args.max_steps
    # 前置校验：batch×累积 必须能被 num_generations 整除（TRL 硬约束）
    _bs = cfg_kwargs.get("per_device_train_batch_size", 1) * \
        cfg_kwargs.get("gradient_accumulation_steps", 1)
    _ng = cfg_kwargs.get("num_generations", 1)
    if _bs % _ng != 0:
        raise SystemExit(
            f"配置错误：batch({cfg_kwargs.get('per_device_train_batch_size')}) × "
            f"累积({cfg_kwargs.get('gradient_accumulation_steps')}) = {_bs}，"
            f"必须能被 num_generations={_ng} 整除。")
    logger.info(f"GRPOConfig: {sorted(cfg_kwargs)}")
    grpo_cfg = GRPOConfig(**cfg_kwargs)

    # ---- 奖励：沙箱跑测试（并行）----
    reward_fn = make_code_reward(
        timeout=sb.get("timeout", 3.0),
        mem_mb=sb.get("mem_mb", 1024),
        max_workers=sb.get("max_workers", 8),
        weights=cfg.get("reward", {}).get("weights"),   # v2：部分分权重
    )

    trainer = GRPOTrainer(
        model=model, reward_funcs=[reward_fn], args=grpo_cfg,
        train_dataset=train_ds, processing_class=tokenizer,
    )

    # ---- 断点续训（同 run_grpo.py）----
    ckpts = sorted(_glob.glob(os.path.join(cfg_kwargs["output_dir"], "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    resume = ckpts[-1] if ckpts else None
    logger.info(f"续训起点: {resume or '从头训练'}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    logger.info(f"Code-GRPO 训练结束，用时 {(time.time()-t0)/60:.1f} 分钟")

    # ---- merge 保存（bf16）----
    from _bootstrap import ROOT as _ROOT
    _sd = cfg.get("output", {}).get("save_dir") or "outputs/qwen3.5-4b-grpo-code"
    save_dir = _ROOT / _sd if not os.path.isabs(_sd) else Path(_sd)
    save_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload().to(torch.bfloat16)
    merged.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Code-GRPO 模型已保存: {save_dir}")


if __name__ == "__main__":
    main()
