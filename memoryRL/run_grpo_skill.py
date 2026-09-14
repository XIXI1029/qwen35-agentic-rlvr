# -*- coding: utf-8 -*-
# =====================================================================
# run_grpo_skill.py —— Skill 选择策略的 GRPO 训练
#
# 与 scripts/run_grpo.py 的唯一区别：
#   数据 = 自建技能选择题库；奖励 = 技能集合匹配（skill_verifier）
#   其余（模型加载/LoRA/冻结视觉/GRPOConfig/断点续训/merge 保存）完全复用
#   => 这是"同一套 RLVR 流水线，换可验证任务即可复用"的一次验证
#
# 用法：
#   python memoryRL/run_grpo_skill.py --config memoryRL/skill_config.yaml --max-steps 800
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import ROOT, load_yaml, logger          # noqa: E402
from model_utils import require_gpu                     # noqa: E402
from skill_verifier import make_skill_reward            # noqa: E402

from transformers import AutoModelForImageTextToText, AutoTokenizer   # noqa: E402


def load_trainable_model(model_path_or_id: str):
    """与主项目一致的加载：本地路径优先，缺失自动下载；冻结视觉 + LoRA。"""
    from model_utils import ensure_model
    local = model_path_or_id if os.path.isdir(model_path_or_id) \
        else ensure_model(model_path_or_id)
    logger.info(f"Skill-GRPO 起点: {local}")
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
    ap.add_argument("--config", default="skill_config.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--start", default=None,
                    help="覆盖起点模型（用于对比 Base / SFT / 已 RL 模型）")
    ap.add_argument("--out-dir", default=None,
                    help="覆盖输出目录（双臂对照实验用：不同起点写到不同目录）")
    ap.add_argument("--save-mode", choices=["adapter", "merged"], default="adapter",
                    help="adapter=只存 LoRA 适配器(省内存，推荐; 评估时自动加载基座+适配器)；"
                         "merged=合并成完整权重(省内存差的机器上 merge 会临时上转 fp32，易 OOM)")
    args = ap.parse_args()

    require_gpu()
    cfg = load_yaml(args.config)
    tr = cfg["training"]

    from model_utils import ensure_model
    start = args.start or cfg["model"]["sft_model_dir"]
    if not os.path.isdir(start):
        logger.warning(f"起点不是本地目录({start})，尝试解析/下载")
        start = ensure_model(start) if start != cfg["model"]["base_model_id"] \
            else ensure_model(cfg["model"]["base_model_id"])
    local_start_path = start          # 已是本地目录；保存适配器时要记录基座路径
    model, tokenizer = load_trainable_model(local_start_path)

    # ---- 数据：自建技能选择题库 ----
    from datasets import Dataset
    pool = ROOT / cfg["data"]["train_jsonl"]
    rows = [json.loads(l) for l in open(pool, encoding="utf-8") if l.strip()]
    logger.info(f"Skill-RL 题数: {len(rows)}  (from {pool})")
    train_ds = Dataset.from_list(rows)

    # ---- GRPOConfig（字段过滤 + 整除校验）----
    from trl import GRPOConfig, GRPOTrainer
    valid = set(inspect.signature(GRPOConfig.__init__).parameters)
    gen = cfg.get("generation", {})
    _out = args.out_dir or tr.get("output_dir") or "outputs/qwen3.5-4b-grpo-skill"
    out_dir = _out if os.path.isabs(_out) else str((ROOT / _out).resolve())
    logger.info(f"输出目录: {out_dir}")
    cfg_kwargs = {"output_dir": out_dir,
                  "temperature": gen.get("temperature", 0.9),
                  "top_p": gen.get("top_p", 0.95)}
    cfg_kwargs.update({k: v for k, v in tr.items() if k in valid and k != "output_dir"})
    if args.max_steps is not None:
        cfg_kwargs["max_steps"] = args.max_steps
    _bs = cfg_kwargs.get("per_device_train_batch_size", 1) * \
        cfg_kwargs.get("gradient_accumulation_steps", 1)
    _ng = cfg_kwargs.get("num_generations", 1)
    if _bs % _ng != 0:
        raise SystemExit(f"配置错误：batch×累积={_bs} 必须能被 num_generations={_ng} 整除")
    logger.info(f"GRPOConfig: {sorted(cfg_kwargs)}")
    grpo_cfg = GRPOConfig(**cfg_kwargs)

    reward_fn = make_skill_reward(cfg.get("reward", {}).get("weights"))
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=grpo_cfg,
                          train_dataset=train_ds, processing_class=tokenizer)

    ckpts = sorted(_glob.glob(os.path.join(out_dir, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    resume = ckpts[-1] if ckpts else None
    logger.info(f"续训起点: {resume or '从头训练'}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    logger.info(f"Skill-GRPO 训练结束，用时 {(time.time()-t0)/60:.1f} 分钟")

    _sd = args.out_dir or cfg.get("output", {}).get("save_dir") or "outputs/qwen3.5-4b-grpo-skill"
    save_dir = ROOT / _sd if not os.path.isabs(_sd) else Path(_sd)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mode == "adapter":
        # 只存 LoRA 适配器（几 MB）——避开 merge 的 fp32 上转（16GB 内存机器必选）
        model.save_pretrained(save_dir)                    # peft -> adapter_model.safetensors
        tokenizer.save_pretrained(save_dir)
        # 写一个"基座路径"提示文件，评估脚本据此自动拼装
        (save_dir / "base_model_hint.txt").write_text(
            str(local_start_path), encoding="utf-8")
        logger.info(f"已保存 LoRA 适配器: {save_dir} (基座 {local_start_path})")
    else:
        merged = model.merge_and_unload().to(torch.bfloat16)
        merged.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Skill-GRPO 已合并保存: {save_dir}")


if __name__ == "__main__":
    main()
