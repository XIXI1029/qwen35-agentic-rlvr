# -*- coding: utf-8 -*-
# =====================================================================
# run_sft.py —— Phase 3 冷启动 SFT
#
# 【输入】 data/processed/sft_coldstart.jsonl（build_sft_coldstart.py 产物）
#    每行 = {"messages": [user题, assistant解题轨迹]} + question/ground_truth
# 【输出】 outputs/qwen3.5-4b-sft/  （merge 后的完整模型，可直接评估/做 GRPO 起点）
#
# 【要点】
#   - 用 bf16 + LoRA（Qwen3.5 社区反馈不推荐 4bit QLoRA）
#   - 冻结多模态视觉塔，只训文本注意力的低秩适配器（显存主力 ~8.4GB）
#   - 12GB 卡用 batch=1 + 梯度累积 + gradient_checkpointing 压住激活
#   - 训练文本 = chat_template(user题 + assistant轨迹)，与 RL/评估格式一致
#
# 【用法】 python scripts/run_sft.py --config configs/sft_config.yaml
# =====================================================================

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from _bootstrap import DATA_PROCESSED, OUTPUTS_DIR, load_yaml, logger
from model_utils import ensure_model, require_gpu

# 先配好环境再 import 重型库
from transformers import AutoModelForImageTextToText, AutoTokenizer


# ---------------------------------------------------------------
# 1. 加载 + 冻结视觉 + 挂 LoRA（和 smoke_test 同一套路）
# ---------------------------------------------------------------
def load_trainable_model(model_id: str):
    # ⚠️ 必须加载【本地路径】：若直接传模型 id 字符串，transformers 会去 HF hub
    # 下载（国内 xet 通道 401/被墙）。ensure_model 会先找本地/registry，
    # 都没有则自动下载（服务器从零跑用得上）。
    local_path = ensure_model(model_id) if not os.path.isdir(model_id) else model_id
    logger.info(f"加载基座(本地): {local_path}  (bf16, 冻结视觉, 挂 LoRA)")
    model = AutoModelForImageTextToText.from_pretrained(
        local_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=False,
        local_files_only=True,        # 强制只用本地文件，绝不联网
    )
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 冻结视觉塔
    for name, p in model.named_parameters():
        if "visual" in name.lower() or "vision" in name.lower():
            p.requires_grad_(False)

    # 挂 LoRA（只训 q/k/v/o 投影）
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # 兜底：视觉上任何被 LoRA 误挂的适配器也冻结
    for name, p in model.named_parameters():
        if ("visual" in name.lower() or "vision" in name.lower()) and p.requires_grad:
            p.requires_grad_(False)
    model.print_trainable_parameters()
    model.train()
    return model, tokenizer


# ---------------------------------------------------------------
# 2. 把冷启动 JSONL 变成"text"训练集
# ---------------------------------------------------------------
def build_dataset(jsonl_path, tokenizer):
    """messages -> 应用 chat_template 得到纯文本序列，交给 SFT 续写训练。"""
    from datasets import Dataset

    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False
            )
            rows.append({"text": text, "question": rec["question"],
                         "ground_truth": rec["ground_truth"]})
    logger.info(f"冷启动 SFT 样本数: {len(rows)}")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sft_config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖 num_train_epochs（服务器想多训可传，如 --epochs 3）")
    parser.add_argument("--data", default=None,
                        help="覆盖 SFT 数据路径（默认 data/processed/sft_coldstart.jsonl）")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    tr = cfg["training"]
    if args.epochs is not None:
        tr["num_train_epochs"] = args.epochs
    if args.data is not None:
        tr["_data_override"] = args.data   # 不会传进 SFTConfig，仅本地用
    sft_data_path = (args.data or DATA_PROCESSED / "sft_coldstart.jsonl")
    require_gpu()    # 明确自检 GPU，别让训练在后半段才报 CUDA 错误

    # ---- 3.1 数据 + 模型 ----
    model, tokenizer = load_trainable_model(cfg["model"]["model_id"])
    train_ds = build_dataset(sft_data_path, tokenizer)

    # ---- 3.2 组 SFTConfig（按 trl 1.12 实际字段过滤，防止参数名漂移报错）----
    from trl import SFTConfig, SFTTrainer
    import inspect
    valid = set(inspect.signature(SFTConfig.__init__).parameters)
    tr["output_dir"] = str(OUTPUTS_DIR / "qwen3.5-4b-sft")
    cfg_kwargs = {k: v for k, v in tr.items() if k in valid}
    logger.info(f"SFTConfig 使用参数: {sorted(cfg_kwargs)}")
    sft_cfg = SFTConfig(**cfg_kwargs)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,   # ⚠️ trl 1.12 参数名，旧版叫 tokenizer
        args=sft_cfg,
        train_dataset=train_ds,
    )

    # ---- 3.3 训练 ----
    logger.info("开始 SFT 训练 ...")
    t0 = time.time()
    trainer.train()
    logger.info(f"SFT 训练完成，用时 {(time.time()-t0)/60:.1f} 分钟")

    # ---- 3.4 merge 成完整权重并保存（供 GRPO/评估直接加载）----
    save_dir = OUTPUTS_DIR / "qwen3.5-4b-sft"
    save_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()      # 把 LoRA 适配器并回主权重
    # ⚠️ merge_and_unload 会把权重上转到 fp32，必须显式转回 bf16 再存，
    # 否则 4.5B 参数会写出 ~17GB（fp32）占满磁盘且后续加载慢。
    merged = merged.to(torch.bfloat16)
    merged.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"SFT 模型已保存: {save_dir}")


if __name__ == "__main__":
    main()
