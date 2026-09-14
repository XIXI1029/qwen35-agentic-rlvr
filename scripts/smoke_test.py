# -*- coding: utf-8 -*-
# =====================================================================
# smoke_test.py —— Phase 1「技术选型冒烟测试」（关键闸口）
#
# 【为什么必须有这一步】
#   Qwen3.5-4B-Base 是多模态模型 Qwen3_5ForConditionalGeneration，
#   TRL 的 GRPOTrainer 官方文档说"主要支持纯文本 CausalLM"，VLM 支持仍有
#   bug。所以我们先用【极小配置】验证三件事，再决定要不要真花几天跑训练：
#
#   ① 能否加载该模型（bf16，不含视觉解码器梯度）
#   ② 冻结视觉塔 + 只训文本 LoRA 后，能否在 12GB 上完成一次前向+反向
#   ③ TRL 1.12 的 GRPOTrainer 能否在这个模型上做 1 步更新
#
# 【判定】
#   - SFT 步通过 + GRPO 步通过  -> 走 Qwen3.5-4B-Base 全流程
#   - SFT 过、GRPO 不过          -> 训练核心仍可 SFT，再针对 GRPO 适配
#   - 两者都因架构不兼容失败     -> 自动切换 Qwen3-4B-Base（configs 已留切换口）
#
# 【用法】
#   python scripts/smoke_test.py                  # sft + grpo 都跑
#   python scripts/smoke_test.py --stage sft      # 只跑 SFT 步
#   python scripts/smoke_test.py --stage grpo     # 只跑 GRPO 步
# =====================================================================

from __future__ import annotations

import argparse
import gc
import time
from typing import Optional

import torch

from _bootstrap import logger

# 模型加载相关（推迟到 setup 后 import，保证镜像环境变量生效）
from transformers import AutoModelForImageTextToText, AutoTokenizer


# ---------------------------------------------------------------
# 配置（冒烟用小尺寸，正式训练请用 configs/*.yaml）
# ---------------------------------------------------------------
SMOKE_PROMPTS = [
    "Solve the math: 15 * 17 + 3 = ? Answer only the final number.",
    "What is 7 times 8? Answer only the final number.",
]
MAX_NEW_TOKENS = 32       # 冒烟只生成 32 token，验证"能采样"即可
MAX_LEN = 128             # 训练序列截断，冒烟不必长


# ---------------------------------------------------------------
# 1. 加载模型（bf16 + 冻结视觉塔）
# ---------------------------------------------------------------
def load_model_and_tokenizer(model_path: str):
    """按多模态正确的姿势加载模型。

    关键点：
      - 用 AutoModelForImageTextToText 而不是 AutoModelForCausalLM
        （后者会因参数命名空间不匹配而报错，见 TRL issue #6028）
      - torch_dtype=bf16：社区反馈 Qwen3.5 用 4bit QLoRA 量化方差大，
        这里冒烟先试 bf16 + LoRA；若显存不够，smoke 会告诉我们
      - 之后冻结 vision/visual 参数（纯文本任务不需要视觉分支）
    """
    logger.info(f"加载模型: {model_path}  (dtype=bf16)")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,      # transformers 5.x 原生支持 qwen3_5
        attn_implementation="sdpa",   # 用 SDPA 省显存（比 eager 快且省）
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 补上 pad token：Qwen 生成/批处理需要
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"加载完成，用时 {time.time()-t0:.1f}s，显存 "
                f"{torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ---- 冻结视觉塔 + 统计可训练参数 ----
    frozen = 0
    for name, p in model.named_parameters():
        if "visual" in name.lower() or "vision" in name.lower():
            p.requires_grad_(False)
            frozen += p.numel()
    logger.info(f"已冻结视觉参数: {frozen/1e6:.0f}M (不参与训练)")
    return model, tokenizer


# ---------------------------------------------------------------
# 2. 挂 LoRA（只训文本注意力）
# ---------------------------------------------------------------
def attach_lora(model):
    """用 peft 挂 LoRA。返回 (model, 是否成功)。

    备注：Qwen3.5 的文本分支参数名形如 language_model.model.layers.N.self_attn.q_proj，
    视觉分支通常是 fused qkv，一般不会误伤；若万一匹配到 visual，下面会再补冻。
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    # 保险：任何仍落在视觉上的 LoRA 适配器也冻结
    for name, p in model.named_parameters():
        if ("visual" in name.lower() or "vision" in name.lower()) and p.requires_grad:
            p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable/1e6:.2f}M / 总量 {total/1e6:.0f}M "
                f"({trainable/total*100:.2f}%)")
    model.train()
    return model


# ---------------------------------------------------------------
# 3. 冒烟 A：手动跑 1 步「文本前向 + 反向」（不依赖 TRL，先验证模型本身）
# ---------------------------------------------------------------
def smoke_sft(model, tokenizer) -> bool:
    """手动构造一条文本样本，用 loss 反传一步。

    返回 True=成功。这一步通过 => 模型加载 + LoRA + 显存三者都 OK。
    """
    logger.info("-" * 60)
    logger.info("[SFT 冒烟] 尝试 1 步文本前向/反向 ...")
    try:
        texts = ["Question: What is 15*17+3?\nAnswer: "]
        enc = tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_LEN,
        ).to("cuda")
        # labels = input_ids：让模型学"续写/预测下一个 token"（纯文本）
        labels = enc["input_ids"].clone()

        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        # 注意：多模态 ForConditionalGeneration 不一定接受 labels，
        # 若此处 TypeError 说明要走 image-text-to-text 专用训练路径
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
        loss = out.loss if hasattr(out, "loss") else out["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"[SFT 冒烟] ✅ 成功。loss={loss.item():.4f}, 峰值显存={mem:.2f}GB")
        return True
    except Exception as e:
        logger.error(f"[SFT 冒烟] ❌ 失败: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------
# 4. 冒烟 B：用真 TRL GRPOTrainer 跑 1 步
# ---------------------------------------------------------------
def smoke_grpo(model, tokenizer) -> bool:
    """构造 2 条 prompt，GRPO 采样一组 -> 奖励 -> 更新 1 步。"""
    logger.info("-" * 60)
    logger.info("[GRPO 冒烟] 尝试 TRL GRPOTrainer 1 步 ...")

    # ---- 一个"玩具版"可验证奖励：completion 里包含数字就算对 ----
    # 正式奖励函数见 run_grpo.py；这里只验证训练回路通不通
    def toy_reward(prompts, completions, **kwargs):
        rewards = []
        for c in completions:
            # 简单判定：包含阿拉伯数字即 +1（纯冒烟，不追求正确性）
            rewards.append(1.0 if any(ch.isdigit() for ch in c) else 0.0)
        return rewards

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    ds = Dataset.from_list([{"prompt": p} for p in SMOKE_PROMPTS])

    # GRPOConfig 冒烟最小化：只看更新回路，2 组 * 2 生成
    # ⚠️ 参数名以内省到的 trl 1.12 签名为准（旧版 max_prompt_length /
    #    disable_dropout_in_model 等已不存在，别照抄旧教程）
    cfg = GRPOConfig(
        output_dir=".smoke_grpo_out",
        max_steps=1,                  # 只走 1 步
        per_device_train_batch_size=2,
        num_generations=2,            # 每个 prompt 采 2 条
        max_completion_length=MAX_NEW_TOKENS,
        learning_rate=1e-6,
        beta=0.04,
        bf16=True,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,   # TRL 新版参数名（旧版是 tokenizer=）
        reward_funcs=[toy_reward],
        args=cfg,
        train_dataset=ds,
    )
    t0 = time.time()
    trainer.train()                   # 由 max_steps=1 控制只跑一步
    mem = torch.cuda.max_memory_allocated() / 1024**3
    logger.info(f"[GRPO 冒烟] ✅ 成功。用时 {time.time()-t0:.1f}s, 峰值显存={mem:.2f}GB")
    return True


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="本地模型路径；默认读 models/registry.json")
    parser.add_argument("--stage", choices=["sft", "grpo", "both"], default="both")
    args = parser.parse_args()

    # 确定模型路径：优先命令行，否则查 registry.json（download_models.py 写入的）
    model_path: Optional[str] = args.model
    if model_path is None:
        import json
        from pathlib import Path
        reg_file = Path("models") / "registry.json"
        if reg_file.exists():
            reg = json.loads(reg_file.read_text(encoding="utf-8"))
            model_path = reg.get("Qwen/Qwen3.5-4B-Base")
    if model_path is None:
        logger.error("找不到模型。请先运行: python scripts/download_models.py")
        return

    model, tokenizer = load_model_and_tokenizer(model_path)
    model = attach_lora(model)

    results = {}
    if args.stage in ("sft", "both"):
        results["sft"] = smoke_sft(model, tokenizer)
    if args.stage in ("grpo", "both"):
        results["grpo"] = smoke_grpo(model, tokenizer)

    # 输出冒烟结论（按"实际跑了哪些 stage"判定，别漏判）
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info(f"冒烟 [{k}] : {'通过 ✅' if v else '失败 ❌'}")

    all_pass = all(results.values()) if results else False
    sft_pass = results.get("sft", False)
    grpo_pass = results.get("grpo", False)
    if all_pass:
        logger.info("结论: 全部通过 -> 可继续 Qwen3.5-4B-Base 正式流程")
    elif sft_pass and not grpo_pass:
        logger.info("结论: SFT 可用、GRPO 需适配 -> 先深挖 GRPO 报错")
    elif sft_pass:
        logger.info("结论: 已跑的部分通过（SFT OK）")
    elif grpo_pass:
        logger.info("结论: 已跑的部分通过（GRPO OK）")
    else:
        logger.info("结论: 架构不兼容 -> 切换 Qwen3-4B-Base 路线")


if __name__ == "__main__":
    # gc + 时间（WDDM 下显存释放需要点时间）
    gc.collect()
    main()
