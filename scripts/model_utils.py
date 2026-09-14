# -*- coding: utf-8 -*-
# =====================================================================
# model_utils.py —— 所有脚本共用的「模型加载」工具
#
# 统一解决三件事：
#   1) 按 modelscope/HF 模型 id 自动查 registry.json，定位本地权重路径
#      （避免每个脚本都硬编码 E:\...\snapshots\master）
#   2) 用正确姿势加载 Qwen3.5（多模态 ForConditionalGeneration）：
#      AutoModelForImageTextToText，而不是会报错的 AutoModelForCausalLM
#   3) 按需 4-bit 量化加载 —— 9B 模型 bf16≈18GB 超 12GB 显存，
#      评估 9B 时用 4-bit 才能塞进单卡（RTX 4070 Super 12GB）
# =====================================================================

from __future__ import annotations

import json
from typing import Optional

import torch
from _bootstrap import MODELS_DIR, logger

# 延迟 import（避免脚本 import 本模块就触发 transformers 初始化）
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers import BitsAndBytesConfig


# ---------------------------------------------------------------
# 0. GPU 可用性自检（训练脚本启动时调用，给清晰报错而不是含糊崩溃）
# ---------------------------------------------------------------
def require_gpu() -> None:
    """训练/评测需要 GPU。若不可用给出可执行的排查建议。"""
    if not torch.cuda.is_available():
        msg = (
            "CUDA 不可用（服务器 torch 检测到 0 张 GPU）。请按顺序排查：\n"
            "  1) nvidia-smi          # 看驱动是否正常、是否被别的进程占满\n"
            "  2) echo $CUDA_VISIBLE_DEVICES\n"
            "  3) python -c \"import torch;print(torch.__version__, torch.version.cuda)\"\n"
            "  最可能原因：torch 的 CUDA 版本高于驱动支持版本 -> 用与驱动匹配的"
            " torch 重装（见 README 的环境说明）"
        )
        raise RuntimeError(msg)
    logger.info(f"GPU 自检通过: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------
# 1. 注册表：模型 id -> 本地路径
# ---------------------------------------------------------------
def resolve_model_path(model_id: str) -> str:
    """从 models/registry.json 查模型 id 对应的本地目录。

    Args:
        model_id: 如 "Qwen/Qwen3.5-4B-Base"
    Returns:
        本地绝对路径字符串
    Raises:
        RuntimeError: 未下载该模型时给出下载指引
    """
    reg_file = MODELS_DIR / "registry.json"
    if not reg_file.exists():
        raise RuntimeError("models/registry.json 不存在，请先运行 "
                           "python scripts/download_models.py --model " + model_id)
    reg = json.loads(reg_file.read_text(encoding="utf-8"))
    if model_id not in reg:
        raise RuntimeError(f"registry 里没有 {model_id}。可用: {list(reg)}。"
                           f"请先运行 python scripts/download_models.py --model {model_id}")
    return reg[model_id]


# ---------------------------------------------------------------
# 1.5 确保模型可用：本地/registry 都没有就自动下载（服务器场景必需）
# ---------------------------------------------------------------
def ensure_model(model_id_or_path: str) -> str:
    """返回一个本地可加载的模型路径；不存在时自动下载并登记 registry。

    服务器上没有 models/ 时靠这个兜底：
      1) 若已是本地目录 -> 直接返回
      2) 查 registry.json 命中 -> 返回路径
      3) 都没有 -> 调用 download_models.download()（优先 modelscope；
         设置 HF_ENDPOINT=https://hf-mirror.com 时走 hf 镜像）
    """
    import os
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    reg_file = MODELS_DIR / "registry.json"
    if reg_file.exists():
        reg = json.loads(reg_file.read_text(encoding="utf-8"))
        if model_id_or_path in reg:
            return reg[model_id_or_path]
    # 自动下载：有 HF_ENDPOINT(=hf-mirror) 说明想走 hf；否则默认 modelscope
    logger.info(f"本地没有 {model_id_or_path}，开始自动下载 ...")
    from download_models import download
    import os as _os
    source = "hf" if _os.environ.get("HF_ENDPOINT") else "modelscope"
    download(model_id_or_path, source=source)
    return resolve_model_path(model_id_or_path)


# ---------------------------------------------------------------
# 2. 统一加载器
# ---------------------------------------------------------------
def load_qwen35_model(
    model_id_or_path: str,
    load_in_4bit: bool = False,
    cpu: bool = False,
) -> tuple:
    """加载 Qwen3.5（或任何 ImageTextToText）模型 + tokenizer，返回 (model, tokenizer)。

    Args:
        model_id_or_path: 模型 id（自动查 registry）或本地路径
        load_in_4bit:  True 时用 bitsandbytes 4-bit 量化加载
        cpu:           True 时强制纯 CPU（device_map=None）。用于 9B 这类
                       GPU 加载会 segfault 的模型；bf16 下 9B≈18GB 内存风险高，
                       用 float16≈9GB 更稳（CPU 推理慢但能出数）
    Returns:
        (model, tokenizer)

    说明：
      - dtype 用的是 transformers 5.x 新参数（torch_dtype 已弃用，会警告）
      - attn_implementation="sdpa"：省显存且快
    """
    import os
    if os.path.exists(model_id_or_path):
        path = model_id_or_path
        logger.info(f"加载本地模型: {path}")
    else:
        path = resolve_model_path(model_id_or_path)
        logger.info(f"加载模型(registry): {model_id_or_path}")

    kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    if cpu:
        # 纯 CPU：device_map=None 留在 CPU；9B 用 fp16 减半内存
        kwargs["device_map"] = None
        kwargs["dtype"] = torch.float16
        kwargs.pop("attn_implementation")   # CPU 上 eager 最稳，sdpa 无收益
        kwargs["attn_implementation"] = "eager"
        logger.info("纯 CPU + float16 加载（慢但避开 GPU segfault）")
    elif load_in_4bit:
        # 塞进 12GB GPU 的关键：NF4 双量化，参数只剩 ~1/4
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("dtype")  # 4bit 量化时不能同时传 dtype
        logger.info("采用 4-bit 量化加载")
    else:
        logger.info("采用 bf16 全精度加载")

    model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()  # 纯推理场景默认 eval 模式（训练脚本会再 model.train()）
    return model, tokenizer


# ---------------------------------------------------------------
# 3. 统一文本生成（所有评估脚本共用）
# ---------------------------------------------------------------
@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """对单条 prompt 做贪心/采样生成，返回纯文本。

    Qwen3.5 是多模态模型，但这里只喂 input_ids 文本 -> 视觉分支自动跳过，
    冒烟已验证该生成路径可用。
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        # 优先走 chat template（Instruct/chat 模型有这个）
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (ValueError, AttributeError):
        # Base 模型 tokenizer 往往没有 chat_template，退化成直接续写原始文本
        logger.debug("tokenizer 无 chat_template，按纯文本续写处理")
        text = prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # 贪心：评估要确定性（Greedy）
        pad_token_id=tokenizer.eos_token_id,
    )
    # 只取"生成"部分（去掉 prompt），并解码
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


# ---------------------------------------------------------------
# 4. 数值/答案解析工具（GSM8K 等"答案是一个数"的任务共用）
# ---------------------------------------------------------------
def extract_last_number(text: str) -> Optional[str]:
    """从模型输出里提取最后的数字（含小数/负数/逗号容忍）。"""
    import re
    # 去掉千分位逗号再匹配，避免 "1,234" 被拆成 1 和 234
    cleaned = text.replace(",", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    return nums[-1] if nums else None


def normalize_answer(s: Optional[str]) -> Optional[float]:
    """把字符串答案归一化成 float 用于精确比对。"""
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None
