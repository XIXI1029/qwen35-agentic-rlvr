# -*- coding: utf-8 -*-
# =====================================================================
# check_env.py —— 环境自检脚本
#
# 【作用】安装完依赖后跑一次，确认：
#   - Python / torch / CUDA 是否就绪（GPU 能不能用）
#   - 关键库版本（transformers/trl/peft/bitsandbytes/datasets）
#   - 模型缓存目录是否在项目内（而不是 C 盘）
#
# 【运行】
#   python scripts/check_env.py
#
# 预期看到：torch.cuda.is_available() = True；缓存路径含 E:/qwen/...
# =====================================================================

import os
import sys

# 必须先 import _bootstrap（它会自动设好镜像/缓存环境变量），再 import torch
from _bootstrap import logger

# 重要：真正 import 重量级库之前，环境变量必须已经就位
import torch  # noqa: E402


def main() -> None:
    """打印环境体检报告。"""
    logger.info("=" * 60)
    logger.info("环境自检开始")
    logger.info("=" * 60)

    # ---- 1. Python / torch 版本 ----
    logger.info(f"Python       : {sys.version.split()[0]}")
    logger.info(f"PyTorch      : {torch.__version__}")
    logger.info(f"CUDA 可用    : {torch.cuda.is_available()}")

    # ---- 2. GPU 详情 ----
    if torch.cuda.is_available():
        logger.info(f"GPU          : {torch.cuda.get_device_name(0)}")
        # 显存单位换算成 GB（MiB/1024），对照"12GB 预算"是否成立
        total_mib = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        logger.info(f"显存         : ~{total_mib / 1024:.1f} GB")
        logger.info(f"torch CUDA   : {torch.version.cuda}  (与驱动无需完全一致，向后兼容即可)")
    else:
        # CPU 版常见原因：pip 装到了 CPU wheel / 驱动太旧
        logger.warning("!! CUDA 不可用。若在 Windows，多半是装成了 CPU 版 torch。")
        logger.warning("!! 修复：pip install torch --index-url https://download.pytorch.org/whl/cu128")

    # ---- 3. 关键库版本 ----
    logger.info("-" * 60)
    for lib in ("transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate"):
        try:
            m = __import__(lib)
            logger.info(f"{lib:12s}: {m.__version__}")
        except Exception as e:  # 没装 / 装坏
            logger.warning(f"{lib:12s}: 导入失败 -> {e}")

    # ---- 4. 缓存目录落点检查（应指向项目内，不是 C 盘）----
    logger.info("-" * 60)
    for k in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "MODELSCOPE_CACHE"):
        v = os.environ.get(k, "")
        # 简单判断是否落在 C 盘用户目录下（默认坏位置）
        flag = "OK(项目内)" if "E:/qwen" in v.replace("\\", "/") else "注意"
        logger.info(f"{k:22s}: {v}   [{flag}]")

    logger.info("=" * 60)
    ok = torch.cuda.is_available()
    logger.info(f"结论：{'环境就绪 ✅ 可进入 Phase 1' if ok else 'GPU 不可用，按上面提示修复后再继续 ❌'}")


if __name__ == "__main__":
    # ROOT / _bootstrap 已 import，保证导入阶段就配好环境
    main()
