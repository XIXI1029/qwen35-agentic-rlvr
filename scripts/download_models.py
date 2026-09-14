# -*- coding: utf-8 -*-
# =====================================================================
# download_models.py —— 下载模型到项目 models/ 目录（不落 C 盘）
#
# 用法
#   python scripts/download_models.py                       # 下载默认 4B-Base
#   python scripts/download_models.py --model Qwen/Qwen3.5-9B-Base
#   python scripts/download_models.py --source hf           # 走 hf-mirror（需先设 HF_ENDPOINT）
#
# 说明
#   huggingface.co 在国内被墙，这里默认走 ModelScope（国内最稳）。
#   下载完成后会把 {模型id: 本地目录} 写进 models/registry.json，
#   之后 load 模型统一查这个注册表，避免到处硬编码路径。
# =====================================================================

from __future__ import annotations

import argparse
import json

from _bootstrap import MODELS_DIR, logger  # 会自动配好镜像/缓存环境变量

# ModelScope 的 snapshot_download：按需 import（没装时会给出友好提示）
try:
    from modelscope import snapshot_download as ms_snapshot_download
except ImportError:
    ms_snapshot_download = None

# HuggingFace 的 snapshot_download（走 HF_ENDPOINT 镜像）
try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except ImportError:
    hf_snapshot_download = None

REGISTRY_FILE = MODELS_DIR / "registry.json"


def update_registry(model_id: str, local_dir: str) -> None:
    """把 {模型id -> 本地路径} 写进 registry.json，供其它脚本读取。

    Args:
        model_id:   HF/ModelScope 上的模型 id，如 "Qwen/Qwen3.5-4B-Base"
        local_dir: 下载后模型实际落地的本地目录
    """
    reg = {}
    if REGISTRY_FILE.exists():
        reg = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    reg[model_id] = local_dir
    REGISTRY_FILE.write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"已登记模型路径 -> {REGISTRY_FILE}")


def download(model_id: str, source: str = "modelscope") -> None:
    """下载一个模型并登记路径。

    Args:
        model_id: 模型 id
        source:   "modelscope"（默认）或 "hf"（huggingface 镜像）
    """
    logger.info(f"开始下载: {model_id}  (source={source})  ->  {MODELS_DIR}")

    if source == "modelscope":
        if ms_snapshot_download is None:
            raise RuntimeError("未安装 modelscope，请先: pip install modelscope")
        # cache_dir 指定落地目录（默认在 MODELSCOPE_CACHE，也已在项目内）
        local_dir = ms_snapshot_download(model_id, cache_dir=str(MODELS_DIR))
    else:  # source == "hf"
        if hf_snapshot_download is None:
            raise RuntimeError("未安装 huggingface_hub")
        try:
            # 镜像地址已在 _bootstrap.setup_hf_env() 里设成 hf-mirror，
            # 且已禁用 Xet（HF_HUB_DISABLE_XET=1）走普通 HTTP
            local_dir = hf_snapshot_download(
                model_id, cache_dir=str(MODELS_DIR / "hf")
            )
        except Exception as e:
            # hf 下载失败(如 401/Xet 被墙)则自动回退 modelscope
            logger.warning(f"hf 下载失败({type(e).__name__}: {str(e)[:100]})"
                           f"，自动回退 modelscope ...")
            if ms_snapshot_download is None:
                raise
            local_dir = ms_snapshot_download(model_id, cache_dir=str(MODELS_DIR))

    logger.info(f"下载完成: {local_dir}")
    update_registry(model_id, str(local_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="下载模型到项目 models/ 目录")
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-4B-Base",
        help="模型 id，默认 Qwen/Qwen3.5-4B-Base",
    )
    parser.add_argument(
        "--source", choices=["modelscope", "hf"], default="modelscope",
        help="下载源：modelscope(默认,国内快) 或 hf(hf-mirror)",
    )
    args = parser.parse_args()
    download(args.model, args.source)


if __name__ == "__main__":
    main()
