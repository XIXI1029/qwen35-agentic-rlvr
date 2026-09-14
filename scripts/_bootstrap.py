# -*- coding: utf-8 -*-
# =====================================================================
# _bootstrap.py —— 所有脚本共用的「引导模块」
#
# 【为什么需要它】
#   每个脚本开头都要做同样几件事，抽到这里避免重复、也避免漏配：
#     1) 设好国内镜像环境变量（huggingface.co 被墙，统一走 hf-mirror）
#     2) 提供"项目根目录 / 常用路径"（所有缓存/产物都在项目里，不落 C 盘）
#     3) 一个加载 configs/*.yaml 的小工具
#     4) 统一的日志配置
#
# 【用法】任何脚本第一行都写：
#   from _bootstrap import ROOT, load_yaml, logger  # 或 sys.path 导入
#  ⚠️ 注意：必须在 import transformers / datasets 之前先调用 setup_hf_env()，
#     因为镜像地址是通过环境变量读取的，装完才 import 才有效。
# =====================================================================

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------
# 0. 项目根目录 = 本文件上一级（scripts/ 的父目录）
#   用 __file__ 求绝对路径，保证无论从哪个目录执行脚本都能定位项目
# ---------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[1]

# 脚本统一放在 scripts/ 下，把它加进 import 路径（让脚本之间可互相 import）
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------
# 1. 环境变量：镜像 + 缓存全部指向项目内
# ---------------------------------------------------------------
def setup_hf_env() -> None:
    """配置 HuggingFace / ModelScope 相关环境变量。

    核心两点：
      a) HF_ENDPOINT=https://hf-mirror.com
         transformers / datasets 下载时会把 huggingface.co 换成该镜像。
         若用户已经在系统里设了自己的 HF_ENDPOINT，则尊重用户不覆盖。
      b) 各缓存目录默认放进 项目/.cache/ 下（不占用 C 盘）。
         若用户已在 conda 环境里通过 `conda env config vars set` 配好，
         则 os.environ 里已有同名变量，此处 setdefault 不会覆盖。
    """
    # 国内镜像兜底：没设就默认 hf-mirror
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # ⚠️ 关键：huggingface_hub 新版大文件默认走 Xet 专用通道
    # (cas-server.xethub.hf.co)。hf-mirror 无法代理它、也无授权 -> 报 401。
    # 这里全局禁用 Xet，改走普通 HTTP 下载（镜像可正常转发，更稳）。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_XET_DISABLE", "1")

    # 缓存目录：一律放项目内（对齐 README §5.2）
    cache_root = ROOT / ".cache"
    # HF 家族缓存都挂在 HF_HOME 下最省事（tokenizers/hub/datasets 都认它）
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "hf" / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hf" / "hub"))
    # ModelScope（国内下载 Qwen 模型用）
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_root / "modelscope"))
    # pip 缓存
    os.environ.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))

    # 确保目录存在（有些库不自动建目录会报错）
    for k in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "MODELSCOPE_CACHE"):
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# 2. 目录常量（脚本里直接引用，语义清晰）
# ---------------------------------------------------------------
MODELS_DIR = ROOT / "models"            # 模型权重（modelscope snapshot 落这里）
DATA_RAW = ROOT / "data" / "raw"        # 原始数据集缓存
DATA_PROCESSED = ROOT / "data" / "processed"  # 过滤/格式化后的训练数据
OUTPUTS_DIR = ROOT / "outputs"          # 一切训练产物/评估结果
CONFIGS_DIR = ROOT / "configs"          # YAML 配置目录

# 创建必要的目录骨架（幂等，多跑无害）
for _d in (MODELS_DIR, DATA_RAW, DATA_PROCESSED, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# 3. YAML 配置读取
# ---------------------------------------------------------------
def load_yaml(name_or_path: str | Path) -> dict:
    """读取 configs/ 下的 YAML，返回 dict。

    Args:
        name_or_path: 直接传文件名（如 "sft_config.yaml"，自动拼 CONFIGS_DIR），
                      或传完整/相对路径都行。
    Returns:
        dict: 解析后的配置字典。
    """
    p = Path(name_or_path)
    if not p.is_absolute() and CONFIGS_DIR.joinpath(p).exists():
        p = CONFIGS_DIR.joinpath(p)   # 只给文件名时，默认到 configs/ 找
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------
# 4. 日志（所有脚本统一走 logger，输出带时间/级别）
# ---------------------------------------------------------------
def _make_logger() -> logging.Logger:
    # Windows 控制台默认是 GBK 编码，打印中文/emoji 会 UnicodeEncodeError 崩溃。
    # 统一把 stdout/stderr 重配成 UTF-8（errors='replace' 兜底），日志就正常了。
    for _s in (sys.stdout, sys.stderr):
        try:
            # line_buffering=True：nohup/重定向到文件时也逐行刷新，
            # 否则日志会攒到 4KB 才落盘 —— 长任务看起来像"卡住了"
            _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):   # 某些环境不支持 reconfigure，跳过
            pass
    lg = logging.getLogger("agentic_rl")
    if not lg.handlers:                 # 防止重复加 handler
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        lg.addHandler(sh)
        lg.setLevel(logging.INFO)
    return lg


logger = _make_logger()

# 模块一被 import 就自动完成环境配置（省得每个脚本都手动调 setup_hf_env）
setup_hf_env()


# ---------------------------------------------------------------
# 5. 便捷工具
# ---------------------------------------------------------------
def maybe_num_workers() -> int:
    """Windows 下 DataLoader 多进程有已知坑（spawn 会重复执行主模块），
    脚本里默认开 0（主进程跑），需要时再手动调高。"""
    return 0
