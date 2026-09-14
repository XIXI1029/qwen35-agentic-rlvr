#!/usr/bin/env bash
# =====================================================================
# run_9b_server.sh —— 在【服务器】上跑 9B 基准（本机 12GB 跑不动的场景）
#
# 用法（在项目根目录）：
#   bash run_9b_server.sh                  # 全流程
#   HF_ENDPOINT=https://hf-mirror.com bash run_9b_server.sh   # 国内服务器走 hf 镜像
#
# 它会做：
#   1) (可选) 检查依赖，缺就装（可注释掉）
#   2) 自动下载本地缺失的 Qwen3.5-4B-Base / 9B-Base（modelscope 或 hf-mirror）
#   3) 评估 9B-Base 与 4B-Base：GSM8K@50 + AIME2024
#   4) 若你把本地训好的 outputs/qwen3.5-4b-{sft,grpo} 一起拷过来，也会一并评估
#   5) compare_models.py 汇总出 outputs/results/SUMMARY.md
#
# 加载策略由 evaluate.py --strategy auto 自动降级：
#   bf16 全上 GPU(大显存) -> 4bit GPU(12~16G) -> CPU fp16(纯兜底)
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"          # 可覆盖：PYTHON=/path/to/env/bin/python bash run_9b_server.sh

echo "==> 使用解释器: $($PY --version 2>&1)"

# ---- 0.(可选) 装依赖：首次在服务器上可打开；之后可注释 ----
# $PY -m pip install -r requirements.txt
# $PY -m pip install modelscope

# ---- 1. 评估目标模型（缺模型会自动下载）----
# 9B 是主角；AIME 题量<50，--max-samples 60 表示"能拿多少拿多少"
echo "==> [1/3] 评估 9B-Base ..."
$PY scripts/evaluate.py --model Qwen/Qwen3.5-9B-Base  --task gsm8k   --max-samples 50 --tag 9B
$PY scripts/evaluate.py --model Qwen/Qwen3.5-9B-Base  --task aime2024 --max-samples 60 --tag 9B

echo "==> [2/3] 评估 4B-Base（对照组）..."
$PY scripts/evaluate.py --model Qwen/Qwen3.5-4B-Base  --task gsm8k   --max-samples 50 --tag Base
$PY scripts/evaluate.py --model Qwen/Qwen3.5-4B-Base  --task aime2024 --max-samples 60 --tag Base

# ---- 3. 若本地的 4B-SFT / 4B-GRPO checkpoint 也拷过来了，一并评估 ----
if [ -d "outputs/qwen3.5-4b-grpo" ]; then
  echo "==> [3/3] 评估 4B-GRPO ..."
  $PY scripts/evaluate.py --model outputs/qwen3.5-4b-grpo --task gsm8k   --max-samples 50 --tag grpo
  $PY scripts/evaluate.py --model outputs/qwen3.5-4b-grpo --task aime2024 --max-samples 60 --tag grpo
fi
if [ -d "outputs/qwen3.5-4b-sft" ]; then
  echo "==> 评估 4B-SFT ..."
  $PY scripts/evaluate.py --model outputs/qwen3.5-4b-sft  --task gsm8k   --max-samples 50 --tag sft
  $PY scripts/evaluate.py --model outputs/qwen3.5-4b-sft  --task aime2024 --max-samples 60 --tag sft
fi

# ---- 汇总对比表 ----
echo "==> 汇总对比表 → outputs/results/SUMMARY.md"
$PY scripts/compare_models.py

echo "✅ 完成。结果看 outputs/results/SUMMARY.md"
