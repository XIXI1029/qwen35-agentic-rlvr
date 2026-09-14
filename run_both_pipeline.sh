#!/usr/bin/env bash
# =====================================================================
# run_both_pipeline.sh —— 顺序跑两个实验：先数学 RL，再代码 RL
#
# 顺序执行：两个实验都需要独占 GPU，串行运行可避免显存竞争与 OOM。
#
# 覆盖内容（都用 v2 配置：组大小 8 + 部分分奖励 + 更大步数）：
#   [A] 数学：SFT(已有则跳过) -> GRPO v2 训练 -> GSM8K/AIME 评估 -> 汇总
#   [B] 代码：MBPP/HumanEval 题库 -> Code-GRPO v2 训练 -> HumanEval 评估 -> 汇总
#
# 断点续跑：两个子脚本各自幂等——中断后重跑同一条命令，已完成步骤自动跳过，
#           GRPO 会自动从最近 checkpoint 续训。
#
# 可调环境变量：
#   MATH_MAX_STEPS(默认2000)  CODE_MAX_STEPS(默认800)  EVAL_N(默认50)
#   SFT_N(默认2500)  GRPO_N(默认4973)
#   数学配置： configs/grpo_config.v2.yaml        产物 outputs/qwen3.5-4b-grpo-v2
#   代码配置： code_rl/code_config.v2.yaml        产物 outputs/qwen3.5-4b-grpo-code-v2
#
# 用法（服务器，务必挂后台）：
#   mkdir -p logs
#   nohup bash run_both_pipeline.sh > logs/both_pipeline.log 2>&1 &
#   tail -f logs/both_pipeline.log
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY="${PYTHON:-python}"
MATH_MAX_STEPS="${MATH_MAX_STEPS:-2000}"
CODE_MAX_STEPS="${CODE_MAX_STEPS:-800}"
EVAL_N="${EVAL_N:-50}"
SFT_N="${SFT_N:-2500}"
GRPO_N="${GRPO_N:-4973}"

echo "############################################################"
echo "# [A] 数学实验 (GRPO v2)   max_steps=$MATH_MAX_STEPS"
echo "############################################################"
GRPO_CONFIG=configs/grpo_config.v2.yaml \
MATH_OUT=outputs/qwen3.5-4b-grpo-v2 MATH_TAG=grpo-v2 \
GRPO_MAX_STEPS="$MATH_MAX_STEPS" \
SFT_N="$SFT_N" GRPO_N="$GRPO_N" EVAL_N="$EVAL_N" \
  bash run_full_pipeline.sh

echo
echo "############################################################"
echo "# [B] 代码实验 (Code-GRPO v2)   max_steps=$CODE_MAX_STEPS"
echo "############################################################"
CODE_CONFIG=code_rl/code_config.v2.yaml \
CODE_OUT=outputs/qwen3.5-4b-grpo-code-v2 CODE_TAG=grpo-v2 \
CODE_MAX_STEPS="$CODE_MAX_STEPS" \
  bash code_rl/run_code_pipeline.sh

echo
echo "两个实验全部完成。汇总表：outputs/results/SUMMARY.md"
echo "   数学 v2 产物: outputs/qwen3.5-4b-grpo-v2"
echo "   代码 v2 产物: outputs/qwen3.5-4b-grpo-code-v2"
