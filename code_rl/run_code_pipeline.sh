#!/usr/bin/env bash
# =====================================================================
# run_code_pipeline.sh —— Code RLVR 实验一键脚本（可断点续跑）
#   对应数学实验的 run_full_pipeline.sh：
#     [1] 构建 MBPP 训练题库 (code_rl/data/mbpp_train.jsonl)
#     [2] 构建 HumanEval 评估题库 (code_rl/data/humaneval.jsonl)
#     [3] Code-GRPO 训练  -> outputs/qwen3.5-4b-grpo-code
#     [4] 评估 4B-Base / SFT起点 / Code-GRPO 的 HumanEval pass@1
#     [5] compare_models.py 汇总 -> outputs/results/SUMMARY.md
#
# 规模变量：CODE_MAX_STEPS(默认300) CODE_CONFIG(默认code_rl/code_config.yaml)
# 用法：
#   bash code_rl/run_code_pipeline.sh
#   CODE_MAX_STEPS=600 nohup bash code_rl/run_code_pipeline.sh > code_rl/logs/code.log 2>&1 &
#
# ⚠️ 安全：沙箱会执行模型生成的代码。请不要用 root 跑长任务；
#    理想情况放进容器/低权限用户。详见 code_rl/README.md
# =====================================================================
set -Eeuo pipefail
cd "$(dirname "$0")/.."          # 回到项目根目录
mkdir -p code_rl/logs code_rl/data outputs/results
export PYTHONUNBUFFERED=1

# 出错时打印行号，避免 set -e 静默退出
trap 'rc=$?; echo "[ERROR] run_code_pipeline.sh 在第 ${LINENO} 行失败 (exit=$rc)" >&2' ERR

PY="${PYTHON:-python}"
CODE_MAX_STEPS="${CODE_MAX_STEPS:-300}"
CODE_CONFIG="${CODE_CONFIG:-code_rl/code_config.yaml}"
CODE_EVAL_N="${CODE_EVAL_N:-164}"
# 模型产物目录（v2 用独立目录，避免覆盖 v1）
CODE_OUT="${CODE_OUT:-outputs/qwen3.5-4b-grpo-code}"
# 结果标签（v2 建议 grpo-v2）
CODE_TAG="${CODE_TAG:-grpo}"

echo "== Code RL 实验: max_steps=$CODE_MAX_STEPS config=$CODE_CONFIG"

# [1][2] 题库（存在即跳过）
if [[ ! -s code_rl/data/mbpp_train.jsonl ]]; then
  echo "==> [1/5] 构建 MBPP 训练题库"
  $PY code_rl/build_code_pool.py --which train
else
  echo "==> [1/5] MBPP 题库已存在，跳过"
fi
if [[ ! -s code_rl/data/humaneval.jsonl ]]; then
  echo "==> [2/5] 构建 HumanEval 评估题库"
  $PY code_rl/build_code_pool.py --which eval
else
  echo "==> [2/5] HumanEval 题库已存在，跳过"
fi

# [3] Code-GRPO 训练：没有成品，或最新 checkpoint 步数 < 目标步数 -> 继续训
ckpt_step() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }
ck=$(ckpt_step "$CODE_OUT"); ck=${ck:-0}
if [[ ! -f "$CODE_OUT/model.safetensors" ]] || [[ "$ck" -lt "$CODE_MAX_STEPS" ]]; then
  echo "==> [3/5] Code-GRPO 训练 (max_steps=$CODE_MAX_STEPS, 现有 checkpoint=$ck) ..."
  $PY code_rl/run_grpo_code.py --config "$CODE_CONFIG" --max-steps "$CODE_MAX_STEPS"
else
  echo "==> [3/5] Code-GRPO 已训到 $ck 步(>=目标)，跳过"
fi

# [4] 评估三个模型（各自检测，缺谁跑谁）
run_eval() {  # $1=model $2=tag
  local newest
  newest=$(ls -t outputs/results/humaneval_${2}_*.json 2>/dev/null | grep -v samples | head -1)
  # 结果不存在，或模型权重比结果新（模型变过）-> 重新评估
  if [[ -z "$newest" ]] || { [[ -f "$1/model.safetensors" ]] && [[ "$1/model.safetensors" -nt "$newest" ]]; }; then
    echo "==> [4/5] 评估 humaneval: $2"
    $PY code_rl/evaluate_code.py --model "$1" --tag "$2" --max-samples "$CODE_EVAL_N"
  else
    echo "==> [4/5] 跳过已有: humaneval/$2"
  fi
}
run_eval Qwen/Qwen3.5-4B-Base     Base
run_eval outputs/qwen3.5-4b-sft   sft
run_eval "$CODE_OUT"              "$CODE_TAG"

# [5] 汇总
echo "==> [5/5] 汇总 -> outputs/results/SUMMARY.md"
$PY scripts/compare_models.py

echo "✅ Code RL 实验完成（可反复运行续跑）。"
