#!/usr/bin/env bash
# =====================================================================
# run_full_pipeline.sh —— 服务器完整流程（可断点续跑）
#
# 覆盖：SFT数据 -> 4B 冷启动 SFT -> GRPO 题库 -> 4B GRPO -> 评估 4B×3 + 9B
#       × {gsm8k, aime2024} -> compare 汇总 SUMMARY.md
#
# 断点续跑原理
#   每步开工前检查"产物是否完整"，完整就跳过、只跑没做完的：
#     阶段            检测依据(存在即视为完成)
#     SFT 数据        data/processed/sft_coldstart.jsonl 行数 == SFT_N
#     4B SFT 训练     outputs/qwen3.5-4b-sft/model.safetensors
#     GRPO 数据       data/processed/grpo_train.jsonl 行数 == 期望题数
#     4B GRPO 训练    outputs/qwen3.5-4b-grpo/model.safetensors
#     各评估          outputs/results/<task>_<tag>_*.json 已存在
#     compare         汇总已有结果并生成对比表（开销极小）
#   注意：SFT/GRPO 的 checkpoint 只在训练结束才落盘；如果中断发生在
#      训练中途，该步没有成品会被整步重跑（无法从中间续），但前面已完成的
#      大步不会浪费。
#
# 控制
#   FORCE=1 bash run_full_pipeline.sh    # 忽略断点，全部重跑
#   规模环境变量同前：SFT_N SFT_EPOCHS GRPO_N GRPO_MAX_STEPS EVAL_N
#
# 防中断建议 nohup bash run_full_pipeline.sh > logs/full_pipeline.log 2>&1 &
# =====================================================================
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs/results
export PYTHONUNBUFFERED=1

# 出错时打印行号，避免 set -e 静默退出
trap 'rc=$?; echo "[ERROR] run_full_pipeline.sh 在第 ${LINENO} 行失败 (exit=$rc)" >&2' ERR

PY="${PYTHON:-python}"
SFT_N="${SFT_N:-600}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
GRPO_N="${GRPO_N:-1000}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-150}"
EVAL_N="${EVAL_N:-50}"
FORCE="${FORCE:-0}"
# GRPO 配置：默认本地 12GB 版；服务器大显存用 configs/grpo_config.server.yaml
# v2（组大小8+部分分奖励）用 configs/grpo_config.v2.yaml
GRPO_CONFIG="${GRPO_CONFIG:-configs/grpo_config.yaml}"
# 模型产物目录（v2 用独立目录，避免覆盖旧结果）
MATH_OUT="${MATH_OUT:-outputs/qwen3.5-4b-grpo}"
# 结果标签（v2 建议 grpo-v2，便于与 v1 的 grpo 区分）
MATH_TAG="${MATH_TAG:-grpo}"
GSM8K_TRAIN=7473   # openai/gsm8k train 总量（build_grpo_pool 内部同源）

echo "== 配置: SFT_N=$SFT_N SFT_EPOCHS=$SFT_EPOCHS GRPO_N=$GRPO_N "`
     `"GRPO_MAX_STEPS=$GRPO_MAX_STEPS EVAL_N=$EVAL_N FORCE=$FORCE"

# ---------- 小工具 ----------
line_count() { [[ -f "$1" ]] && wc -l < "$1" 2>/dev/null || echo 0; }
# 最新 checkpoint 的步数（没有则 0）
ckpt_step() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }
# 评估是否需要重跑：结果文件不存在，或比模型权重旧（模型更新过就要重评）
need_eval() {   # $1=task $2=tag $3=模型目录(可为空)
  local newest
  newest=$(ls -t outputs/results/${1}_${2}_*.json 2>/dev/null | grep -v samples | head -1)
  [[ -z "$newest" ]] && return 0
  if [[ -n "$3" && -f "$3/model.safetensors" ]]; then
    [[ "$3/model.safetensors" -nt "$newest" ]]   # 权重更新 -> 需重评
  else
    return 1                                     # 远程模型：有结果即算完成
  fi
}

# ---------- [1/6] SFT 冷启动数据 ----------
expect_sft=$(($(line_count data/processed/sft_coldstart.jsonl)))
if [[ $FORCE -eq 1 ]] || [[ $expect_sft -lt $SFT_N ]]; then
  echo "==> [1/6] 构建 SFT 冷启动数据 (需要 $SFT_N 条, 现有 $expect_sft)"
  $PY scripts/build_sft_from_gsm8k.py --n "$SFT_N"
else
  echo "==> [1/6] SFT 数据已就绪($expect_sft 条)，跳过"
fi

# ---------- [2/6] 4B 冷启动 SFT 训练 ----------
if [[ $FORCE -eq 1 ]] || [[ ! -f outputs/qwen3.5-4b-sft/model.safetensors ]]; then
  echo "==> [2/6] SFT 训练 (epochs=$SFT_EPOCHS) ..."
  $PY scripts/run_sft.py --config configs/sft_config.yaml --epochs "$SFT_EPOCHS"
else
  echo "==> [2/6] SFT 模型已存在，跳过"
fi

# ---------- [3/6] GRPO 题库 ----------
grpo_avail=$(( GSM8K_TRAIN - SFT_N ))          # 排掉 SFT 用过的
[[ $grpo_avail -lt 0 ]] && grpo_avail=0
grpo_expect=$(( GRPO_N < grpo_avail ? GRPO_N : grpo_avail ))
have_grpo=$(line_count data/processed/grpo_train.jsonl)
if [[ $FORCE -eq 1 ]] || [[ $have_grpo -lt $grpo_expect ]]; then
  echo "==> [3/6] 构建 GRPO 题库 (需要 $grpo_expect 条, 现有 $have_grpo)"
  $PY scripts/build_grpo_pool.py --n "$GRPO_N" --sft_seen "$SFT_N"
else
  echo "==> [3/6] GRPO 题库已就绪($have_grpo 条)，跳过"
fi

# ---------- [4/6] 4B GRPO 训练 ----------
# 训练判断：没有成品模型，或最新 checkpoint 步数 < 目标步数(说明没训够) -> 继续训
ck=$(ckpt_step "$MATH_OUT"); ck=${ck:-0}
if [[ $FORCE -eq 1 ]] || [[ ! -f "$MATH_OUT/model.safetensors" ]] || [[ "$ck" -lt "$GRPO_MAX_STEPS" ]]; then
  echo "==> [4/6] GRPO 训练 (config=$GRPO_CONFIG, 目标 max_steps=$GRPO_MAX_STEPS, 现有 checkpoint=$ck) ..."
  $PY scripts/run_grpo.py --config "$GRPO_CONFIG" --max-steps "$GRPO_MAX_STEPS"
else
  echo "==> [4/6] GRPO 已训到 $ck 步(>=目标 $GRPO_MAX_STEPS)，跳过"
fi

# ---------- [5/6] 评估 4 个模型 × 2 任务（各自检测，缺谁跑谁）----------
run_eval() { # $1=model $2=tag $3=task $4=samples
  local model=$1 tag=$2 task=$3 ns=$4
  if [[ $FORCE -eq 0 ]] && ! need_eval "$task" "$tag" "$model"; then
    echo "==> [5/6] 跳过已有评估: $task/$tag"
  else
    echo "==> [5/6] 评估: $tag × $task"
    $PY scripts/evaluate.py --model "$model" --task "$task" \
        --max-samples "$ns" --tag "$tag"
  fi
}

run_eval Qwen/Qwen3.5-4B-Base    Base gsm8k   "$EVAL_N"
run_eval Qwen/Qwen3.5-4B-Base    Base aime2024 60
run_eval outputs/qwen3.5-4b-sft  sft  gsm8k   "$EVAL_N"
run_eval outputs/qwen3.5-4b-sft  sft  aime2024 60
run_eval "$MATH_OUT"             "$MATH_TAG" gsm8k   "$EVAL_N"
run_eval "$MATH_OUT"             "$MATH_TAG" aime2024 60
run_eval Qwen/Qwen3.5-9B-Base    9B   gsm8k   "$EVAL_N"
run_eval Qwen/Qwen3.5-9B-Base    9B   aime2024 60

# ---------- [6/6] 汇总 ----------
echo "==> [6/6] 汇总对比表 → outputs/results/SUMMARY.md"
$PY scripts/compare_models.py

echo "完成（可反复运行续跑/补漏）。结果在 outputs/results/SUMMARY.md"
