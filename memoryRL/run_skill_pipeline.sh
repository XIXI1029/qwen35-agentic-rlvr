#!/usr/bin/env bash
# =====================================================================
# run_skill_pipeline.sh —— Skill-RLVR 一键流程（可断点续跑）
#
#   [P1] 建题库：自建训练池 + held-out 泛化池 + BFCL 评测池
#   [P2] 离线基线：similarity / rule / load_all / random（无需 GPU）
#   [P3] **双臂对照训练**：
#          arm=base   : 起点 = Qwen3.5-4B-Base            -> outputs/...-skill-base
#          arm=mathrl : 起点 = outputs/qwen3.5-4b-grpo-v2 -> outputs/...-skill-mathrl
#        （用于回答："推理 RL 是否正向迁移到技能选择能力"）
#   [P4] 评测：Base / grpo-v2（两者都是"未做技能RL"的参照）+ 两个 arm 的模型
#   [P5] 汇总：2×2 对照表 + 迁移效应（Δ_mathrl − Δ_base）
#
# 变量：
#   SKILL_ARMS      默认 "base mathrl"（想只跑一臂就设 SKILL_ARMS=mathrl）
#   SKILL_MAX_STEPS 默认 800（每臂；V100 上约 5-7 小时/臂）
#   EVAL_N          评测上限（默认 400，BFCL 共 2771 题）
#
# 用法：
#   bash memoryRL/run_skill_pipeline.sh
#   SKILL_ARMS=mathrl SKILL_MAX_STEPS=1200 nohup bash memoryRL/run_skill_pipeline.sh \
#       > memoryRL/logs/skill.log 2>&1 &
# =====================================================================
set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p memoryRL/logs outputs/results

# 关键：禁用 Python 输出缓冲，nohup 到文件时也能实时看到日志
export PYTHONUNBUFFERED=1

# 出错时打印"第几行"，避免 set -e 静默退出（排查困难）
trap 'rc=$?; echo "[ERROR] run_skill_pipeline.sh 在第 ${LINENO} 行失败 (exit=$rc)" >&2' ERR

PY="${PYTHON:-python}"
SKILL_MAX_STEPS="${SKILL_MAX_STEPS:-800}"
SKILL_CONFIG="${SKILL_CONFIG:-memoryRL/skill_config.yaml}"
SKILL_ARMS="${SKILL_ARMS:-base mathrl}"
EVAL_N="${EVAL_N:-400}"

# 起点 / 输出目录 / 结果标签 的映射
start_of() { case "$1" in
  base)   echo "Qwen/Qwen3.5-4B-Base" ;;
  mathrl) [[ -d outputs/qwen3.5-4b-grpo-v2 ]] && echo "outputs/qwen3.5-4b-grpo-v2" \
            || { echo "Qwen/Qwen3.5-4B-Base"; } ;;   # 兜底：没有就退回 Base
  *) echo "Qwen/Qwen3.5-4B-Base" ;;
esac; }
out_of()  { case "$1" in base) echo "outputs/qwen3.5-4b-grpo-skill-base" ;; mathrl) echo "outputs/qwen3.5-4b-grpo-skill-mathrl" ;; *) echo "outputs/qwen3.5-4b-grpo-skill-$1" ;; esac; }
tag_of()  { case "$1" in base) echo "skillrl-from-base" ;; mathrl) echo "skillrl-from-mathrl" ;; *) echo "skillrl-from-$1" ;; esac; }

echo "== Skill-RLVR: arms='$SKILL_ARMS'  max_steps/arm=$SKILL_MAX_STEPS  eval_N=$EVAL_N"

# ---------- P1 题库 ----------
[[ -s data/processed/skill_train.jsonl ]] || { echo "==> [P1] 自建训练池"; $PY memoryRL/build_skill_pool.py --per-skill 40; }
[[ -s data/processed/skill_bfcl_eval.jsonl ]] || { echo "==> [P1] BFCL 评测池"; $PY memoryRL/build_bfcl_pool.py; }

# ---------- P2 离线基线 ----------
for m in similarity rule load_all random; do
  if compgen -G "outputs/results/bfcl_skill_B$m"*_*.json >/dev/null; then
    echo "==> [P2] 基线 $m 已有，跳过"
  else
    echo "==> [P2] 基线: $m"; $PY memoryRL/evaluate_skill.py --model $m --tag "B$m" --limit "$EVAL_N"
  fi
done

# ---------- P3 双臂训练 ----------
# 注意：这两个函数必须在"找不到目标"时也返回 0，
# 否则 set -e 会因为命令替换失败而静默终止整个脚本（曾注意事项）。
ckpt_step() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1 || true; }
# 权重文件：合并保存=model.safetensors；适配器保存=adapter_model.safetensors
weights_of() {
  for f in "$1/model.safetensors" "$1/adapter_model.safetensors"; do
    [[ -f "$f" ]] && { echo "$f"; return 0; }
  done
  return 0          # 没找到也返回成功（否则 set -e 会杀掉脚本）
}
for arm in $SKILL_ARMS; do
  st="$(start_of "$arm")"; od="$(out_of "$arm")"
  ck=$(ckpt_step "$od"); ck=${ck:-0}
  w="$(weights_of "$od")"
  if [[ -z "$w" ]] || [[ "$ck" -lt "$SKILL_MAX_STEPS" ]]; then
    echo "==> [P3] arm=$arm 起点=$st 输出=$od (目标 $SKILL_MAX_STEPS 步, 现有 ckpt=$ck)"
    $PY memoryRL/run_grpo_skill.py --config "$SKILL_CONFIG" \
        --start "$st" --out-dir "$od" --max-steps "$SKILL_MAX_STEPS"
  else
    echo "==> [P3] arm=$arm 已训到 $ck 步（>=目标），跳过"
  fi
done

# ---------- P4 评测（参照 + 各臂）----------
run_eval() {  # $1=model $2=tag
  local newest w
  newest=$(ls -t outputs/results/bfcl_skill_${2}_*.json 2>/dev/null | head -1 || true)
  w="$(weights_of "$1")"
  if [[ -n "$newest" && -n "$w" && "$w" -ot "$newest" ]]; then
    echo "==> [P4] $2 结果已最新，跳过"; return
  fi
  echo "==> [P4] 评测: $2  ($1)"
  $PY memoryRL/evaluate_skill.py --model "$1" --tag "$2" --limit "$EVAL_N"
}
run_eval Qwen/Qwen3.5-4B-Base       "Base"              # 参照1：未做任何 RL
run_eval outputs/qwen3.5-4b-grpo-v2 "start-grpo-v2"     # 参照2：只做了数学 RL
for arm in $SKILL_ARMS; do
  run_eval "$(out_of "$arm")" "$(tag_of "$arm")"
done

# ---------- P5 汇总 + 迁移效应 ----------
echo "==> [P5] 汇总"
$PY - <<'PYSUM'
import json, glob, os
best = {}
for f in glob.glob("outputs/results/bfcl_skill_*.json"):
    d = json.load(open(f, encoding="utf-8"))
    t = d.get("tag") or d.get("strategy")
    if t not in best or os.path.getmtime(f) > best[t][0]:
        best[t] = (os.path.getmtime(f), d)
order = ["Bload_all","load_all","Brandom","random","Bsimilarity","similarity",
         "Brule","rule","Base","start-grpo-v2",
         "skillrl-from-base","skillrl-from-mathrl"]
rows = [(t, best[t][1]["overall"]) for t in order if t in best]
print(f"\n{'策略/模型':22s} {'exact':>7s} {'F1':>6s} {'误加载':>7s} {'漏加载':>7s} {'平均加载':>8s}")
for t,o in rows:
    print(f"{t:22s} {o['exact_acc']:7.3f} {o['micro_f1']:6.3f} "
          f"{o['false_load_rate']:7.3f} {o['miss_rate']:7.3f} {o['avg_pred_size']:8.2f}")

# ---- 迁移效应：Δ_mathrl vs Δ_base ----
def acc(t): return best[t][1]["overall"]["exact_acc"] if t in best else None
b, m = acc("Base"), acc("start-grpo-v2")
sb, sm = acc("skillrl-from-base"), acc("skillrl-from-mathrl")
print("\n== 迁移效应分析 ==")
if b is not None and m is not None:
    print(f"① 只做数学RL（未做技能RL）：Base {b:.3f} -> grpo-v2 {m:.3f}  (Δ={m-b:+.3f})")
if sb is not None and b is not None:
    print(f"② 起点=Base   做技能RL：{b:.3f} -> {sb:.3f}  (Δ_base={sb-b:+.3f})")
if sm is not None and m is not None:
    print(f"③ 起点=数学RL 做技能RL：{m:.3f} -> {sm:.3f}  (Δ_mathrl={sm-m:+.3f})")
if None not in (sb, sm, b, m):
    print(f"④ 迁移效应 = Δ_mathrl − Δ_base = {(sm-m)-(sb-b):+.3f}"
          f"   （>0 说明『推理RL』让技能选择学得更好）")
    print(f"⑤ 绝对值对比：skillrl-from-mathrl {sm:.3f} vs skillrl-from-base {sb:.3f}")
PYSUM
echo "Skill-RLVR 流程完成（双臂）"
