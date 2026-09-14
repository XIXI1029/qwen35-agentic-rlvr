#!/usr/bin/env bash
# =====================================================================
# run_full_eval.sh —— 全量评测（把结论钉死）
#
# 覆盖：
#   [1] GSM8K 全量 test（1319 题）× {Base, SFT, GRPO-v2, 9B}
#   [2] BFCL v3 全量（2771 题）× {Base, start-grpo-v2, skillrl-from-base, skillrl-from-mathrl}
#   [3] BFCL 离线基线（similarity/rule/load_all/random，开销极小）
#   [4] 汇总打印（GSM8K 与 BFCL 两张表）
#
# 耗时预估（V100，按实测单题速度）
#   GSM8K：约 15-20s/题 → 1319 题 ≈ 5.5-7h/模型 × 4 ≈ 22-28h
#   BFCL ：约 100-150s/80题 → 2771 题 ≈ 6-8h/模型 × 4 ≈ 24-32h
#   → 合计约 2-2.5 天；建议 nohup 挂后台，跑完一段看一段
#   （想省时间：先只跑 GSM8K 那 4 个，或把 9B 换成抽样）
#
# 断点续跑每个模型的每个任务独立判断：结果文件存在且比权重新 → 跳过
#
# 用法：
#   nohup bash run_full_eval.sh > logs/full_eval.log 2>&1 &
#   GSM8K_N=300 bash run_full_eval.sh          # 想快就用 300 题
#   ONLY=gsm8k bash run_full_eval.sh           # 只跑 GSM8K 部分
# =====================================================================
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs/results
export PYTHONUNBUFFERED=1
trap 'rc=$?; echo "[ERROR] run_full_eval.sh 在第 ${LINENO} 行失败 (exit=$rc)" >&2' ERR

PY="${PYTHON:-python}"
GSM8K_N="${GSM8K_N:-1319}"        # 1319 = GSM8K test 全量
BFCL_LIMIT="${BFCL_LIMIT:-}"       # 空 = 全量 2771 题
ONLY="${ONLY:-all}"                # all | gsm8k | bfcl

# 权重文件判断（合并=model.safetensors；LoRA 适配器=adapter_model.safetensors）
weights_of() {
  for f in "$1/model.safetensors" "$1/adapter_model.safetensors"; do
    [[ -f "$f" ]] && { echo "$f"; return 0; }
  done
  return 0
}
# 结果是否需要重跑
need_run() {  # $1=结果前缀(如 gsm8k) $2=tag $3=模型目录(远程ID可空)
  local newest w
  newest=$(ls -t outputs/results/${1}_${2}_*.json 2>/dev/null | head -1 || true)
  [[ -z "$newest" ]] && return 0
  w="$(weights_of "$3")"
  if [[ -n "$w" ]]; then [[ "$w" -nt "$newest" ]]; else return 1; fi
}

echo "== 全量评测开始: GSM8K_N=$GSM8K_N  BFCL_LIMIT=${BFCL_LIMIT:-全量}  ONLY=$ONLY"

# ---------------- [1] GSM8K 全量 ----------------
if [[ "$ONLY" == "all" || "$ONLY" == "gsm8k" ]]; then
  for pair in "Qwen/Qwen3.5-4B-Base:Base" "outputs/qwen3.5-4b-sft:sft" \
              "outputs/qwen3.5-4b-grpo-v2:grpo-v2" "Qwen/Qwen3.5-9B-Base:9B"; do
    m="${pair%%:*}"; t="${pair##*:}"; tag="${t}-full"
    if need_run gsm8k "$tag" "$m"; then
      echo "==> [1] GSM8K 全量: $tag  ($m)"
      $PY scripts/evaluate.py --model "$m" --task gsm8k --max-samples "$GSM8K_N" --tag "$tag"
    else
      echo "==> [1] GSM8K 全量: $tag 已有结果，跳过"
    fi
  done
fi

# ---------------- [2] BFCL 全量 ----------------
if [[ "$ONLY" == "all" || "$ONLY" == "bfcl" ]]; then
  for pair in "Qwen/Qwen3.5-4B-Base:Base" "outputs/qwen3.5-4b-grpo-v2:start-grpo-v2" \
              "outputs/qwen3.5-4b-grpo-skill-base:skillrl-from-base" \
              "outputs/qwen3.5-4b-grpo-skill-mathrl:skillrl-from-mathrl"; do
    m="${pair%%:*}"; t="${pair##*:}"; tag="${t}-full"
    if need_run bfcl_skill "$tag" "$m"; then
      echo "==> [2] BFCL 全量: $tag  ($m)"
      if [[ -n "$BFCL_LIMIT" ]]; then
        $PY memoryRL/evaluate_skill.py --model "$m" --tag "$tag" --limit "$BFCL_LIMIT"
      else
        $PY memoryRL/evaluate_skill.py --model "$m" --tag "$tag"
      fi
    else
      echo "==> [2] BFCL 全量: $tag 已有结果，跳过"
    fi
  done

  # ---------------- [3] BFCL 离线基线（开销极小）----------------
  for b in similarity rule load_all random; do
    if [[ -z "$(ls -t outputs/results/bfcl_skill_full-$b'_'*.json 2>/dev/null | head -1 || true)" ]]; then
      echo "==> [3] BFCL 基线(全量): $b"
      if [[ -n "$BFCL_LIMIT" ]]; then
        $PY memoryRL/evaluate_skill.py --model "$b" --tag "full-$b" --limit "$BFCL_LIMIT"
      else
        $PY memoryRL/evaluate_skill.py --model "$b" --tag "full-$b"
      fi
    fi
  done
fi

# ---------------- [4] 汇总 ----------------
echo "==> [4] 汇总"
$PY - <<'PYSUM'
import json, glob, os
import re as _re
def _tag_of(path, j):
    """tag 优先取 JSON 字段；没有就从文件名 <task>_<tag>_<yyyyMMdd-HHMM>.json 推断。"""
    t = j.get("tag")
    if t:
        return t
    m = _re.match(r"^[^_]+_(.+)_\d{8}-\d{4}\.json$", os.path.basename(path))
    return m.group(1) if m else (j.get("strategy") or os.path.basename(path))

def latest(pat):
    best={}
    for f in glob.glob(pat):
        if ".samples." in f:            # 逐题明细是 list，跳过
            continue
        j=json.load(open(f,encoding="utf-8"))
        if not isinstance(j, dict):     # 双保险
            continue
        t=_tag_of(f, j)
        if t not in best or os.path.getmtime(f)>best[t][0]: best[t]=(os.path.getmtime(f),j)
    return best

print("\n========== GSM8K 全量 ==========")
g={**latest("outputs/outputs_server/results/gsm8k_*.json"),
   **latest("outputs/outputs_serverv2/results/gsm8k_*.json"),
   **latest("outputs/results/gsm8k_*.json")}
order=["Base","Base-full","sft","sft-full","grpo","grpo-full","grpo-v2","grpo-v2-full","9B","9B-full"]
print(f"{'tag':16s} {'n':>6s} {'acc':>8s} {'答对':>7s}")
for t in order:
    if t in g:
        o=g[t][1]; print(f"{t:16s} {o['n']:6d} {o['accuracy']*100:7.2f}% {o['correct']:7d}")

print("\n========== BFCL 全量（技能选择）==========")
b=latest("outputs/results/bfcl_skill_*.json")
border=["full-load_all","full-random","full-similarity","full-rule",
        "Base-full","start-grpo-v2-full","skillrl-from-base-full","skillrl-from-mathrl-full"]
print(f"{'tag':26s} {'n':>6s} {'exact':>7s} {'F1':>6s} {'误加载':>7s} {'漏加载':>7s}")
for t in border:
    if t in b:
        o=b[t][1]["overall"]
        print(f"{t:26s} {o['n']:6d} {o['exact_acc']:7.3f} {o['micro_f1']:6.3f} "
              f"{o['false_load_rate']:7.3f} {o['miss_rate']:7.3f}")
PYSUM
echo "全量评测流程结束（可重复运行续跑）"
