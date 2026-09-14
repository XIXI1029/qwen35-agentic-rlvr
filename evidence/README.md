# evidence/ —— 结论的原始证据（精简版）

体积刻意压到几 MB，只保留"支撑结论所需的最小集合"：

| 目录/文件 | 内容 |
|-----------|------|
| `results/gsm8k_*.json` | 推理层：GSM8K 各模型评估汇总（Base / SFT / GRPO v1 / GRPO v2 / 9B）|
| `results/humaneval_*.json` | 执行层：HumanEval pass@1 汇总 |
| `results/aime2024_*.json` | AIME2024（**无区分度**，仅留档，不写入结论）|
| `results/bfcl_skill_*.json` | 选择层：BFCL 技能选择评估（Base / start-grpo-v2 / 双臂 / 分层基线）|
| `logs/math_training.log` | 数学 GRPO 训练日志（含 `frac_reward_zero_std` 从 0.5→0 的证据）|
| `logs/code_training.log` | 代码 GRPO 训练日志 |
| `logs/skill_training.log` | 技能选择双臂训练日志（800 步 × 2 臂）|
| `skill_training_curve.png` | 技能 RL 两臂 reward 曲线 |

> 逐题明细（`*.samples.json`）、模型权重、BFCL 原始数据均未入库：
> 权重可由 HF/ModelScope 重新下载；数据处理与评测可由仓库内脚本一键重跑（见 `FINAL_REPORT.md` §6 复现命令）。
