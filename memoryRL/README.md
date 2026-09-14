# Skill-RLVR：让模型学会「何时/如何加载技能」

> 与主项目（数学/代码 RLVR）**共用同一套 GRPO+RLVR 流水线**，这是"能力栈"的第三层。
> 目标：训练一个**技能选择策略**，解决 Agent 逐级下降的 **"技能/工具加载不准"** 问题。

## 为什么做这个（真实痛点）
- 工具数 ≥20 时前沿模型调用准确率从 95–96% 掉到 **65–78%**，失败中 **错选占 18%**
- 研究（*Looking Is Not Picking*）发现模型**常"看到了"正确工具仍选错** → 瓶颈是**决策读出**，不是检索
- 我们的基线也复现了这个现象：**相似度检索（现状做法）永远不会拒绝加载 → 误加载率 100%**

## 任务与奖励（A1：单轮选择）
- **输入**：用户请求 + K 个候选技能（name/desc/args）
- **输出**：要加载的技能名集合（可为空 = 不需要任何技能）
- **奖励**（`skill_verifier.py`）：集合完全一致 1.0；Jaccard 部分分 ×0.5；**"该拒绝却加载"/"该加载却拒绝"= 0**；
  能解析出 JSON +0.05；每多加载一个技能 −0.1（鼓励"少而准"）

## 数据（刻意避免污染）
| 用途 | 来源 | 规模 |
|------|------|------|
| **训练** | 自建技能库（24 技能 / 6 域，域内互为 near-miss；含"无需技能"样本）| 818 条（`data/processed/skill_train.jsonl`）|
| **泛化评测** | held-out 技能（训练中从未出现的 6 个技能）| 60 条（`skill_heldout.jsonl`）|
| **主评测（held-out 公开基准）** | **BFCL v3**（multiple / live_multiple / simple / irrelevance / live_irrelevance）| **2771 题**（`skill_bfcl_eval.jsonl`）|

> BFCL 是公开测试集，**只做评测不做训练**——这是把它当"泛化测试"而不是同分布指标。

## 最终结果（BFCL v3 分层 400 题，五类各 80 —— 同口径可比）

| 策略/模型 | exact | F1 | **误加载率** | 漏加载率 | 平均加载数 |
|-----------|-------|-----|-----------|---------|-----------|
| 全加载 | 0.200 | 0.492 | 100% | 0% | 1.84 |
| 随机 | 0.330 | 0.412 | 100% | 0% | 1.00 |
| **相似度检索（现状做法）** | 0.485 | 0.606 | **100%** | 0% | 1.00 |
| 检索+阈值规则 | 0.590 | 0.658 | 65.0% | 15.4% | 0.77 |
| Base（未 RL） | 0.207 | 0.301 | 89.4% | 10.0% | 2.17 |
| start-grpo-v2（只做数学 RL） | 0.160 | 0.311 | 88.1% | 14.2% | 1.97 |
| Skill-RL（起点=Base） | 0.360 | 0.341 | 65.6% | 12.1% | 1.76 |
| **Skill-RL（起点=数学RL）** | **0.795** | **0.652** | **22.5%** | **6.7%** | **1.00** |

分类别看（该拒绝的两类，最能体现"加载不准"）：`irrelevance` 相似度检索 **0.000** → 规则 0.325 →
**Skill-RL 0.650**；`live_irrelevance` 检索 **0.000** → 规则 0.375 → **Skill-RL 0.900（误加载仅 10%）**。

### 训练动态（两臂各 800 步；曲线图 `memoryRL/skill_training_curve.png`）
| | arm A：起点=Base | arm B：起点=数学RL |
|---|---|---|
| 时长 / 每步 | 5.6h / 25.3s | 6.3h / 28.5s |
| 训练内 reward | 0.492 → 0.984 | 0.332 → 0.962 |
| 零方差步占比 | 0.383 | 0.312 |
| **BFCL held-out exact** | **0.360** | **0.795** |

> ⚠️ **训练奖励 ≠ 泛化能力**：两臂训练奖励都冲到 0.96+，但只有数学RL 起点那一臂泛化到 BFCL（0.795 vs 0.360）。
> 这正说明"训练自建题库 / 评测 BFCL"的分离设计是必要的。

**结论**：技能选择策略 **exact 0.795 > 最佳基线 0.590（+20.5pt）**，并把**误加载率从检索的 100% 降到 22.5%**；
同时验证**跨任务正迁移 +48.3pt**（数学RL 起点 vs Base 起点）。

## 指标定义
`exact_acc`（集合完全匹配，主指标）· `micro P/R/F1`（技能级）·
`false_load_rate`（该拒绝却加载，BFCL irrelevance 类）· `miss_rate`（该加载却拒绝）·
`avg_pred_size`（平均加载技能数，越少越省上下文）· `avg_prompt_chars`（上下文规模代理）

## 与主项目（数学/代码 RLVR）的融合
三层能力共用一套流水线（模型加载/冻结视觉/LoRA、GRPO 配置、断点续训、评估与汇总）：

| 能力层 | 任务 | 可验证奖励 | 已有结果 |
|--------|------|-----------|---------|
| 推理 | GSM8K / AIME | 数字答案比对 | **34%→50%（超 9B 的 40%）** |
| 执行 | HumanEval / MBPP | 沙箱跑单测（`code_rl/code_sandbox.py`）| **21.9%→25.0%** |
| **选择** | **BFCL multiple/irrelevance** | **技能集合匹配 + 拒绝** | **本次** |

**融合点**：① 共享训练底座（零重复建设）② 技能库含代码类技能 → 选中后用 `code_sandbox` 执行验证（A2 闭环）
③ 统一"能力矩阵"叙事；④ 科研点：**用已 RL 的 checkpoint 作为选择策略的起点**，检验"推理 RL 是否正向迁移到选择能力"。

## 双臂迁移实验（回答"推理 RL 是否正向迁移到技能选择"）
`run_skill_pipeline.sh` 默认跑**两臂**（用 `SKILL_ARMS` 控制）：

| 组 | 起点 | 输出目录 | 结果 tag |
|----|------|---------|---------|
| 参照1 | Qwen3.5-4B-Base（未 RL）| — | `Base` |
| 参照2 | `outputs/qwen3.5-4b-grpo-v2`（只做数学 RL）| — | `start-grpo-v2` |
| arm A | Base | `outputs/qwen3.5-4b-grpo-skill-base` | `skillrl-from-base` |
| arm B | 数学 RL 后的模型 | `outputs/qwen3.5-4b-grpo-skill-mathrl` | `skillrl-from-mathrl` |

**怎么读结果**（脚本 P5 会自动算）：
- `Δ_base = skillrl-from-base − Base`：技能 RL 从零起步的增益
- `Δ_mathrl = skillrl-from-mathrl − start-grpo-v2`：**叠加在数学 RL 之上**的技能 RL 增益
- **迁移效应 = Δ_mathrl − Δ_base**：>0 说明"推理 RL"让技能选择**学得更好**（跨任务正迁移）
- 绝对值对比：`skillrl-from-mathrl` vs `skillrl-from-base`

> 成本：每臂 `SKILL_MAX_STEPS=800` 在 V100 上约 5–7 小时；只跑一臂用 `SKILL_ARMS=mathrl`。

## 运行
```bash
# 只跑基线（秒级，无需 GPU）
python memoryRL/evaluate_skill.py --model similarity --tag B3
python memoryRL/evaluate_skill.py --model rule       --tag B4

# 全流程（服务器；建议 nohup 挂后台）
bash memoryRL/run_skill_pipeline.sh
# 或拉长训练：
SKILL_MAX_STEPS=1500 nohup bash memoryRL/run_skill_pipeline.sh > memoryRL/logs/skill.log 2>&1 &
```

## 文件
| 文件 | 作用 |
|------|------|
| `skills_lib.py` | 自建技能库（24 技能 / 6 域 / near-miss / held-out）+ prompt 渲染 |
| `build_skill_pool.py` | 训练池与 held-out 泛化池 |
| `build_bfcl_pool.py` | BFCL v3 下载与转换（评测池）|
| `skill_verifier.py` | 奖励函数（集合匹配 + 部分分 + 拒绝语义）|
| `run_grpo_skill.py` | Skill-GRPO 训练（复用既有零件，可断点续训）|
| `evaluate_skill.py` | 评测 + 离线基线（similarity/rule/load_all/random）|
| `skill_config.yaml` / `run_skill_pipeline.sh` | 配置 / 一键（可断点续跑）|

## 风险与对策
| 风险 | 对策 |
|------|------|
| 任务与技能描述重合 → 退化成文本匹配 | 域内 near-miss + 多措辞模板；held-out 技能泛化测试 |
| BFCL 与训练分布不同 | 明确写"训练自建 / BFCL 评测"，当**泛化测试**报告 |
| 规则基线已经 0.658 | 我们的目标不是刷分，而是**在同等或更高准确率下大幅降低误加载率与上下文** |
