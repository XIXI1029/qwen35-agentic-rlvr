# 简历定稿：Qwen3.5-4B 的 GRPO+RLVR 三层 Agent 能力训练

> 所有数字均为实测（来源见文末"数字溯源"）。三层能力（推理 / 执行 / 技能选择）共用**同一套 TRL GRPO + 可验证奖励流水线**。

---

## 0. 一句话定位（选一个当标题）

- **算法/后训练岗**：`面向 Agent 的 GRPO+RLVR 三层能力训练：推理 / 执行 / 技能选择（单卡可复现）`
- **Agent 工程岗**：`Agent 技能加载策略的 RL 训练（BFCL）+ Agentic RLVR 推理/代码能力训练`

---

## 1. 中文简历（推荐：合并成一条，算法岗）

> **项目：面向 Agent 的 GRPO + RLVR 三层能力训练（推理 / 执行 / 技能选择）**
> - 用**同一套 TRL GRPO + 可验证奖励（RLVR）流水线**，在 Qwen3.5-4B 上完成三层 Agent 能力训练，全程**单卡可复现**（12GB 消费卡 / 32GB V100）
> - **① 推理层**：GSM8K 冷启动 SFT→GRPO，**34.0% → 50.0%（+16.0pt）**，**超过同代 9B-Base 的 40.0%**；定位并修复"组内奖励零方差导致 RL 空转"（占比 0.50），用**组大小 4→8 + 部分分奖励**把零方差步占比降到 **0**，两任务同时转为净提升
> - **② 执行层**：换"沙箱执行单测"奖励，HumanEval **21.9% → 25.0%**；**自研受限代码沙箱**（硬超时 / 进程组 kill / 断网 / rlimit / 临时目录）；并发现数学冷启动 SFT 可跨任务迁移到代码（**+7.3pt**）
> - **③ 选择层**：针对工业界"**技能/工具加载不准**"（工具≥20 时准确率跌至 65–78%，错选占失败 18%），把"何时加载哪个技能"建模为 RLVR 任务；BFCL v3（分层 400 题）exact **0.795** vs 相似度检索 0.485 / 阈值规则 0.590，**误加载率 22.5% vs 检索方案 100%**；并验证**跨任务正迁移 +48.3pt**（数学RL 起点 vs Base 起点）
> - 工程：SFT 与 RL 题库**严格不相交**（防评估污染）、**训练自建题库 / 公开基准评测**（防泄漏）、断点续训、模型自动下载与镜像缓存治理、统一评估与结果汇总
> - **技术栈**：PyTorch · TRL(GRPO) · transformers 5 · peft/LoRA · 自研沙箱 · BFCL · MBPP/HumanEval · GSM8K

**能力矩阵一行版**（附在项目末尾，排版紧凑）：
```
推理 GSM8K 34→50%（>9B 40%）｜执行 HumanEval 21.9→25.0%｜选择 BFCL exact 0.795（误加载 100%→22.5%）｜跨任务正迁移 +48.3pt
```

---

## 2. 中文简历（备选：拆两条，Agent 岗）

> **条目 1（置前）：Agent 技能加载策略的 RL 训练（BFCL）**
> - 针对 Agent"**技能/工具加载不准**"问题，把"何时加载、加载哪个技能"建模为**可验证的选择任务**，用 GRPO 训练选择策略
> - 自建带 **near-miss 干扰项**的技能题库（24 技能/6 域）与"无需技能"样本，保留 **held-out 技能**做泛化评测；**BFCL v3 公开基准**评测（multiple / irrelevance 等 5 类）
> - 设计"**集合匹配 + 拒绝加载**"奖励（正确拒绝与正确选择同等重要），修复现状"相似度检索永不拒绝 → 误加载率 100%"缺陷
> - **结果**：exact **0.795**（检索 0.485 / 规则 0.590）；**误加载率 22.5%**，live_irrelevance 上仅 **10%**；平均只加载 1.00 个技能
> - **技术栈**：TRL(GRPO) · BFCL · transformers 5 · LoRA

> **条目 2：Qwen3.5-4B 的 GRPO + RLVR 推理/代码能力训练**
> - 复现"冷启动 SFT → GRPO + 可验证奖励"流水线：数学（答案比对）与代码（**自研沙箱执行单测**）两类任务
> - GSM8K **34% → 50%**（超过 9B 基线 40%）；HumanEval **21.9% → 25.0%**；数学 SFT 跨任务迁移代码 **+7.3pt**
> - 诊断并修复"组内零方差导致 RL 空转"（占比 0.5 → 0），组大小 4→8 + 部分分奖励
> - 单卡可复现（LoRA + 冻结视觉塔 + 梯度检查点 + 断点续训）；SFT/RL 题库不相交设计

---

## 3. 英文版（合并条）

> **Project: Three-layer Agentic RLVR on Qwen3.5-4B (Reasoning / Code Execution / Skill Selection)**
> - Built one **TRL GRPO + verifiable-reward** pipeline and trained three agent capabilities on a **single GPU** (12GB consumer / 32GB V100)
> - **Reasoning**: GSM8K **34.0% → 50.0%** (+16.0pt), **surpassing the same-generation 9B baseline (40.0%)**; diagnosed "zero-variance groups" (50% of steps had no gradient) and fixed it with **group size 4→8 + partial-credit reward** (zero-variance ratio → 0)
> - **Execution**: swapped the reward to **sandboxed unit-test execution** (custom sandbox: hard timeout, process-group kill, network isolation, rlimits); HumanEval **21.9% → 25.0%**; found math cold-start SFT transfers to code (**+7.3pt**)
> - **Skill selection**: framed "when to load which skill" as an RLVR task targeting the industry pain of **inaccurate tool/skill loading** (accuracy drops to 65–78% with ≥20 tools; wrong-tool errors = 18% of failures); on **BFCL v3** (stratified 400) exact **0.795** vs. similarity-retrieval 0.485 / threshold-rule 0.590, cutting the **false-load rate from 100% to 22.5%**; demonstrated **cross-task positive transfer (+48.3pt)** when starting from the math-RL checkpoint vs. Base
> - Engineering: leakage-free SFT/RL splits, self-built training pool + public-benchmark evaluation, checkpoint resume, unified evaluation harness
> - **Stack**: PyTorch · TRL(GRPO) · transformers 5 · peft/LoRA · custom sandbox · BFCL · MBPP/HumanEval · GSM8K

---

## 4. 面试问答（每层 3 问，答案都在项目里）

**推理层**
1. GRPO 为什么比 PPO 省显存？→ 组内相对优势作为 baseline，**无需 Critic 价值网络**；对应 `num_generations`（组大小）。
2. v1 为什么不涨、v2 为什么涨？→ v1 组大小 4 → **50% 的步组内奖励方差为 0**（优势=0，白跑）；v2 组大小 8 + 部分分奖励 → 零方差步占比 **0**，GSM8K 34→50。
3. 怎么确认不是过拟合评测集？→ SFT 与 RL 题库**严格不相交**（GSM8K train 分段），且逐题翻转分析（+14/-6）。

**执行层**
1. 代码奖励最大工程难点？→ **安全与稳定**：模型生成的死循环代码会卡死训练，必须硬超时 + 进程组 kill + 断网 + 资源限制。
2. 为什么 HumanEval 提升只有 3pt？→ 训练分布（MBPP）与评测分布（HumanEval）差异 + 代码任务本身更难；如实报告而非刷分。
3. 跨任务迁移怎么发现的？→ 用数学 SFT 模型直接评 HumanEval，比 Base **+7.3pt**，说明"分步推理/结构化输出"是通用能力。

**选择层**
1. 为什么不直接用 BFCL 训练？→ 它是公开测试集，训练会污染评测；我们**训练自建（818 条，含 near-miss 干扰）/ 评测 BFCL**，把它当**泛化测试**。
2. 奖励怎么处理"该拒绝"？→ 空集也是正确答案；**该拒绝却加载 / 该加载却拒绝都判 0**，并对多加载扣分。
3. 为什么"起点"这么关键？→ 只做数学 RL 并不会自动提升选择能力（0.160 < Base 0.207），但它作为**起点**让技能 RL 学得快得多（Δ +63.5pt vs +15.2pt）→ **跨任务正迁移**。

**通用**
- 12GB 怎么训 4B？→ LoRA（0.07% 参数）+ 冻结视觉塔 + 梯度检查点 + batch/累积调优；大显存版批量并发。
- 工程上最大的教训？→ `set -e` 下命令替换失败会**静默杀脚本**（我们踩过）；以及 nohup 输出缓冲造成"假死"——都写进了 ERR trap / line-buffering 修复。

---

## 5. 避雷清单（不要写）
- ❌ AIME2024 / LiveCodeBench 成绩（前者 4B/9B 都 0~1 题，无区分度；后者未跑）
- ❌ "v1 GRPO 有提升"（v1 净收益≈0；要写就写 v2）
- ❌ 把 BFCL 说成训练数据；❌ 提"Evo-Memory"（那是已作废的旧计划）
- ⚠️ 标注口径：GSM8K n=50、BFCL 分层 400；如被追问统计力，答"可扩到 GSM8K 全量 1319 / BFCL 全量 2771"

## 6. 数字溯源（被要求复核时直接给）
| 数字 | 来源 |
|------|------|
| GSM8K 30/34/34/50/40 | `outputs/outputs_serverv2/results/gsm8k_{Base,sft,grpo,grpo-v2,9B}_*.json` |
| HumanEval 14.6/21.9/21.9/25.0 | `.../humaneval_{Base,sft,grpo,grpo-v2}_*.json` |
| 零方差 0.5→0 | 训练日志 `frac_reward_zero_std` 字段（数学 v1/v2）|
| 技能选择全表 | `server2/results/bfcl_skill_{Base,start-grpo-v2,skillrl-from-base,skillrl-from-mathrl}_*.json` + 本地 `strat-*` 基线 |
| 技能训练动态/曲线 | `server2/skill.log`、`memoryRL/skill_training_curve.png` |
| 中文能力矩阵 | 见本文件 §1 末 |
