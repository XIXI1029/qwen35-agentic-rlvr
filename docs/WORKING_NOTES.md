> ⚠️ **内部工作笔记**（原仓库根 README）：包含 plan.md 纠错、阶段 checklist、服务器操作步骤、简历话术草稿等**对内**内容，不面向外部读者。
> 对外介绍请看根目录 [`README.md`](../README.md)；正式报告看 [`FINAL_REPORT.md`](../FINAL_REPORT.md)。

# Qwen3.5-4B Agentic RL：小模型对齐大模型（GRPO + RLVR）

> 📄 **定稿文档**：[`FINAL_REPORT.md`](FINAL_REPORT.md)（项目报告）· [`RESUME.md`](RESUME.md)（简历与面试问答）· [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)（详细实验）· [`PROJECT_LOG.md`](PROJECT_LOG.md)（逐日排查）
> 🧩 **三层能力**（同一套流水线）：推理 GSM8K 34%→**50%**（>9B 40%）｜执行 HumanEval 21.9%→**25.0%**｜选择 BFCL exact **0.795**（误加载率 100%→**22.5%**，跨任务正迁移 **+48.3pt**）

> 简历项目 · 基于 TRL 复现 Agentic RL 训练，验证 **小模型（4B）能否通过强化学习逼近大模型（9B）** 的能力。

---

## 1. 一句话卖点（写简历用）

在 **Qwen3.5-4B** 上复现 **GRPO + RLVR**（可验证奖励的强化学习 + 工具调用/推理轨迹），
只用 **12GB 单卡**（4-bit QLoRA）训练出具备 Agent 推理能力的小模型，
并通过 GSM8K / AIME2024 / LiveCodeBench 评估它向 **Qwen3.5-9B** 对齐的程度。

简历上的三个可量化创新点：
1. **小模型对齐大模型**：用 RL 让 4B 逼近 9B（同族模型能力蒸馏的"强化版"）
2. **低门槛可复现**：全程 12GB 显存可跑（4-bit + QLoRA + GRPO）
3. **数据质量工程**：设计"轨迹效率过滤"，用更少的高质量 RL 数据取得更好效果

---

## 2. 核心方法（技术原理，理解后再写代码）

| 名词 | 全称 | 一句话解释 | 在本项目里的位置 |
|------|------|-----------|-----------------|
| **GRPO** | Group Relative Policy Optimization | DeepSeek-R1 提出：同一 prompt 采样一组轨迹，用**组内相对奖励**做策略更新，**不需要 Critic/价值模型**，省一半显存 | `scripts/run_grpo.py`（TRL `GRPOTrainer`）|
| **RLVR** | Reinforcement Learning from Verifiable Rewards | 奖励不由模型/人类打分，而由**客观验证**（答案字符串比对、执行代码跑测试）给出，信号无噪声 | 奖励函数 `reward_fn` |
| **QLoRA** | Quantized Low-Rank Adaptation | 先把权重压成 4-bit 再挂低秩适配器训练，让 12GB 卡跑 4B/9B | `configs/sft_config.yaml` |
| **SFT 冷启动** | Supervised Fine-Tuning | RL 前先用专家轨迹教模型"怎么说话/调用工具"，RL 稳定性的关键前置 | `scripts/run_sft.py` |
| **轨迹效率过滤** | Trajectory Efficiency Filter | 同样是答对的轨迹，工具调用次数更少的更优。用效率得分筛数据 | `scripts/prepare_data.py` |

**训练流水线**（一图流）：

```
Qwen3.5-4B(原始,4bit)
   │  ── SFT(3K 专家轨迹冷启动) ──▶ Qwen3.5-4B-SFT
   │                                      │
   │                            GRPO+RLVR(30K RL 轨迹,
   │                            奖励=可验证答案匹配+效率/格式惩罚)
   │                                      ▼
   │                              Qwen3.5-4B-GRPO  ◀── 本项目核心产出
   ▼
Qwen3.5-9B(原始)  ──(不训练,只评估)──▶  9B 能力基线 = 对齐目标
```

**实测结果（服务器 V100 32GB 复现版，详见 `EXPERIMENT_REPORT.md`）**

GSM8K（test 随机 50 题，贪心）：

| 模型 | 角色 | GSM8K@50 |
|------|------|----------|
| Qwen3.5-4B-Base | 基线1（未训练） | 30.0% (15/50) |
| Qwen3.5-4B-SFT | 基线2（冷启动 SFT） | 34.0% (17/50) |
| 4B-GRPO (v1) | 数学 RLVR | 34.0% (17/50) |
| **4B-GRPO (v2)** | 组大小8+部分分奖励（2000步）| **50.0% (25/50)** |
| **Qwen3.5-9B-Base** | **对齐目标** | **40.0% (20/50)** |

HumanEval（164 题，pass@1）与 AIME2024（60 题）：

| 模型 | HumanEval | AIME2024 |
|------|-----------|----------|
| 4B-Base | 14.6% (24/164) | 1.7% (1/60) |
| 4B-SFT（数学冷启动） | 21.9% (36/164) | 0% |
| 4B-GRPO-Code (v1) | 21.9% (36/164) | — |
| **4B-GRPO-Code (v2)** | **25.0% (41/164)** ✅ | 0% |
| 9B-Base | — | 0% |

**关键结论（诚实版）**
- ✅ **4B-GRPO 达 9B 的 85%**（v1：34.0% vs 40.0%）；**v2 反超 9B**（50.0% vs 40.0%）
- ✅ **数学冷启动 SFT 迁移到代码有显著增益**：HumanEval 14.6% → **22.0%（+7.3pt，n=164）**
- ✅ **技能选择（BFCL，分层 400 题）**：exact **0.795** vs 检索基线 0.485 / 规则基线 0.590；**误加载率 22.5% vs 检索 100%**；且**跨任务正迁移 +48.3pt**（数学 RL 起点 vs Base 起点）
- ⚠️ v1 的 GRPO 净收益≈0（组大小 4、零方差步占 50%）；**v2 用组大小 8 + 部分分奖励修复后，两任务均转为显著净提升**
- ⚠️ AIME2024 对 4B/9B 都太难（0~1 题），无区分度
- 归因与改进（组大小、部分分奖励、步数/学习率）见 `EXPERIMENT_REPORT.md`

---

## 2.5 简历话术（只写实测，安全版）

**项目**：基于 GRPO+RLVR 的 Qwen3.5-4B Agentic Reasoning 强化学习
- 在 Qwen3.5-4B（多模态 Base，4B 参数）上，用 TRL GRPOTrainer 实现 **GRPO + 可验证奖励（RLVR）**训练，全程 **12GB 单卡**（bf16+LoRA+梯度检查点）即可跑
- 完整流水线：**冷启动 SFT → GRPO 在线强化**；奖励用"答案数字比对"的客观验证，无 Critic/奖励模型
- **结果**：GSM8K 上 4B-GRPO(v2) **50.0%（25/50）**，**超过 9B-Base 的 40.0%（+10pt）**；代码任务 HumanEval 21.9% → **25.0%（+3.1pt）**；技能选择 BFCL exact **0.795**（相似度检索基线 0.485、阈值规则 0.590），**误加载率 22.5% vs 检索 100%**
- 工程细节：多模态模型正确加载（`AutoModelForImageTextToText`+冻结视觉塔+LoRA）、TRL 1.12 新 API 适配、自研轻量评估器与 `verifier` 统一判定、国内镜像/缓存工程化
- 技术栈：PyTorch · TRL(GRPO) · transformers 5 · peft · datasets

> ⚠️ 不要在简历里写：**"GRPO 带来提升"**（实测 SFT→GRPO 净收益≈0，虽逐题有翻转）、AIME/LCB 分数（AIME 无区分度、LCB 未跑）。
> ✅ 可以写：达 9B 的 85%（GSM8K）、数学 SFT 迁移到代码 +7.3pt、两任务 RLVR 流水线 + 受限代码沙箱。

## 3. ⚠️ 事实核查表（重要！plan.md 有错，已修正）

开始写代码前把 plan 里写错的外部资源逐一核对过，务必以本表为准：

| 项目 | plan.md 写的（错） | 真实情况（以本表为准） |
|------|--------------------|------------------------|
| RL 数据集 | `OpenAgentRL/Open-AgentRL` | **`Gen-Verse/Open-AgentRL-30K`**（~30.1K 条真实端到端 agent 轨迹，由 DAPO-Math 17K + MegaScience 3K 可验证科学题 + LeetCode/Skywork-OR1 组成）|
| SFT 数据集 | 同一个数据集的 "sft" 子集 | **独立数据集 `Gen-Verse/Open-AgentRL-SFT-3K`**（不在 30K 数据集里）|
| 开源仓库 | `OpenAgentRL/Open-AgentRL` | **`Gen-Verse/Open-AgentRL`**（对应 DemyAgent 项目，RLAnyTool/AutoTool，ICML 2026）|
| 架构参考 | 按"普通因果语言模型"加载 | Qwen3.5 是**原生多模态** `Qwen3_5ForConditionalGeneration` + linear/full 混合注意力 —— 见 §5 风险 |
| 参考基准 | 论文里 Qwen3.5-4B 强到"对齐后超 9B" | 原 DemyAgent 论文训的是 **Qwen3-4B**（不同代际），效果数字不能照搬 |

网络环境实测（Windows 本机 2026-09-07）：
- ✅ 可达：清华/阿里 pip 镜像、`pypi.org`、`modelscope.cn`、`hf-mirror.com`
- ❌ 被墙：`huggingface.co` 直连
- ⚡ 推论：**pip 走清华/阿里镜像，模型走 ModelScope 或 hf-mirror**

---

## 4. 目录结构（与 plan.md 对齐，补充了实际需要的文件）

```
agentic_rl_4b_project/
├── configs/                     # 所有训练/数据配置（YAML，脚本读取）
│   ├── sft_config.yaml          #   SFT 冷启动配置（SFTConfig + QLoRA）
│   ├── grpo_config.yaml         #   GRPO 训练配置（GRPOConfig + 奖励）
│   └── data_config.yaml         #   数据集 ID、路径、过滤参数
├── data/
│   ├── raw/                     # 下载的原始 HF 数据集（gitignore）
│   └── processed/               # 过滤+格式化后的训练数据（gitignore）
├── models/                      # 模型权重缓存（gitignore）
├── scripts/                     # 全部可执行脚本（每个都配 argparse + 详尽注释）
│   ├── prepare_data.py          #   Phase2 数据准备 + 轨迹效率过滤
│   ├── run_sft.py               #   Phase3 冷启动 SFT
│   ├── run_grpo.py              #   Phase4 GRPO RL 训练（核心）
│   ├── evaluate.py              #   Phase5 单模型评估（GSM8K/AIME/LCB）
│   └── compare_models.py        #   Phase5 多模型对比 + 生成报告
├── outputs/                     # 训练产物与评估结果（gitignore）
│   ├── qwen3.5-4b-sft/
│   ├── qwen3.5-4b-grpo/
│   └── results/
├── notebooks/
│   └── analysis.ipynb           # 结果可视化/训练曲线分析
├── Makefile                     # 一键复现入口
├── requirements.txt             # 依赖清单
└── README.md                    # 本文件
```

---

## 5. 环境准备（先读这一段，坑都踩过了）

### 5.1 硬件确认
已实测：**RTX 4070 12GB + CUDA 13.1 驱动 + Python 3.12.4 (anaconda)** ✅
12GB 显存正好卡在阈值上，所以全程用 **4-bit 量化**：模型 ~4B×0.5GB/B ≈ 2.5GB，
GRPO 一次 batch 同时展开 4 条轨迹的激活才是显存大头，需要保守设 `num_generations`。

### 5.2 安装依赖（用 conda 建独立环境）
```bash
# 1) 创建独立环境（python 3.11，安装走清华镜像，约 2~3 分钟）
conda create -n qwen35_rl python=3.11 pip -y
conda activate qwen35_rl

# 2) 关键：把【所有模型/数据集缓存】从 C 盘指到项目目录（否则默认落 C:\Users\...\.cache）
#    一次性写入该 conda 环境，以后每次 activate 自动生效
conda env config vars set \
  HF_HOME=E:/qwen/agentic_rl_4b_project/.cache/hf \
  TRANSFORMERS_CACHE=E:/qwen/agentic_rl_4b_project/.cache/hf/hub \
  HF_DATASETS_CACHE=E:/qwen/agentic_rl_4b_project/.cache/hf/datasets \
  MODELSCOPE_CACHE=E:/qwen/agentic_rl_4b_project/.cache/modelscope \
  PIP_CACHE_DIR=E:/qwen/agentic_rl_4b_project/.cache/pip \
  HF_ENDPOINT=https://hf-mirror.com

# 3) 安装依赖（必须走国内镜像；实测 pypi.org 直连不稳定）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
# 4) 装完验证 GPU 可用（应输出 True；若 False 说明装成 CPU 版，
#    改成： pip install torch --index-url https://download.pytorch.org/whl/cu128）
python -c "import torch; print(torch.cuda.is_available())"
```
> 注：Windows 下 git-bash 里直接跑脚本时，环境变量需临时带上前缀，例如
> `HF_ENDPOINT=https://hf-mirror.com python scripts/xxx.py`（脚本内部也会统一处理）。

### 5.3 模型下载策略（国内网络）
| 渠道 | 命令/方式 | 适用 |
|------|----------|------|
| **ModelScope**（首选）| `python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', cache_dir='./models')"` | 国内最快最稳 |
| hf-mirror | 设环境变量 `export HF_ENDPOINT=https://hf-mirror.com` 后正常用 `transformers` 下载 | 能直接用 HF 工具链 |
| HF 直连 | ❌ 不可用 | — |

### 5.4 Phase 1 前必须做的技术选型决策（见 §6）

---

## 6. ⚠️ 风险与待决策项（本阶段重点）

核查 Qwen3.5 架构后发现 plan 里"当普通 CausalLM 加载"的假设站不住，Phase 1 前需二选一：

**风险 A：模型架构不兼容 TRL 的标准用法**
Qwen3.5-4B（含 **-Base 版，已核实**）都是 `Qwen3_5ForConditionalGeneration`
（原生多模态，带 vision_config / image·video token），且是
`linear_attention`+`full_attention` 混合注意力、`text_config` 为 `qwen3_5_text`。
TRL 的 `GRPOTrainer` 默认按"纯文本 CausalLM + `lm_head`"计算逐 token logprob，
对这种结构可能需要额外适配（或用 `transformers` 已支持的 VLM 路径），不确定性中等。

**风险 B：数据集是"agent 轨迹"而非纯文本问答**
Open-AgentRL-30K 的轨迹含多轮工具调用，奖励需"执行工具→看结果→比对最终答案"，
比纯 GSM8K 答案字符串比对复杂。需要把 reward_fn 做成真正能跑工具的 RLVR。

> **我的建议**：如果 5.4 决策选"稳"，训练主力可换 **Qwen3-4B / Qwen3-4B-Base**
> （标准 CausalLM，TRL + DemyAgent 官方路线，跑通的确定性最高），
> 9B 目标仍用 Qwen3.5-9B（同一代里更强的 9B，做"小→大对齐"叙事依然成立）。
> 若坚持 Qwen3.5-4B（同代同族对齐，叙事更漂亮），我们要先做一次加载冒烟测试。

---

## 6.5 在服务器跑 9B / 完整基准（本机 12GB 跑不动，用服务器）

把项目（**不必带 `models/` 和 `data/raw/`，脚本会自动下载/重建**）拷到大显存 Linux 服务器：

```bash
# 推荐：直接跑整套 9B+4B 基准（GSM8K@50 + AIME2024），自动出 SUMMARY.md
bash run_9b_server.sh
# 国内服务器走 hf 镜像下载：
#   HF_ENDPOINT=https://hf-mirror.com bash run_9b_server.sh
```

脚本自动能力：
- **模型缺失自动下载**（`model_utils.ensure_model`：设了 `HF_ENDPOINT` 走 hf-mirror，否则 modelscope）
- **加载策略自动降级**（`evaluate.py --strategy auto`：bf16 大显存 → 4bit 12~16G → CPU fp16 兜底）
- **AIME2024 任务**已内置（数据源 `AI-MO/aimo-validation-aime`，自动筛 2024 年、答案归一成整数）
- 若把本机的 `outputs/qwen3.5-4b-{sft,grpo}` 也拷过去，`run_9b_server.sh` 会连它们一起评

单条跑法示例：
```bash
python scripts/evaluate.py --model Qwen/Qwen3.5-9B-Base --task gsm8k   --tag 9B
python scripts/evaluate.py --model Qwen/Qwen3.5-9B-Base --task aime2024 --tag 9B
```
> 跑完把 `outputs/results/*.json` 拷回本机（或直接在服务器）跑 `python scripts/compare_models.py` 就能看到 4B vs 9B 的完整对比表。

## 6.6 在服务器跑【完整流程】（4B 从头训 + 4B/9B 全测）

想让服务器不只测 9B，而是把整条流水线（SFT→GRPO→评估）从头跑一遍：

```bash
bash run_full_pipeline.sh
# 更高规格（更多 SFT 数据 / GRPO 步数，分数通常更高）：
SFT_N=1500 GRPO_MAX_STEPS=300 bash run_full_pipeline.sh
# 国内镜像下载：
HF_ENDPOINT=https://hf-mirror.com bash run_full_pipeline.sh
```
它依次做：SFT 冷启动数据(GSM8K CoT) → **4B 冷启动 SFT** → GRPO 题库 → **4B GRPO+RLVR** →
评估 **4B-Base/SFT/GRPO 与 9B-Base** × {gsm8k@50, aime2024} → `outputs/results/SUMMARY.md`。
规模用环境变量调：`SFT_N` `SFT_EPOCHS` `GRPO_N` `GRPO_MAX_STEPS` `EVAL_N`。
> 4B 和 9B 模型都无需手动下载：脚本首次用到会自动下载（`ensure_model`）。

## 6.7 训练量级怎么选（对照表）

`run_full_pipeline.sh` 用环境变量调规模。GSM8K train 共 7473 题，**SFT 与 GRPO 强约束不相交**（防混淆），所以两者之和 ≤7473。

| 档位 | 目的 | SFT_N | SFT_EPOCHS | GRPO_N | GRPO_MAX_STEPS | 大概 GPU 时长* | 预期效果 |
|------|------|-------|-----------|--------|----------------|----------------|----------|
| **T0 冒烟/验证** | 打通流程（本机已跑） | 600 | 2 | 1000 | 150 | ~1.5h | 已验证：GSM8K 28→32% |
| **T1 推荐服务器** | 平衡质量与时间 | 2500 | 3 | 4973(全余) | 2000 | ~10–20h | GSM8K 应显著更高 |
| **T2 论文量级** | 追求高分 | 3500 | 3 | 3973 | 3000 | ~30h+ | 更接近收敛/上限 |
| **T3 真·扩展** | 换更大题库 | 4000 | 3 | 见下方 dapo | 3000+ | 多日 | 加难题泛化 |

> *时长估算按服务器 ~20–30s/GRPO 步、SFT 若干 min；不同显卡差别大，跑起来看日志估。

**T1 推荐命令**：
```bash
SFT_N=2500 SFT_EPOCHS=3 GRPO_N=4973 GRPO_MAX_STEPS=2000 EVAL_N=50 \
  nohup bash run_full_pipeline.sh > logs/full_pipeline.log 2>&1 &
```
> 断点续跑：中断后重跑同一条命令即可，已完成的 SFT/GRPO/评估自动跳过。

**想更逼近论文（T3，额外增强，非本脚本默认）**：
1. GRPO 组大小调大更稳更准：编辑 `configs/grpo_config.yaml` 把 `num_generations: 4` → `8`（显存够才改；不够就保持 4）
2. 难题泛化：把 Open-AgentRL-30K 里已处理好的 25k 数学池（`data/processed/rl.jsonl`）按需混入 GRPO 题库 —— 注意这些是 AIME/MATH 级难题，reward 稀疏，**建议在 T1/T2 用 GSM8K 跑通后，再用小比例难题（如 10–20%）做第二阶段增强**，不要直接全程用难题。

**为什么不把 SFT_N 拉到 7473（全量）**：留给 GRPO 的量就归零了。SFT 只需"教会输出格式"，用 2500–3500 条已足够；把剩余题留给 RL 在线探索收益更大（这也符合 RLVR 的认知：SFT 少、RL 多）。

## 6.8 v2 改进版配置 + 「数学→代码」顺序跑（针对 GRPO 净收益≈0）

v1 的教训（见 `EXPERIMENT_REPORT.md`）：组太小导致一半的步没有梯度信号、步数少、奖励过于稀疏。v2 针对性改进：

| 改动 | v1 | **v2** | 理由 |
|------|----|--------|------|
| `num_generations`（组大小） | 4 | **8** | 降低"组内全同分→优势=0"的比例 |
| `max_steps` | 600 / 300 | **2000 / 800** | v1 明显欠训练 |
| `learning_rate` | 1e-6 | **2e-6** | 提高更新幅度 |
| `beta`（KL） | 0.04 | **0.02** | 放松约束，让策略走远一点 |
| 奖励 | 纯 0/1 | **正确性 + 格式分 + 长度惩罚**；代码再按"通过/断言失败/语法错"分档 | 制造组内差异，信号更密 |
| 产物目录 | `...-grpo` | **`...-grpo-v2`**（独立，不覆盖 v1）| 可对比 v1/v2 |

**一键顺序跑（先数学、后代码；两个实验串行独占 GPU）**：
```bash
mkdir -p logs
nohup bash run_both_pipeline.sh > logs/both_pipeline.log 2>&1 &
tail -f logs/both_pipeline.log
```
规模可覆盖：`MATH_MAX_STEPS=2000 CODE_MAX_STEPS=800 EVAL_N=50 SFT_N=2500 GRPO_N=4973`。

> ⚠️ 建议先在服务器用**小步数**验证显存与速度（组变大后显存更紧）：
> `MATH_MAX_STEPS=30 CODE_MAX_STEPS=30 bash run_both_pipeline.sh`
> 通过后再挂正式长跑（断点续跑：中断后重跑同命令即可，会从 checkpoint 续训）。

新文件：`configs/grpo_config.v2.yaml`、`code_rl/code_config.v2.yaml`、`run_both_pipeline.sh`。

## 7. 一键复现

> Windows 若无 `make`，直接用 README 里的 `python scripts/xxx.py` 命令等价执行。
> 阶段顺序有依赖：data → sft → grpo → evaluate → compare。

```bash
make setup     # 装依赖 + 下载模型 + 下载/过滤数据
make data      # 只重跑数据准备
make sft       # 冷启动 SFT（约数小时）
make grpo      # GRPO 训练（最耗时）
make evaluate  # 跑三个模型在各 benchmark 上的分数
make compare   # 汇总对比表 + 可视化
make serve     # 部署 OpenAI 兼容 API
```

---

## 8. 阶段任务清单（每阶段完成 → review → 再进下一阶段）

- [x] **Phase 0** 环境 + 脚手架
- [x] **Phase 1** 模型加载冒烟 + 选型（Qwen3.5-4B-Base 路线确定）+ 4B 基线
- [x] **Phase 2** 数据下载、schema 探查、质量过滤、格式转换
- [x] **Phase 3** 冷启动 SFT（GSM8K CoT 300 条 ×1epoch）
- [x] **Phase 4** GRPO + RLVR 训练（80 步），GSM8K 32.0%
- [~] **Phase 5** 对比表已出（GSM8K）；AIME/LCB 未跑；**9B 参考分受环境阻塞**（详见上）
- [~] **Phase 6** 文档/日志/简历话术已备；`serve.py` 已写好未实测（需装 fastapi/uvicorn）

---

## 9. 参考

- Open-AgentRL / DemyAgent（RLAnyTool, ICML 2026）：<https://github.com/Gen-Verse/Open-AgentRL>
- RL 数据集 30K：<https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K>
- SFT 数据集 3K：<https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-SFT-3K>
- Qwen3.5 模型：<https://huggingface.co/Qwen/Qwen3.5-4B>（或 modelscope `Qwen/Qwen3.5-4B`）
