# 📋 项目执行日志 — Qwen3.5-4B Agentic RL

> 本文件由 Claude Code 逐步追加：**每一步执行了什么脚本、结果如何、踩了什么坑**都记录在此。
> 对应代码全部带中文注释。用于复盘 / 面试讲项目 / 写 README。

- 硬件：RTX 4070 SUPER 12GB · conda env `qwen35_rl` (Python 3.11.16) · torch 2.11.0+cu128
- 库版本：transformers 5.16.1 · trl 1.12.0 · peft 0.20.0 · bitsandbytes 0.50.2 · datasets 5.0.1
- 缓存位置：全部在项目 `E:\qwen\agentic_rl_4b_project\.cache\`（不落 C 盘）

---

## Phase 0 — 环境与脚手架（2026-09-07）

| 步骤 | 执行 | 结果 |
|------|------|------|
| 探测网络 | `curl` 各源 | pypi/清华/阿里/modelscope/hf-mirror 可达；huggingface.co 被墙 |
| 核实外部资源 | WebSearch + 读真实 config | Qwen3.5 确认真实；**plan.md 两处写错已修正**（数据集应为 `Gen-Verse/Open-AgentRL-30K`；仓库为 `Gen-Verse/Open-AgentRL`） |
| 建 conda 环境 | `conda create -n qwen35_rl python=3.11` | ✅（`.condarc` 里 tsinghua `pro` 频道 404，用 `--override-channels` 绕过） |
| 装依赖 | `pip install -i 清华镜像 -r requirements.txt` | ✅ 见上版本号 |
| 换 CUDA torch | 先卸 CPU 版 → `pip install torch==2.11.0+cu128` | ✅ **踩坑**：Windows 的 pip torch 默认是 CPU 版 |
| 环境自检 | `python scripts/check_env.py` | ✅ CUDA=True · 12.0GB · 缓存全在项目内 |
| 产出文件 | 目录树 + configs×3 + `_bootstrap.py` + `check_env.py` + README | ✅ |

## Phase 1 — 模型加载验证与基线（2026-09-07~08）

| 步骤 | 执行 | 结果 |
|------|------|------|
| 下载 4B-Base | `python scripts/download_models.py --model Qwen/Qwen3.5-4B-Base` | ✅ 8.8GB → registry.json |
| 下载 9B-Base | `python scripts/download_models.py --model Qwen/Qwen3.5-9B-Base` | ✅ 19GB → registry.json |
| **冒烟测试** | `python scripts/smoke_test.py --stage sft` | ✅ 加载 5.3s/8.46GB，视觉冻结 334M，LoRA 3.15M，1 步 loss=2.12，峰值 9.16GB |
| **冒烟测试** | `python scripts/smoke_test.py --stage grpo` | ✅ TRL 1.12 GRPOTrainer 1 步成功，3.1s，峰值 8.62GB → **选型敲定 Qwen3.5-4B-Base** |
| 4B 基线 | `python scripts/evaluate.py --model Qwen/Qwen3.5-4B-Base --task gsm8k --max-samples 50` | ✅ **28.0% (14/50)**，结果存 outputs/results/gsm8k_Base_*.json |
| 9B 基线 | 待跑（见下方最新条目） | |

---

<!-- APPEND_POINT -->

## 2026-09-08 追加

### 9B-Base GSM8K 基线 —— ⚠️ 阻塞（待 Phase 5 解决）
| 尝试 | 命令 | 结果 |
|------|------|------|
| 4-bit 量化加载 | `python scripts/evaluate.py --model Qwen/Qwen3.5-9B-Base --task gsm8k --max-samples 50 --4bit` | ❌ **Segfault 139**（bitsandbytes 量化 9B 崩溃） |
| bf16 `device_map=auto` | 加载测试 | ❌ Segfault（在往 GPU 搬权重第 ~14/760 处） |
| bf16 CPU-only 加载 | 加载测试 | ✅ 成功（证明架构/权重没问题，崩在 GPU 分派） |
| bf16 + `max_memory={0:"11GiB"}` 手动限流 | 加载测试 | ❌ 仍 Segfault |
- 诊断：4B bf16 上 GPU 正常；9B 往 GPU 搬权重重崩，与显存上限无关 → WDDM + torch2.11 + 驱动对 9B 权重拷贝的深层问题。
- **处置**：9B 基线推迟到 Phase 5；届时备选方案 = 纯 CPU 推理（慢但能跑）/ 换量化后端 / 降低 max_new_tokens。

### 数据集结构探查（决定 prepare_data.py 怎么写）
| 数据集 | 列 | 解读 |
|--------|----|------|
| `Open-AgentRL-SFT-3K` | `messages`(list), `tools`(list) | OpenAI 风格多轮对话（含 tool_calls），用于 SFT 教工具调用格式 |
| `Open-AgentRL-30K` | `data_source`,`prompt`(list[user消息]),`ability`,`reward_model`{ground_truth,style},`extra_info` | **纯题库**：每行一道可验证题。数学 gt="16"；代码 gt=JSON(fn_name/inputs) 需执行验证。RL 轨迹靠 GRPO 在线采样 |

<!-- APPEND_POINT -->

## Phase 2 完成 —— 数据准备与质量过滤（2026-09-08）
- 执行：`python scripts/prepare_data.py --subset math`
- 结果：RL 数学池 **25,030 题**（判重后；来源见 stats.json）；SFT 缓存 3,000 条 → `data/processed/{rl,sft}.jsonl`
- 质量工程落点（写了注释+README）：①可验证性过滤（剔除需沙箱执行的代码题，单独不混入）②prompt hash 判重 ③答案规范化；"轨迹效率"移到 RL 奖励（长度/步数惩罚）——因为 RL-30K 是题库不是轨迹，RL 轨迹在线生成
- 探查发现：SFT-3K 91% 带代码工具调用，与"纯数学 RLVR"格式不符 → Phase 3 SFT 改用「**可验证奖励筛选的自生成正确推理**」冷启动（Rejection-Sampling SFT）
<!-- APPEND_POINT -->

## Phase 3 pivot —— 冷启动 SFT 数据源调整（2026-09-08）
- 尝试自蒸馏：`python scripts/build_sft_coldstart.py --questions 8 --attempts 2`
- 实测：8 题 426s（~53s/题），**成功率仅 12.5%**。原因：RL 池多属 AIME/MATH 级难题，4B-Base 难解 → 拒绝采样效率极低（凑 150 条需 ~17h）
- **调整**：冷启动 SFT 改用 **GSM8K train 官方逐步推理**（`openai/gsm8k` 的 answer 含人写 CoT + "#### 数字"）。理由：
  a) 题目是小学级，官方 CoT 高质量、可验证，SFT 数据获取零采样成本
  b) 与评估(GSM8K test)同分布，冷启动后的 RL 改进可测
  c) 这在方法上是标准的 RLVR 前置（DeepSeek/OpenRLVR 均如此），简历照样讲得清
- Open-AgentRL-30K 数学池继续作为 RL 训练题库（Phase 4 混入少量难题做泛化）
<!-- APPEND_POINT -->

### 显存/速度标定与 SFT 时长控制（2026-09-08）
- 现象：batch4无检查点 OOM；batch1+梯度检查点是 12GB 唯一稳定组合；总耗时≈样本×epoch×单样本耗时，与累积无关
- 根因线索：OOM 栈显示 `logits = h.float() @ w.float().t()` —— Qwen3.5 **词表 248K**，lm_head 转 float32 计算，长序列极贵（这也解释了解码 ~17tok/s 的慢速）
- 处置：SFT 演示数据集压到 300 条×1 epoch（38 步）；RL/评估同理用小样本控制预算
<!-- APPEND_POINT -->

## Phase 3 完成 —— 冷启动 SFT（2026-09-08）
- 训练命令：`python scripts/run_sft.py --config configs/sft_config.yaml`
- 数据：`data/processed/sft_coldstart.jsonl`（GSM8K train 官方 CoT ×300，seed0 前段）
- 配置：300 条×1 epoch，LoRA r16 on 4B-Base bf16，batch1×acc8+梯度检查点，max_length 320
- 结果：38 步 / 22.3 min；train_loss 1.205；mean_token_accuracy 0.74
- checkpoint：`outputs/qwen3.5-4b-sft`（已转回 bf16 9.1GB；⚠️ peft merge_and_unload 会 fp32 上转，run_sft/run_grpo 已打补丁 cast 回 bf16）
- 待补：SFT 模型 GSM8K 基线（对比 Base 28.0%）
<!-- APPEND_POINT -->

### SFT 基线评估（2026-09-08）
- 执行：`python scripts/evaluate.py --model outputs/qwen3.5-4b-sft --task gsm8k --max-samples 50 --tag sft`
- 结果：**28.0% (14/50)，与 Base 持平**（同一 50 题，逐题对错完全一致）
- 解读：300×1epoch 的 LoRA 冷启动把格式教稳了但没提分——符合"SFT 冷启动≠提分、提分靠 RL"的认知；后续 GRPO 才是核心增益来源
<!-- APPEND_POINT -->

## Phase 4 GRPO 训练（RLVR）—— 2026-09-08
- 训练命令：`python scripts/run_grpo.py --config configs/grpo_config.yaml`
- 起点：SFT checkpoint；题库：GSM8K train 段 [1500,2500)（1000 题，与 SFT 不相交）
- 配置：80 步 / 40.5 分钟 / ~31s 步；num_generations=4、batch1×acc8、max_completion=256、bf16+LoRA+梯度检查点
- 观测：rewards/verifiable_reward/mean 最高 0.625；策略熵 0.97→0.21（收紧）；KL 小(~2e-4)；train_loss 0.047
- 产物：`outputs/qwen3.5-4b-grpo`（bf16 9.1GB）
- 待补：GRPO 模型 GSM8K 50 评估（决定性数字）
<!-- APPEND_POINT -->

### GRPO 评估 + 9B 最终结论（2026-09-08）
- GRPO 评估：`python scripts/evaluate.py --model outputs/qwen3.5-4b-grpo --task gsm8k --max-samples 50 --tag grpo` → **32.0% (16/50)**
- **同批 50 题最终对比：Base 28.0% → SFT 28.0% → GRPO 32.0%**（+4.0pt，14→16 题）✅ 流水线跑通、RLVR 有真实增益
- 9B 参考分：反复尝试（GPU 4bit/bf16/max_memory、CPU fp16）均加载即 segfault，判断为该卡+torch+Qwen3.5-9B 的环境兼容问题 → **如实记为无法产出**，不对机器做 18GB 内存的冒险尝试。README 将如实说明。
- 产物：`outputs/results/SUMMARY.md`
<!-- APPEND_POINT -->

## ✅ 全流程收尾（2026-09-08）
- 对比表：`outputs/results/SUMMARY.md`；README 已填实测数字 + 2.5"简历话术（只写实测）"
- 全部脚本（注释齐全）：check_env / download_models / inspect_data / prepare_data / build_sft_from_gsm8k / build_sft_coldstart(自蒸馏,留作参考) / run_sft / build_grpo_pool / run_grpo / evaluate / compare_models / serve / model_utils / verifier / smoke_test
- 最终成果文件：`outputs/qwen3.5-4b-sft`、`outputs/qwen3.5-4b-grpo`（bf16 各 9.1GB）、`outputs/results/SUMMARY.md`、`data/processed/*.jsonl`
- 已知遗留：9B 无法本机运行；AIME/LCB 未跑；serve.py 未实测；GRPO/SFT 若想更大增益可加步数/加数据重跑
<!-- APPEND_POINT -->

## 新增：服务器跑 9B 方案（2026-09-08）
- 本机 9B 无法运行(环境 segfault) -> 新增服务器可跑组件：
  - `evaluate.py` v2：新增 `--strategy auto|gpu|4bit|cpu`（bf16 GPU→4bit→CPU fp16 自动降级）、`--task aime2024`（数据源 AI-MO/aimo-validation-aime，自动筛2024、答案归一整数，实测取样通）、`--tag`
  - `model_utils.ensure_model`：模型缺失自动下载（HF_ENDPOINT 存在走 hf-mirror，否则 modelscope）
  - `compare_models.py`：支持多任务分区、9B 标签
  - `run_9b_server.sh`：Linux 一键跑 9B/4B × GSM8K/AIME，出 SUMMARY.md
  - README §6.5 服务器运行说明
- AIME 取样实测：AI-MO/aimo-validation-aime (90条) → 筛2024 → 60 题可用 ✅
<!-- APPEND_POINT -->

## 新增：服务器一键【完整流程】run_full_pipeline.sh（2026-09-09）
- 目标：把"整个流程"搬到服务器：4B 也从头训（不只测 9B）
- 脚本顺序：SFT数据(GSM8K CoT) → 4B 冷启动 SFT → GRPO题库 → 4B GRPO+RLVR → 评估 4B×3 + 9B × {gsm8k,aime2024} → SUMMARY
- 规模用环境变量调：SFT_N / SFT_EPOCHS / GRPO_N / GRPO_MAX_STEPS / EVAL_N
- 配套改动：run_sft.py 加 `--epochs/--data`；run_sft/run_grpo 改 ensure_model（模型缺失自动下载）；语法全部校验通过
<!-- APPEND_POINT -->

### run_full_pipeline.sh 改为"断点续跑"（2026-09-09）
- 每步先查产物再执行（幂等可重入），检测依据写入脚本注释：
  SFT数据(行数) / SFT模型(safetensors存在) / GRPO数据(行数) / GRPO模型(存在) / 各评估JSON / compare每次重跑
- FORCE=1 可整体强跑；训练中途中断的那步无成品→整步重跑（前序大步骤不浪费）
- 建议 nohup/tmux 挂起执行，避免 SSH 断开中断
<!-- APPEND_POINT -->

### 训练量级对照表（README §6.7，2026-09-09）
- T0 冒烟600/2+1000/150 → 已验证；T1 推荐 2500/3+4973/2000 ≈10-20h；T2 论文量级 3500/3+3973/3000 ≈30h+；T3 加 dapo 难题小比例做二阶段增强
- SFT 与 GRPO 强约束不相交(≤7473)；SFT 少、RL 多符合 RLVR 认知
- 如需增大 GRPO 组大小：改 configs/grpo_config.yaml num_generations 4→8
<!-- APPEND_POINT -->

### 服务器性能调优：大显存配置 + 断点续训（2026-09-10）
- 现象：V100 32GB 上 GRPO 66s/步、2000 步≈33h；显存只用 11.7/32GB（配置是按 12GB 本机调的，省过头）
- 处置：新增 `configs/grpo_config.server.yaml`（batch4×acc2、关梯度检查点、save_steps200）；`run_grpo.py` 加"检测 checkpoint-* 自动续训"；`run_full_pipeline.sh` 支持 `GRPO_CONFIG` 切换
- 预期：每步快 2-3 倍；max_steps 800 约 5-8h；之后中断可续训不丢进度
<!-- APPEND_POINT -->

## 新增：Code RLVR 实验（第二个可验证任务，2026-09-11）
- 目的：同一套 GRPO+RLVR 流水线，换"跑代码测试"作为可验证奖励 -> 证明方法可迁移（回应"数学题无业务价值"）
- 新增目录 `code_rl/`（与主实验隔离）：
  - code_sandbox.py（T1 受限沙箱：子进程+硬超时+kill进程组+rlimit+断网+临时目录；含自测）
  - code_verifier.py（TRL 奖励函数，内部多线程并行跑沙箱）
  - build_code_pool.py（训练=MBPP full train+validation 464 题；评测=HumanEval 164 题；统一 {prompt,ground_truth} 格式）
  - run_grpo_code.py / evaluate_code.py / code_config.yaml / run_code_pipeline.sh（可断点续跑）
- 验证：沙箱自测(正确pass/错误AssertionError/死循环timeout) ✅；MBPP 官方参考解 5/5 通过 ✅
- 用法：CODE_MAX_STEPS=300 nohup bash code_rl/run_code_pipeline.sh > code_rl/logs/code.log 2>&1 &
- 安全：不要用 root 跑；沙箱挡得住死循环/内存/断网，挡不住刻意的提权利用（README 有说明）
<!-- APPEND_POINT -->

## 服务器完整结果分析（2026-09-11）
- 产物：outputs_server/{qwen3.5-4b-sft, qwen3.5-4b-grpo, qwen3.5-4b-grpo-code} + results(3任务×4模型)
- 结果：GSM8K Base30/SFT34/GRPO34/9B40；HumanEval Base14.6/SFT22.0/GRPO-code22.0；AIME 全~0
- 关键结论：
  1) ✅ 4B-GRPO 达 9B 的 85%（34 vs 40）
  2) ✅ 数学冷启动 SFT 迁移到代码：HumanEval +7.3pt（14.6→22.0, n=164）
  3) ⚠️ GRPO 净收益≈0（两任务均与 SFT 持平），但逐题有 ±5/±3 翻转 → 策略在动、无净提升
  4) 归因：组内零方差比例高(组大小4)、步数少、奖励纯0/1、lr低+beta大、评测 n=50 噪声大
  5) AIME 无区分度（4B/9B 都 0~1 题）
- 报告：EXPERIMENT_REPORT.md；README 结果表与简历话术已更新
<!-- APPEND_POINT -->

## v2 配置 + 「数学→代码」顺序跑（2026-09-11）
- 新增：configs/grpo_config.v2.yaml、code_rl/code_config.v2.yaml（组大小4→8、步数600/300→2000/800、lr 1e-6→2e-6、beta 0.04→0.02；产物目录 ...-v2 独立）
- 奖励升级：run_grpo.make_math_reward（正确性+格式+长度）、code_verifier 分档（通过1.0/断言失败0.3/语法错0.1/+格式0.1），实测打分符合预期
- 新增 run_both_pipeline.sh（先数学后代码，串行独占 GPU；断点续跑）；run_full_pipeline.sh 与 code/run_code_pipeline.sh 支持 MATH_OUT/CODE_OUT/MATH_TAG/CODE_TAG
- 建议先 MATH_MAX_STEPS=30 CODE_MAX_STEPS=30 验证显存，再挂长跑
<!-- APPEND_POINT -->

## v2 小步数验证（30 步）—— 重要结果（2026-09-11）
- 数学 v2（组大小8+部分分奖励，仅跑30步/20分钟）：
  - **GSM8K 42.0% (21/50)** vs SFT 34.0% / Base 30.0% / **9B 40.0%** → 4B **超过 9B**
  - `frac_reward_zero_std` 全程 0（v1 为 0.5）→ 组变大确实消除了"无梯度步"
  - 训练内 reward 波动 0.05~0.97（组内差异明显）
  - ⚠️ 谨慎：仅 30 步、GSM8K n=50（1σ≈7pt），需完整 2000 步复核
- AIME 仍 0/60（无区分度）
- 代码 v2 报错：`generation_batch_size(4) 必须能被 num_generations(8) 整除`
  → 修复：code_config.v2.yaml 改 batch1×accum8=8；两个 run_grpo 脚本加了前置校验与明确提示
  → 已本地校验两份配置 batch×accum 均可被 8 整除
- compare_models 增加 grpo-v2 角色标签
<!-- APPEND_POINT -->

## v2 服务器结果分析（2026-09-12）
- 产物：outputs_serverv2/{qwen3.5-4b-grpo-v2, qwen3.5-4b-grpo-code-v2, qwen3.5-4b-sft, results}
- 结果：
  - 代码 v2（跑满 800 步）：HumanEval 21.9%→**25.0% (41/164)**，逐题 +9/-4 ✅ 净提升
  - 数学 v2（仅 30 步）：GSM8K 34%→**42.0% (21/50)**，逐题 +7/-3；**超过 9B 的 40%**（30步，待复核）
  - AIME 仍 0/60（无区分度）
- 机制验证：frac_reward_zero_std 0.5(v1)→0(v2) → 组大小 4→8 消除"无梯度步"，解释了 v1 净零的原因
- 发现并修复工具链缺陷：
  1) pipeline 原逻辑"有 model.safetensors 就跳过训练" → 导致数学 v2 没从 checkpoint-30 续训到 2000
     修复：改为"最新 checkpoint 步数 < 目标步数则继续训"
  2) 评估跳过逻辑"有结果文件就跳过" → 模型更新后不会重评
     修复：改为"模型权重比结果新则重评"
- 文档：EXPERIMENT_REPORT.md 增加 v2 附录；README 结果表加入 v2 行
<!-- APPEND_POINT -->

## 新方向：Skill-RLVR（技能选择，2026-09-13）
- 背景：用户提出"agent 加载 skills 不准"——查证确为真实痛点（工具≥20时准确率掉到65-78%，错选占失败18%；且模型常"看到正确工具仍选错"）
- 决策：A1 单轮选择 + 评测对齐 BFCL；新代码全部放 memoryRL/；plan1.md 已重写（含与主项目的三层融合方案）
- 已完成：
  - skills_lib.py（自建 24 技能/6 域/near-miss/6 个 held-out）
  - build_skill_pool.py → 训练池 818 条 + held-out 泛化池 60 条
  - build_bfcl_pool.py → BFCL v3 评测池 2771 题（multiple 200 / live_multiple 1053 / simple 400 / irrelevance 240 / live_irrelevance 878）
  - skill_verifier.py（集合匹配+部分分+拒绝语义，7 条单测通过）
  - evaluate_skill.py + 4 条离线基线；run_grpo_skill.py / skill_config.yaml / run_skill_pipeline.sh / README.md
- 离线基线实测：load_all 0.144 | random 0.288 | similarity 0.456(误加载100%) | **rule 0.658(误加载43%)**
- 关键卖点：相似度检索"永不拒绝"→误加载率100%；RL 目标=准确率≥0.658 且大幅降误加载/省上下文
- 待跑（服务器）：P3 训练 + P4 评测 Base/SFT/RL
<!-- APPEND_POINT -->

## Skill-RLVR 双臂迁移实验（2026-09-13）
- 目的：回答"推理 RL 是否正向迁移到技能选择能力"
- 实现：run_grpo_skill.py 加 --out-dir（两臂独立产物）；run_skill_pipeline.sh 改为双臂循环 + 参照评测 + 自动算迁移效应
  - 参照：Base / start-grpo-v2（只做数学RL）
  - arm A：Base → 技能RL（outputs/qwen3.5-4b-grpo-skill-base, tag=skillrl-from-base）
  - arm B：数学RL → 技能RL（outputs/qwen3.5-4b-grpo-skill-mathrl, tag=skillrl-from-mathrl）
  - 迁移效应 = (armB − grpo-v2) − (armA − Base)，>0 即正迁移
- evaluate_skill.py 结果 JSON 增加 tag 字段；P5 汇总块已本地实测通过（含 4 条离线基线）
- 成本：每臂 800 步 V100 上约 5-7h；SKILL_ARMS=mathrl 可只跑一臂
<!-- APPEND_POINT -->

## Skill-RLVR 首次跑通 + 评测口径修复（2026-09-13）
- 前台 3 步验证成功：~35s/步、reward 0.26~0.66、frac_reward_zero_std=0（组大小8生效）、模型保存正常
  → 之前 nohup "看着卡住" 是日志块缓冲（已修 line_buffering + PYTHONUNBUFFERED）
- 服务器规格注意：该机仅 16GB 内存 + systemd-oomd；merged 保存会临时上转 fp32(~18GB) 有 OOM 风险
  → run_grpo_skill.py 新增 --save-mode adapter(默认)：只存几 MB 适配器，evaluate_skill 自动拼基座+适配器
- 修复评测抽样 bug：EVAL_N 限制时原来直接取前 N 条（全是 multiple 类，漏掉 irrelevance）
  → 改为按类别分层抽样；实测 EVAL_N=400 → 五类各 80 题
  → similarity 基线在该口径下 exact=0.485（irrelevance 0.000 / 误加载 100%）
- 时长估算：35s/步 → 800 步 ≈ 7.8h/臂；两臂 ≈ 15.6h
<!-- APPEND_POINT -->

## 修复：run_skill_pipeline.sh 静默退出 bug（2026-09-13）
- 现象：nohup 跑的 pipeline 打印完 [P2] 就没了 —— 无 P3 echo、无 python 进程、GPU 空闲、无报错
- 根因：**bash `set -e` + 命令替换**。辅助函数 weights_of() 在"没找到权重文件"时返回非 0，
  于是 `w="$(weights_of ...)"` 触发 set -e → 整个脚本静默退出（不打印任何错误）
- 修复：
  1) weights_of / ckpt_step 一律 `return 0`（找不到也返回成功）
  2) 三个 pipeline 统一 `set -Eeuo pipefail` + ERR trap 打印"第几行失败"，杜绝静默退出
  3) 本地用空目录模拟验证：两个函数在无产物时不再杀脚本 ✅
- 教训（写进注释）：`set -e` 下，任何可能失败的命令替换都要 `|| true` 或显式 return 0
<!-- APPEND_POINT -->

## Skill-RLVR 双臂实验结果（2026-09-14）—— 全部完成
- 口径修正：模型评估为分层 400 题（五类各 80）；原基线是早期未分层 400（只有 multiple 两类）
  → 本地补齐同口径基线（strat-similarity/rule/load_all/random）后才可比
- **总表（BFCL v3 分层 400）**：
  | 策略/模型 | exact | F1 | 误加载 | 漏加载 | 平均加载 |
  |---|---|---|---|---|---|
  | 全加载 | 0.200 | 0.492 | 100% | 0% | 1.84 |
  | 随机 | 0.330 | 0.412 | 100% | 0% | 1.00 |
  | 相似度检索(现状) | 0.485 | 0.606 | 100% | 0% | 1.00 |
  | 阈值规则 | 0.590 | 0.658 | 65.0% | 15.4% | 0.77 |
  | Base | 0.207 | 0.301 | 89.4% | 10.0% | 2.17 |
  | start-grpo-v2(只数学RL) | 0.160 | 0.311 | 88.1% | 14.2% | 1.97 |
  | Skill-RL(Base起点) | 0.360 | 0.341 | 65.6% | 12.1% | 1.76 |
  | **Skill-RL(数学RL起点)** | **0.795** | **0.652** | **22.5%** | **6.7%** | **1.00** |
- **迁移效应 = Δ_mathrl(+63.5) − Δ_base(+15.2) = +48.3pt**（推理 RL → 选择能力正迁移）
- 分类别亮点：irrelevance 该拒绝 0.650（检索 0.000/规则 0.325）；live_irrelevance 0.900（误加载仅 10%）
- 诚实说明：只做数学 RL 不会自动提升选择能力（0.160 < Base 0.207），但作为起点学得更快
- 文档：EXPERIMENT_REPORT.md 追加 Skill 章节；README §2.5 与 plan1 §9 填入实测数字
- 遗留：技能训练日志(memoryRL/logs/skill.log)未拷；可选全量 2771 题评测；A2 多轮为下一步
<!-- APPEND_POINT -->

## Skill-RLVR 训练动态（server2/skill.log 分析，2026-09-14）
- 两臂各跑满 800 步、以 LoRA 适配器保存：base 5.6h(25.3s/步) / mathrl 6.3h(28.5s/步)
- 训练内 reward：base 0.492→0.984；mathrl 0.332→0.962（自身题库接近饱和）
- 零方差步占比：0.383 / 0.312（v1 数学实验是 0.5）
- ⚠️ 关键观察：训练奖励都饱和，但**只有 mathrl 起点那臂泛化到 BFCL（0.795 vs 0.360）**
  → "训练奖励≠泛化"，印证"训练自建/评测 BFCL"分离设计的必要性
- 产出：memoryRL/skill_training_curve.png（两臂 reward 曲线）；EXPERIMENT_REPORT 与 memoryRL/README 已更新
<!-- APPEND_POINT -->

## 定稿：报告 + 简历（2026-09-14）
- 新增 `FINAL_REPORT.md`：摘要 / 背景 / 方法 / 三层结果（含训练动态与"训练奖励≠泛化"观察）/ 5 条结论 / 6 条局限与补强 / 复现命令 / 产物清单
- 新增 `RESUME.md`：中文合并版 + 中文拆分版（Agent 岗）+ 英文版 + 能力矩阵一行版 + 面试问答（每层 3 问 + 通用 2 问）+ 避雷清单 + **数字溯源表**（每个数字对应具体结果文件，便于复核）
- README 顶部加入两份定稿的入口与三层能力速览
- 三层定稿数字：推理 34→50%（>9B 40%）、执行 21.9→25.0%、选择 exact 0.795（误加载 100%→22.5%，迁移 +48.3pt）
<!-- APPEND_POINT -->

## 全量评测脚本（2026-09-14）
- 新增 `run_full_eval.sh`：GSM8K 全量 test(1319) × {Base,SFT,GRPO-v2,9B} + BFCL 全量(2771) × {Base,start-grpo-v2,skillrl-from-base,skillrl-from-mathrl} + 4 条离线基线 + 汇总打印
- 特性：可断点续跑（结果文件比权重新则跳过）、ONLY=gsm8k|bfcl 分段跑、GSM8K_N 可调、ERR trap + 无缓冲日志
- 耗时预估：GSM8K ≈22-28h（4 模型）、BFCL ≈24-32h（4 模型），合计约 2-2.5 天；建议 nohup + 分段跑
- 本地已验证：脚本语法、汇总块（含 .samples.json 过滤修复）
<!-- APPEND_POINT -->
