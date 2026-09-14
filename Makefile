# =====================================================================
# Makefile —— 一键复现入口
#
# 【Windows 提示】
#   Git Bash 自带的环境一般【没有】 make。两种办法任选：
#   1) 装一个：`conda install -c conda-forge make`
#   2) 不装 make，直接看 .PHONY 目标下的 python 命令，手动照抄执行。
#
# 【阶段依赖顺序】data -> sft -> grpo -> evaluate -> compare -> serve
#   每个脚本都可用 --help 查看参数，且都从 configs/*.yaml 读配置。
# =====================================================================

# 让 .PHONY 目标是"伪目标"：即使存在同名文件也照常执行
.PHONY: help setup models data sft grpo evaluate compare serve clean

# 默认目标：打印帮助
help:
	@echo "用法: make <target>"
	@echo "  setup      一次性：装依赖 + 下载模型 + 准备数据（会很久）"
	@echo "  models     只下载两个模型（走 ModelScope，需先 pip install modelscope）"
	@echo "  data       只准备数据（下载 30K/3K 并做效率过滤）"
	@echo "  sft        冷启动 SFT 训练"
	@echo "  grpo       GRPO 强化学习训练（核心，最耗时）"
	@echo "  evaluate   逐模型跑 GSM8K/AIME2024/LiveCodeBench"
	@echo "  compare    汇总对比表 + 可视化到 outputs/results/"
	@echo "  serve      用 vLLM(WSL/Linux) 或 transformers 起 OpenAI 兼容 API"
	@echo "  clean      清掉 outputs/（谨慎）"

# ---- 一次性初始化：装依赖（走清华镜像）+ 下载模型 + 准备数据 ----
setup:
	pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
	$(MAKE) models
	$(MAKE) data

# ---- 下载 Qwen3.5-4B / 9B（ModelScope，国内最快）----
# 说明：真正的下载代码写在 scripts 里更合适，这里先给出等价 python 命令
models:
	python -c "from modelscope import snapshot_download; \
		[snapshot_download(m, cache_dir='./models') for m in \
		['Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-9B']]"

# ---- 数据准备：下载 + 效率过滤 + 转 GRPO 格式 ----
data:
	python scripts/prepare_data.py --config configs/data_config.yaml

# ---- 冷启动 SFT（需先有 data/processed/sft.jsonl）----
sft:
	python scripts/run_sft.py --config configs/sft_config.yaml

# ---- GRPO 强化学习（需先有 SFT checkpoint + data/processed/rl.jsonl）----
grpo:
	python scripts/run_grpo.py --config configs/grpo_config.yaml

# ---- 逐个评估 4 个模型 ----
evaluate:
	python scripts/evaluate.py --config configs/data_config.yaml

# ---- 汇总对比 ----
compare:
	python scripts/compare_models.py

# ---- 部署 OpenAI 兼容 API ----
serve:
	python scripts/serve.py --model outputs/qwen3.5-4b-grpo

# ---- 清空产物 ----
clean:
	rm -rf outputs/* logs/*
	@echo "已清理 outputs/ 和 logs/"
