# -*- coding: utf-8 -*-
# =====================================================================
# prepare_data.py —— 数据准备 + 质量过滤
#
# 数据结构
#   - Open-AgentRL-30K：纯题库。每行 = 一道可验证题
#       prompt[0].content = 题目；reward_model.ground_truth = 标准答案；
#       style = 判定规则(如 rule-lighteval/MATH_v2)；data_source = 来源；
#       ability = 类别（MATH / code ...）
#     注意：数据集里没有现成轨迹——RL 轨迹要靠 GRPO 训练时模型在线生成。
#   - Open-AgentRL-SFT-3K：messages+tools 的多轮对话（含工具调用轨迹），
#     用于 SFT 冷启动，教模型"规范调用工具、按格式给最终答案"。
#
# 质量工程设计（要讲得清）
#   因为轨迹是"在线"的，plan 里那种"离线轨迹效率过滤"在本数据上不成立，
#   数据质量集中在三个环节：
#     ① 可验证性过滤：只留 ground_truth 清晰、判定规则可达的题
#        （代码题需沙箱执行，先单独归档，不混进纯数学 RLVR 池）
#     ② 去重：同一道题在不同来源重复出现时只留一条（hash 判重）
#     ③ 规范化：统一 prompt/答案文本格式；答案按 style 预解析成可比较对象
#   而"轨迹是否低效"改为在奖励函数里做（长度/步数惩罚）——这才符合 RLVR 逻辑。
#
# 用法
#   python scripts/prepare_data.py                        # 按 data_config.yaml 全量
#   python scripts/prepare_data.py --subset math          # 只留数学(默认)
#   python scripts/prepare_data.py --subset all           # 连代码题一起(仅归档)
# =====================================================================

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import DATA_RAW, DATA_PROCESSED, load_yaml, logger


# ---------------------------------------------------------------
# 1. 工具函数
# ---------------------------------------------------------------
def _norm(s: str) -> str:
    """文本规范化：压空白、统一小写（用于判重/比较）。"""
    return " ".join(s.split()).lower()


def prompt_hash(user_content: str) -> str:
    """题目文本的 hash，用于跨来源判重。"""
    return hashlib.md5(_norm(user_content).encode("utf-8")).hexdigest()


def _style_verifiable(style: str) -> bool:
    """判断该题答案是否可"字符串级"验证（math 规则可，代码规则不可）。

    实际上 style 字符串包含判定后端，这里用保守规则：
      含 code/exec/unit 关键字 -> 需要执行环境，单独归档
    """
    s = (style or "").lower()
    if any(k in s for k in ("code", "exec", "unit", "leetcode")):
        return False
    return True


def _is_code_like(data_source: str, ability: str) -> bool:
    """粗略判断某题是否代码题（需要执行验证而非答案匹配）。"""
    blob = f"{data_source} {ability}".lower()
    return any(k in blob for k in ("code", "leetcode", "taco", "skywork-code"))


# ---------------------------------------------------------------
# 2. 处理 RL 题库（Open-AgentRL-30K）
# ---------------------------------------------------------------
def process_rl(max_samples, subset) -> list:
    from datasets import load_dataset

    cfg = load_yaml("data_config.yaml")
    ds_id = cfg["datasets"]["rl_dataset_id"]
    logger.info(f"加载 RL 题库: {ds_id}")

    # 用非 streaming：30K 行约几十 MB，一次性载入内存简单直接
    ds = load_dataset(ds_id, split="train", cache_dir=str(DATA_RAW))
    logger.info(f"总题数: {len(ds)}")

    rows = []
    seen = set()            # 判重集合（prompt hash）
    n_code_archived = 0     # 归档的代码题计数

    for i, item in enumerate(ds):
        if max_samples is not None and len(rows) >= max_samples:
            break
        try:
            user_content = item["prompt"][0]["content"]
            gt = item["reward_model"]["ground_truth"]
            style = item["reward_model"].get("style", "")
            data_source = item["data_source"]
            ability = item["ability"]
        except Exception as e:      # 个别行格式脏就跳过，别中断整批
            logger.warning(f"跳过 row[{i}]（解析失败 {e}）")
            continue

        # 判重：同一道题只留一条
        h = prompt_hash(user_content)
        if h in seen:
            continue
        seen.add(h)

        code_like = _is_code_like(data_source, ability)
        row = {
            "index": len(rows),        # 重排后的本地序号
            "prompt": user_content,
            "ground_truth": str(gt),
            "style": style,
            "data_source": data_source,
            "ability": ability,
            "is_code": code_like,
            "verifiable": _style_verifiable(style) and not code_like,
        }

        if code_like:
            n_code_archived += 1       # 代码题不进当前池（当前只训练数学 RLVR）
        if subset == "math" and code_like:
            continue
        rows.append(row)

    logger.info(f"判重后有效题数: {len(rows)}（其中代码题归档 {n_code_archived} 道）")
    return rows


# ---------------------------------------------------------------
# 3. 处理 SFT 对话（Open-AgentRL-SFT-3K）
# ---------------------------------------------------------------
def process_sft(max_samples) -> list:
    from datasets import load_dataset

    cfg = load_yaml("data_config.yaml")
    ds_id = cfg["datasets"]["sft_dataset_id"]
    logger.info(f"加载 SFT 数据: {ds_id}")

    ds = load_dataset(ds_id, split="train", cache_dir=str(DATA_RAW))
    logger.info(f"SFT 条数: {len(ds)}")

    rows = []
    for i, item in enumerate(ds):
        if max_samples is not None and len(rows) >= max_samples:
            break
        # 只保留有内容的消息，去掉空 tool_calls 噪音，供 SFT 阶段格式化用
        messages = [m for m in item["messages"] if m.get("content")]
        tools = item.get("tools") or []
        rows.append({"index": len(rows), "messages": messages, "tools": tools})
    return rows


# ---------------------------------------------------------------
# 4. 主流程：跑两个数据集并把结果写 JSONL
# ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["math", "all"], default="math",
                        help="math=只留可字符串验证的数学题(默认)；all=含代码题归档")
    parser.add_argument("--max-rl", type=int, default=None,
                        help="RL 最多处理多少题（调小先跑通流程）")
    parser.add_argument("--max-sft", type=int, default=None)
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # ---- 3.1 RL 题库 ----
    rl_rows = process_rl(args.max_rl, args.subset)
    rl_path = DATA_PROCESSED / "rl.jsonl"
    with open(rl_path, "w", encoding="utf-8") as f:
        for r in rl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"RL 数学池已写出: {rl_path}  ({len(rl_rows)} 条)")

    # ---- 3.2 SFT 对话 ----
    sft_rows = process_sft(args.max_sft)
    sft_path = DATA_PROCESSED / "sft.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"SFT 已写出: {sft_path}  ({len(sft_rows)} 条)")

    # ---- 3.3 统计存档（给评估/日志用）----
    stats = {
        "rl_total": len(rl_rows),
        "rl_verifiable": sum(1 for r in rl_rows if r["verifiable"]),
        "rl_by_source": {},
        "sft_total": len(sft_rows),
    }
    for r in rl_rows:
        stats["rl_by_source"][r["data_source"]] = \
            stats["rl_by_source"].get(r["data_source"], 0) + 1
    stats_path = DATA_PROCESSED / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    logger.info(f"统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    logger.info("数据准备完成")


if __name__ == "__main__":
    main()
