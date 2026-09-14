# -*- coding: utf-8 -*-
# =====================================================================
# skills_lib.py —— 自建"技能库"（Skill-RLVR 的训练数据来源）
#
# 为什么自建而不是用 BFCL 训练
#   BFCL 是公开测试集，用它训练会污染评测（官一问就穿帮）。
#   所以：训练 = 自建技能库（本题库）；评测 = BFCL（held-out 公开基准）。
#
# 设计要点
#   1) 技能按"域(domain)"分组，每个域内 4 个技能**互为 near-miss**
#      （例如 sort_by_value / sort_by_key / sort_desc / sort_numeric），
#      这样"选择"才是真难题，而不是靠关键词撞对。
#   2) 每个技能配 3 条不同措辞的用户请求模板（避免退化成文本匹配）。
#   3) 每个域留 1 个技能为 **held-out**（训练时不出现，只在泛化评测里出现）。
#   4) 提供 render_prompt() 统一渲染"请求 + 候选技能表"，训练/评测共用。
# =====================================================================

from __future__ import annotations

import json
import random
from typing import Dict, List

# ---------------------------------------------------------------
# 1. 技能库（6 域 × 4 技能 = 24 个；每域最后一个是 held-out）
#    字段：name / domain / desc / args / returns / heldout / requests
# ---------------------------------------------------------------
SKILLS: List[Dict] = [
    # ---------------- 域 1：列表处理 ----------------
    dict(name="sort_by_value", domain="list",
         desc="对字典按 value 升序排序并返回键值对列表",
         args={"data": "dict"}, returns="list[tuple]",
         requests=["把这份销售数据按销售额从小到大排好给我",
                   "按出现次数升序排列这些词频统计",
                   "帮我把这个映射按值排个序（小到大）"]),
    dict(name="sort_by_key", domain="list",
         desc="对字典按键名字母序排序并返回键值对列表",
         args={"data": "dict"}, returns="list[tuple]",
         requests=["把这些配置项按键名字母顺序排列",
                   "我想看按名称排序后的字段表",
                   "把这份数据按键的字典序整理一下"]),
    dict(name="sort_desc_numeric", domain="list",
         desc="对数字列表做降序排序",
         args={"data": "list[number]"}, returns="list[number]",
         requests=["把这些分数从高到低排一下",
                   "给我一个从大到小的数字序列",
                   "按降序输出这组数值"]),
    dict(name="sort_by_length", domain="list", heldout=True,
         desc="按字符串长度排序列表",
         args={"data": "list[str]"}, returns="list[str]",
         requests=["把这些句子按长短排一排",
                   "按字符串长度从小到大整理这个列表",
                   "我想看长度递增的字符串序列"]),

    # ---------------- 域 2：文本统计 ----------------
    dict(name="word_count", domain="text",
         desc="统计文本中的词数与字符数",
         args={"text": "str"}, returns="dict",
         requests=["帮我数一下这段话有多少词",
                   "统计这段文字的词数和字符数",
                   "这段文本到底有多长？词数和字数都给我"]),
    dict(name="char_frequency", domain="text",
         desc="统计文本中各字符出现频率",
         args={"text": "str"}, returns="dict",
         requests=["看看这段文字里每个字符出现了多少次",
                   "统计字符频率",
                   "哪些字符在这段文本里最常出现？给我频次"]),
    dict(name="unique_words", domain="text",
         desc="返回文本中的去重词集合",
         args={"text": "str"}, returns="list[str]",
         requests=["这段话里一共用了多少个不同的词？列出来",
                   "把重复的词去掉，只给我唯一词表",
                   "提取不重复的单词"]),
    dict(name="longest_common_prefix", domain="text", heldout=True,
         desc="求一组字符串的最长公共前缀",
         args={"strings": "list[str]"}, returns="str",
         requests=["这几个文件名开头相同的部分是什么",
                   "求最长公共前缀",
                   "找出这组字符串共同的起始片段"]),

    # ---------------- 域 3：时间日期 ----------------
    dict(name="days_between", domain="datetime",
         desc="计算两个日期之间的天数差",
         args={"start": "date", "end": "date"}, returns="int",
         requests=["从 3 月 1 日到 6 月 15 日一共多少天",
                   "算一下这两个日期相差几天",
                   "这两天的间隔天数是多少"]),
    dict(name="add_days", domain="datetime",
         desc="在给定日期上增加若干天",
         args={"date": "date", "days": "int"}, returns="date",
         requests=["今天是 5 月 1 日，过 45 天是哪天",
                   "把这个日期往后推 30 天",
                   "在给定日期上加一段时间"]),
    dict(name="weekday_of", domain="datetime",
         desc="求某个日期是星期几",
         args={"date": "date"}, returns="str",
         requests=["2024 年 12 月 25 日是星期几",
                   "这天是周几？",
                   "帮我查一下这个日期对应星期几"]),
    dict(name="is_leap_year", domain="datetime", heldout=True,
         desc="判断某年是否为闰年",
         args={"year": "int"}, returns="bool",
         requests=["2024 年是不是闰年",
                   "这个年份有 2 月 29 日吗",
                   "判断该年是否为闰年"]),

    # ---------------- 域 4：数值计算 ----------------
    dict(name="mean_value", domain="math",
         desc="计算一组数字的算术平均值",
         args={"values": "list[number]"}, returns="float",
         requests=["这组成绩的平均分是多少",
                   "帮我算一下这组数的均值",
                   "求出平均值"]),
    dict(name="median_value", domain="math",
         desc="计算一组数字的中位数",
         args={"values": "list[number]"}, returns="float",
         requests=["这组数据的中位数是多少",
                   "帮我取中间那个数（中位数）",
                   "求中位数"]),
    dict(name="std_deviation", domain="math",
         desc="计算总体标准差",
         args={"values": "list[number]"}, returns="float",
         requests=["这组数据的波动有多大？给我标准差",
                   "算一下总体标准差",
                   "数据的离散程度如何（标准差）"]),
    dict(name="percentile", domain="math", heldout=True,
         desc="计算给定百分位数",
         args={"values": "list[number]", "p": "float"}, returns="float",
         requests=["这组数据的第 90 百分位是多少",
                   "帮我算 25 分位数",
                   "求给定百分位点"]),

    # ---------------- 域 5：字符串/编码 ----------------
    dict(name="to_snake_case", domain="string",
         desc="把驼峰命名转换为下划线命名",
         args={"text": "str"}, returns="str",
         requests=["把这个 CamelCase 变量名改成 snake_case",
                   "驼峰转下划线",
                   "把这段标识符改成下划线风格"]),
    dict(name="to_camel_case", domain="string",
         desc="把下划线命名转换为小驼峰命名",
         args={"text": "str"}, returns="str",
         requests=["把 user_name 这种写法改成 userName",
                   "下划线转驼峰",
                   "把这段标识符改成驼峰风格"]),
    dict(name="base64_encode", domain="string",
         desc="对字符串做 base64 编码",
         args={"text": "str"}, returns="str",
         requests=["把这段文本做 base64 编码",
                   "我要个 base64 串",
                   "给这段内容做 base64 转换"]),
    dict(name="url_encode", domain="string", heldout=True,
         desc="对字符串做 URL 百分号编码",
         args={"text": "str"}, returns="str",
         requests=["这段带空格和斜杠的文本要做 URL 编码",
                   "把查询参数做百分号转义",
                   "做 URL 编码"]),

    # ---------------- 域 6：文件/路径 ----------------
    dict(name="read_json_file", domain="file",
         desc="读取 JSON 文件并解析为对象",
         args={"path": "str"}, returns="any",
         requests=["把这个 config.json 读进来",
                   "解析这个 JSON 文件内容",
                   "帮我把该 json 文件加载成对象"]),
    dict(name="read_csv_file", domain="file",
         desc="读取 CSV 文件为表格",
         args={"path": "str"}, returns="table",
         requests=["把 data.csv 读成表格",
                   "加载这个 CSV 让我看列",
                   "读取该 csv 文件"]),
    dict(name="write_json_file", domain="file",
         desc="把对象序列化写入 JSON 文件",
         args={"path": "str", "obj": "any"}, returns="bool",
         requests=["把这份结果存成 result.json",
                   "把这个对象写进 JSON 文件",
                   "保存为 json 文件"]),
    dict(name="list_files_in_dir", domain="file", heldout=True,
         desc="列出目录下的文件",
         args={"path": "str"}, returns="list[str]",
         requests=["看看 logs 目录里都有什么文件",
                   "列出该目录下的文件清单",
                   "这个文件夹里有哪些文件"]),
]

# 便捷索引
BY_NAME = {s["name"]: s for s in SKILLS}
DOMAINS: Dict[str, List[Dict]] = {}
for _s in SKILLS:
    DOMAINS.setdefault(_s["domain"], []).append(_s)

HELDOUT = {s["name"] for s in SKILLS if s.get("heldout")}
TRAINABLE = [s for s in SKILLS if not s.get("heldout")]

# 与技能无关的请求（用于构造"无需技能"样本，对应 BFCL 的 irrelevance）
NO_SKILL_REQUESTS = [
    "今天天气怎么样？",
    "给我讲个笑话吧",
    "你叫什么名字？",
    "解释一下什么是强化学习",
    "用一句话夸夸我",
    "明天适合出门吗",
    "帮我写一首关于秋天的短诗",
    "你觉得哪种编程语言更好？说点你的看法",
]


def render_prompt(request: str, candidates: List[Dict]) -> str:
    """渲染"请求 + 候选技能表"，训练与评测共用同一模板。

    输出格式刻意简单、易于解析：
        用户请求：...
        可用技能：
        - name: desc (args)
        ...
        请只输出需要加载的技能名，JSON 形如 {"skills": ["name1", "name2"]}；
        若不需要任何技能，输出 {"skills": []}。
    """
    lines = ["用户请求：" + request, "", "可用技能："]
    for c in candidates:
        lines.append(f"- {c['name']}: {c['desc']} (参数: {json.dumps(c['args'], ensure_ascii=False)})")
    lines += [
        "",
        '请只输出需要加载的技能名，JSON 形如 {"skills": ["name1", "name2"]}；',
        '若不需要任何技能，输出 {"skills": []}。',
    ]
    return "\n".join(lines)


def sample_candidates(skill: Dict, rng: random.Random,
                      n_near: int = 3, n_far: int = 1) -> List[Dict]:
    """为一个任务抽候选技能：1 正确 + n_near 同域 near-miss + n_far 其他域干扰。"""
    same = [s for s in DOMAINS[skill["domain"]] if s["name"] != skill["name"]]
    rng.shuffle(same)
    near = same[:n_near]
    others = [s for s in SKILLS if s["domain"] != skill["domain"] and s["name"] != skill["name"]]
    rng.shuffle(others)
    far = others[:n_far]
    cands = [skill] + near + far
    rng.shuffle(cands)
    return cands


def sample_no_skill_candidates(rng: random.Random, k: int = 5) -> List[Dict]:
    """为"无需技能"的请求抽一批候选（全部为干扰项）。"""
    pool = list(SKILLS)
    rng.shuffle(pool)
    return pool[:k]
