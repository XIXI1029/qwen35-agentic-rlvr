# -*- coding: utf-8 -*-
# =====================================================================
# code_sandbox.py —— 代码执行的「T1 受限沙箱」（本项目 Code RL 的执行裁判）
#
# 【定位】不是生产级沙箱（那是 Docker/nsjail 的活），而是"研究/自建数据集"
#   够用的受限执行器，提供四道护栏：
#     ① 子进程隔离 + 硬超时（超时判定为"错"，并 kill 整个进程组）
#     ② 资源限制（CPU 时间 / 内存 / 文件大小 / 进程数，Linux 用 rlimit）
#     ③ 断网（runner 预置 socket 拦截）
#     ④ 独立临时工作目录（用完即删），并限制输出体积
#
# 【用法】
#   from code_sandbox import extract_python, run_tests, run_many
#   ok, info = run_tests(candidate_code, test_code, timeout=3.0)
#
# ⚠️ 安全提示：模型生成的代码是"不可信输入"。本沙箱能挡住死循环/吃内存/
#    刷屏，但【挡不住】刻意的提权/内核漏洞利用。请不要用 root 跑，
#    最好在容器/低权限用户下运行（README 有说明）。
# =====================================================================

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# ---------------------------------------------------------------
# 1. 从模型输出里抠出 Python 代码
# ---------------------------------------------------------------
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python(text: str) -> str:
    """从模型回复里提取代码：优先取 ```python ... ``` 代码块；
    没代码块就从第一个 def/import/class 开始截到结尾。"""
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    # 退化：找第一个顶格的 def/import/class
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(def |class |import |from )", ln):
            start = i
            break
    return "\n".join(lines[start:]).strip() if start is not None else text.strip()


# ---------------------------------------------------------------
# 2. runner：在被测进程里预置"断网 + 资源限制 + 结果哨兵"
# ---------------------------------------------------------------
_RUNNER = r'''
import sys, os
try:
    import resource          # Linux 才有；Windows 上为 None，退化为只靠超时
except Exception:
    resource = None
# ---- 断网：把 socket 换成直接抛错的桩 ----
try:
    import socket
    def _blocked(*a, **k):
        raise OSError("network disabled in sandbox")
    socket.socket = _blocked
    socket.create_connection = _blocked
except Exception:
    pass
# ---- 资源限制（Linux；Windows 无 resource 则跳过）----
try:
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (__CPU__, __CPU__ + 1))
    resource.setrlimit(resource.RLIMIT_AS, (__MEM__, __MEM__))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except Exception:
        pass
except Exception:
    pass
# ---- 依次执行 candidate.py 与 tests.py，打印哨兵供父进程判断 ----
try:
    g = {"__name__": "__main__"}
    exec(compile(open("candidate.py", encoding="utf-8").read(), "candidate.py", "exec"), g)
    exec(compile(open("tests.py", encoding="utf-8").read(), "tests.py", "exec"), g)
    print("__SANDBOX_PASS__")
except BaseException as e:
    print("__SANDBOX_FAIL__", type(e).__name__, str(e)[:200])
'''


def _limits_fn(cpu_sec: int, mem_bytes: int):
    """返回 preexec_fn：给子进程设置 rlimit（仅 Linux 有效）。"""
    def _fn():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec + 1))
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
                # 注意：NPROC 设太小会导致解释器起线程失败，这里只限制到 64
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            except Exception:
                pass
        except Exception:
            pass  # Windows / 无 resource：退化为只靠超时
    return _fn


# ---------------------------------------------------------------
# 3. 单次执行：candidate 代码 + 测试代码 -> (是否通过, 说明)
# ---------------------------------------------------------------
def run_tests(candidate_code: str,
              test_code: str,
              timeout: float = 3.0,
              mem_mb: int = 1024) -> Tuple[bool, str]:
    """在受限子进程里跑 candidate + tests。

    Args:
        candidate_code: 模型写的函数代码
        test_code:      测试代码（assert 集合 或 含 check() 的 HumanEval 测试）
        timeout:        墙钟超时（秒），超时判错
        mem_mb:         内存上限（MB）
    Returns:
        (passed, info)  passed=True 表示所有测试通过
    """
    if not candidate_code.strip():
        return False, "empty code"
    tmp = tempfile.mkdtemp(prefix="sandbox_")
    try:
        with open(os.path.join(tmp, "candidate.py"), "w", encoding="utf-8") as f:
            f.write(candidate_code)
        with open(os.path.join(tmp, "tests.py"), "w", encoding="utf-8") as f:
            f.write(test_code)
        runner = (_RUNNER.replace("__CPU__", str(int(timeout) + 1))
                         .replace("__MEM__", str(mem_mb * 1024 * 1024)))
        with open(os.path.join(tmp, "runner.py"), "w", encoding="utf-8") as f:
            f.write(runner)

        env = {  # 最小环境，避免继承奇怪的变量/代理
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        t0 = time.time()
        p = subprocess.Popen(
            [sys.executable, "-I", "runner.py"],   # -I: 隔离模式，忽略 PYTHONPATH 等
            cwd=tmp, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,     # 独立进程组，便于整组 kill
            preexec_fn=_limits_fn(int(timeout) + 1, mem_mb * 1024 * 1024)
            if os.name != "nt" else None,
        )
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:                                    # 超时：kill 整个进程组
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                p.kill()
            return False, "timeout"
        out = (out or "")[:4000]                    # 限制输出体积
        dt = time.time() - t0
        if "__SANDBOX_PASS__" in out:
            return True, f"pass ({dt:.2f}s)"
        # 从输出里提取失败原因
        m = re.search(r"__SANDBOX_FAIL__\s*(\S+)\s*(.*)", out)
        reason = f"{m.group(1)}: {m.group(2)[:120]}" if m else out.strip()[-200:]
        return False, reason
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------
# 4. 批量并行执行（训练时一轮有多种本，串行会拖慢每步）
# ---------------------------------------------------------------
def run_many(items: List[Tuple[str, str]],
             timeout: float = 3.0,
             mem_mb: int = 1024,
             max_workers: int = 8) -> List[Tuple[bool, str]]:
    """items = [(candidate_code, test_code), ...]，并行跑。

    subprocess 期间会释放 GIL，所以多线程能真正并行（受 CPU 核数限制）。
    """
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(items)))) as ex:
        return list(ex.map(lambda it: run_tests(it[0], it[1], timeout, mem_mb), items))


# ---------------------------------------------------------------
# 5. 自测：直接 python code_sandbox.py 可验证沙箱行为
# ---------------------------------------------------------------
if __name__ == "__main__":
    ok_code = "def add(a, b):\n    return a + b\n"
    bad_code = "def add(a, b):\n    return a - b\n"
    loop_code = "def add(a, b):\n    while True:\n        pass\n"
    tests = "assert add(1, 2) == 3\nassert add(-1, 1) == 0\n"

    print("正确代码 ->", run_tests(ok_code, tests))
    print("错误代码 ->", run_tests(bad_code, tests))
    print("死循环   ->", run_tests(loop_code, tests, timeout=2))   # 应报 timeout
    print("提取代码 ->", repr(extract_python("说明\n```python\ndef f():pass\n```")))
