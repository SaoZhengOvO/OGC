#!/usr/bin/env python3
"""守护链一致性审计：把两类"不会报错、只在凌晨静默出错"的 bug 变成可检查项。

背景（两类 bug 都是这一夜真实踩到的）
------------------------------------
1. **重复触发条件 → 并发违规**
   `role2-slice-v7.sh` 与 `role2-integration.sh` 原来都在等 `eval-v6-slice200/score.json`，
   于是 v6 一结束两者同时启动，加上还在跑的全量就是 **3 个 GPU 任务**，违反
   `runs/GPU-RULES.md` 的「同时最多 2 个」。这类 bug 不报错，只在触发时多占一张卡。

2. **路径不一致 → 静默跑错目录**
   `role2-integ-full.sh` 一度是「`PAIR=runs/p02-v3+p03new-full`，但
   `--prompt-dir .../p02-v3+p03new`」——把文件拷到 A、却拿 B 去跑。
   结果标签正确、内容来自别处。同样不报错。

用法::

    python tools/emotion_profile/watch_audit.py                       # 审仓库里的 watchers/
    python tools/emotion_profile/watch_audit.py --dir /mnt/workspace/.team-ssh
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "tools" / "emotion_profile" / "watchers"

#: 等待条件：`[ -f <路径> ]` —— 用于找"多个守护等同一个文件"
WAIT_PATTERN = re.compile(r"\[\s*-f\s+([^\s\]]+)\s*\]")
PAIR_PATTERN = re.compile(r"^PAIR=(\S+)", re.M)
PROMPT_DIR_PATTERN = re.compile(r"--prompt-dir\s+(\S+)")

#: 判定"这个脚本会不会占卡"：只有真正跑推理的才算。
#: 必须带上 `python3` 前缀 —— 否则会把**锁文件内容里的 echo 字符串**
#: （例如 `跑什么: tools/eval_loop.py --tag v3-full`）也算成调用，
#: 于是一堆早已不跑的旧脚本被误判为"占卡守护"（审计噪声）。
GPU_MARKER = "python3 tools/eval_loop.py"


def uses_gpu(text: str) -> bool:
    """脚本里有没有真正**调用** eval_loop.py（那才会起推理、占卡）。

    为什么必须区分：`role2-status.sh` / `role2-health.sh` 这类诊断脚本也会去
    `[ -f ... ]` 看各种产物，如果把它们也算进"等待者"，审计会淹没在噪声里。
    **并发违规只可能发生在两个都会占卡的守护之间。**
    """
    return GPU_MARKER in text


def strip_comments(text: str) -> str:
    """去掉整行注释，避免把注释里的路径当参数（审计脚本自己被这个坑过一次）。"""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="守护链一致性审计")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="守护脚本目录")
    args = parser.parse_args(argv)

    base = Path(args.dir)
    if not base.is_dir():
        raise SystemExit(f"目录不存在: {base}")
    scripts = sorted(base.glob("role2-*.sh"))
    if not scripts:
        raise SystemExit(f"{base} 下没有 role2-*.sh")

    texts = {script.name: strip_comments(script.read_text(encoding="utf-8")) for script in scripts}
    problems = 0

    # --- 检查 1：多个守护等同一个文件 ---
    waiters: dict[str, list[str]] = defaultdict(list)
    for name, text in texts.items():
        for target in set(WAIT_PATTERN.findall(text)):
            waiters[target].append(name)

    print("=== 检查 1：占卡守护的等待条件是否撞车（真正的并发风险）===")
    gpu_scripts = {name for name, text in texts.items() if uses_gpu(text)}
    cpu_scripts = sorted(set(texts) - gpu_scripts)
    print(f"  占卡守护 {len(gpu_scripts)} 个：{sorted(gpu_scripts)}")
    print(f"  纯 CPU / 诊断脚本 {len(cpu_scripts)} 个（不参与并发判定）")
    print()

    real_clashes = {}
    soft_clashes = {}
    for target, names in waiters.items():
        unique = sorted(set(names))
        if len(unique) < 2:
            continue
        gpu_waiters = [n for n in unique if n in gpu_scripts]
        if len(gpu_waiters) >= 2:
            real_clashes[target] = unique
        else:
            soft_clashes[target] = unique

    if not real_clashes:
        print("  ✅ 没有两个【占卡】守护等同一个文件（并发 ≤2 的前提成立）")
    else:
        for target, names in sorted(real_clashes.items()):
            print(f"  ❌ {target}")
            for name in names:
                mark = "占卡" if name in gpu_scripts else "CPU"
                print(f"       ← {name}（{mark}）")
            problems += 1

    if soft_clashes:
        print()
        print(f"  （另有 {len(soft_clashes)} 处只涉及 CPU/诊断脚本的撞车，逐个列出仅供参考：）")
        for target, names in sorted(soft_clashes.items()):
            marks = "、".join(f"{n}({'占卡' if n in gpu_scripts else 'CPU'})" for n in names)
            print(f"       {target}  ← {marks}")

    # --- 检查 2：PAIR 与 --prompt-dir 是否一致 ---
    print()
    print("=== 检查 2：PAIR 与 --prompt-dir 是否指向同一个目录 ===")
    for name, text in sorted(texts.items()):
        pair = PAIR_PATTERN.search(text)
        dirs = [d for d in PROMPT_DIR_PATTERN.findall(text) if not d.startswith("$")]
        if not dirs:
            continue
        if pair:
            pair_value = pair.group(1)
            bad = [d for d in dirs if pair_value not in d]
            if bad:
                print(f"  ❌ {name}: PAIR={pair_value} 但 prompt-dir={bad}")
                problems += 1
            else:
                print(f"  ✅ {name}: {pair_value}")
        else:
            print(f"  ➖ {name}: 无 PAIR，prompt-dir={dirs}")

    # --- 检查 3：输出目录是否被多个守护共用 ---
    print()
    print("=== 检查 3：提示词目录是否被多个守护共用 ===")
    owners: dict[str, list[str]] = defaultdict(list)
    for name, text in texts.items():
        for directory in set(PROMPT_DIR_PATTERN.findall(text)):
            if directory.startswith("$") or directory.startswith("/tmp"):
                continue
            owners[directory].append(name)
    shared = {d: names for d, names in owners.items() if len(names) > 1}
    if not shared:
        print("  ✅ 没有两个守护共用同一个提示词目录")
    else:
        for directory, names in sorted(shared.items()):
            # 同一守护里出现多次是正常的（组装 + 使用），只看跨守护
            unique = sorted(set(names))
            if len(unique) > 1:
                print(f"  ⚠️ {directory}")
                for name in unique:
                    print(f"       ← {name}")
                problems += 1
            else:
                print(f"  ✅ {directory}（仅 {unique[0]} 自己组装+使用）")

    print()
    print(f"共 {len(scripts)} 个守护脚本；发现 {problems} 处需要人工确认。")
    print("说明：检查 1 的撞车不一定是 bug（若两个守护本来就是串行设计），")
    print("      但**必须人工确认过**——这一夜就是因为没确认才多占了第三张卡。")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
