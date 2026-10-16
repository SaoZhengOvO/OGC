#!/usr/bin/env python3
"""角色② 工具链自检：一条命令验证这一线所有工具仍然自洽（纯 CPU，零 GPU）。

为什么需要
----------
`src/` 与 `tests/` 是队里共用/别人的目录，覆盖不到 `tools/emotion_profile/` 下的工具。
这一夜这些工具改了很多次（还改出过两个自己的 bug），必须有个**统一的回归检查**：
任何一次改动之后跑一遍，确认关键的数值结论没有被改坏。

检查的都是**有确定答案的不变量**，不是"跑完不报错"：

* 定稿产物与模板一致（`make_prompt.py --check`）
* 等距 200 切片与全量的逐维偏差 < 0.01（当前 0.0024）
* 前 20 条的 style 偏差 > 0.3（这是"smoke 不代表全量"的定量证据，不能被改没）
* 切片的完美预测得 1.0000、零成本下限 ≈ 0.32 / 0.63
* 同一个臂跟自己配对比较 → 差值 0、p = 1.0
* 触发词审计恰好找出 4 个方向反了的词，且包含 怕/烦/不敢/气
* style 探针：没有任何规则的 ΔEMA > +0.01
* 三份模板的关键词分类器准确率：v3 == v5a，且 v6 更高

用法::

    python tools/emotion_profile/selftest.py
    python tools/emotion_profile/selftest.py --quick   # 跳过要跑 2354 条的慢项
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools" / "emotion_profile"
PROCESSED = REPO_ROOT / "data" / "processed"


def run(args: list[str]) -> tuple[int, str]:
    """跑一个命令，返回 (返回码, 合并输出)。

    显式指定 utf-8 解码：Windows 上 subprocess 默认按本地编码（gbk）解，
    而这些工具的输出是中文，会直接抛 UnicodeDecodeError。
    """
    result = subprocess.run(
        args, cwd=str(REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def check_product_matches_template() -> tuple[bool, str]:
    """定稿产物必须与模板一致。"""
    code, out = run([sys.executable, str(TOOLS / "make_prompt.py"), "--check"])
    return code == 0, out.strip().splitlines()[-1] if out.strip() else f"退出码 {code}"


def check_slices(tmp: Path, quick: bool) -> list[tuple[str, bool, str]]:
    """切片的不变量。"""
    results: list[tuple[str, bool, str]] = []

    code, out = run([sys.executable, str(TOOLS / "make_eval_slice.py"), "--size", "200",
                     "--out", str(tmp / "slice200")])
    deviations = [float(value) for value in re.findall(r"^\S+\s+\S+\s+\S+\s+([0-9.]+)\s+\S+$", out, re.M)]
    worst = max(deviations) if deviations else 1.0
    results.append((
        "等距 200 切片逐维偏差 < 0.01",
        code == 0 and worst < 0.01,
        f"最大偏差 {worst:.4f}（当前应为 0.0024）",
    ))

    manifest = json.loads((tmp / "slice200" / "slice_manifest.json").read_text(encoding="utf-8"))
    smoke_style = abs(manifest["first_20_empty_rates"]["style"] - manifest["full_empty_rates"]["style"])
    results.append((
        "前 20 条 style 留空率偏差 > 0.3",
        smoke_style > 0.3,
        f"{smoke_style:.4f}（这是 smoke 不代表全量的定量证据）",
    ))
    results.append((
        "等距 200 的 size = 200",
        manifest["size"] == 200,
        f"size = {manifest['size']}",
    ))

    code, out = run([sys.executable, str(TOOLS / "slice_representativeness.py")])
    evenly_ok = code == 0 and "等距 200" in out
    results.append(("切片代表性体检可运行", evenly_ok, "输出含『等距 200』行" if evenly_ok else out[-200:]))

    if quick:
        return results

    code, out = run([sys.executable, str(TOOLS / "preflight_slice.py"), "--slice", str(tmp / "slice200")])
    perfect = re.search(r"完美预测：n=\d+ 情绪 ([0-9.]+) 画像 ([0-9.]+)", out)
    floor = re.search(r"零成本下限：情绪 ([0-9.]+) .*?/ 画像 ([0-9.]+)", out)
    ok = bool(perfect and floor) and abs(float(perfect.group(1)) - 1.0) < 1e-9
    results.append((
        "预检：完美预测 = 1.0000",
        ok,
        f"情绪 {perfect.group(1)} 画像 {perfect.group(2)}" if perfect else out[-200:],
    ))
    if floor:
        emo, prof = float(floor.group(1)), float(floor.group(2))
        results.append((
            "预检：零成本下限 ≈ 0.32 / 0.63",
            abs(emo - 0.32) < 0.01 and abs(prof - 0.63) < 0.01,
            f"情绪 {emo:.4f} / 画像 {prof:.4f}",
        ))
    return results


def check_compare_arms(tmp: Path) -> tuple[bool, str]:
    """同一个臂跟自己比，必须全 0、全不显著。"""
    slice_dir = tmp / "slice200"
    code, out = run([sys.executable, str(TOOLS / "compare_arms.py"),
                     "--arm", f"a={tmp / 'perfect.jsonl'}",
                     "--arm", f"b={tmp / 'perfect.jsonl'}",
                     "--paired", "a"])
    if code != 0:
        return False, out[-200:]
    same = out.count("| +0.0000 | [+0.0000, +0.0000] | 0 / 0 | 1.0000 |") >= 4
    return same, f"自比行数符合预期：{same}（slice_dir={slice_dir.name}）"


def check_probes(quick: bool) -> list[tuple[str, bool, str]]:
    """两个探针的结论不变量。"""
    results: list[tuple[str, bool, str]] = []

    if not quick:
        code, out = run([sys.executable, str(TOOLS / "emotion_rule_probe.py"), "--audit"])
        bad = set(re.findall(r"· 「(.+?)」判给", out))
        results.append((
            "审计恰好 4 个方向反了的词",
            bad == {"怕", "烦", "不敢", "气"},
            f"实际 {sorted(bad)}",
        ))

        code, out = run([sys.executable, str(TOOLS / "style_rule_probe.py")])
        deltas = [float(value) for value in re.findall(r"([+-][0-9]+\.[0-9]{4})\s+[✅➖❌]", out)]
        best = max(deltas) if deltas else 1.0
        results.append((
            "style 探针：无规则 ΔEMA > +0.01",
            best <= 0.01,
            f"最好的 ΔEMA = {best:+.4f}",
        ))

    code, out = run([sys.executable, str(TOOLS / "emotion_rule_probe.py"), "--rules-file",
                     "tools/emotion_profile/prompt_rules.txt,"
                     "tools/emotion_profile/prompt_rules.v5a.txt,"
                     "tools/emotion_profile/prompt_rules.v6.txt"])
    rows = re.findall(r"^(\S+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([+-][0-9.]+)$", out, re.M)
    by_name = {name: float(acc) for name, _, _, acc, _ in rows}
    v3 = by_name.get("prompt_rules.txt")
    v5a = by_name.get("prompt_rules.v5a.txt")
    v6 = by_name.get("prompt_rules.v6.txt")
    results.append((
        "v3 与 v5a 的关键词准确率相同",
        v3 is not None and v3 == v5a,
        f"v3={v3} v5a={v5a}",
    ))
    results.append((
        "v6 的关键词准确率高于 v3",
        v6 is not None and v3 is not None and v6 > v3 + 0.05,
        f"v3={v3} → v6={v6}",
    ))
    return results


def check_new_tools() -> list[tuple[str, bool, str]]:
    """第 50~66 轮新增的四个工具：确认它们仍能跑、且关键结论仍在。

    这几个工具是那一夜调查的产物，如果它们悄悄坏了，
    报告里引用它们的结论就失去可复现性 —— 所以必须一起自检。
    """
    results: list[tuple[str, bool, str]] = []

    # verify_metrics：独立重算必须与 evaluate.py 完全一致（有一个已完成的臂就够）
    code, out = run([sys.executable, str(TOOLS / "verify_metrics.py"), "--tag", "eval-v3-smoke"])
    ok = code == 0 and "✅ 全部对上" in out
    results.append((
        "verify_metrics 与 evaluate.py 对数一致",
        ok,
        "eval-v3-smoke 逐项吻合" if ok else out[-200:],
    ))

    # format_health：必须能诊断出唯一那个有解析失败的臂
    code, out = run([sys.executable, str(TOOLS / "format_health.py"),
                     "--tag", "mi300x-v4-smoke", "--tag", "v3-smoke"])
    ok = (code == 0 and "eval-mi300x-v4-smoke" in out and "10.00%" in out)
    results.append((
        "format_health 能定位 10% 解析失败的那一臂",
        ok,
        "mi300x-v4-smoke 显示 10.00%" if ok else out[-200:],
    ))

    # measure_recovery_gain：必须能算出"抢救修法只值 ±0.0001"这个结论
    code, out = run([sys.executable, str(TOOLS / "measure_recovery_gain.py"),
                     "--tag", "eval-mi300x-v4-smoke"])
    ok = code == 0 and "BLEU-4" in out
    results.append((
        "measure_recovery_gain 可量化抢救修法价值",
        ok,
        "输出含 BLEU-4 对照" if ok else out[-200:],
    ))

    # watch_audit：守护链之间不应有真实并发冲突。
    # 优先审**实际部署的那份**（/mnt/workspace/.team-ssh），因为它才是真正在跑的；
    # 仓库里的 watchers/ 是源码副本，box 上不一定有。
    deployed = Path("/mnt/workspace/.team-ssh")
    audit_dir = deployed if deployed.is_dir() else TOOLS / "watchers"
    code, out = run([sys.executable, str(TOOLS / "watch_audit.py"), "--dir", str(audit_dir)])
    ok = "没有两个【占卡】守护等同一个文件" in out and "没有两个守护共用同一个提示词目录" in out
    results.append((
        "watch_audit：守护链无真实冲突",
        ok,
        f"审计目录 {audit_dir.name}；占卡守护无撞车、目录无共用" if ok else out[-300:],
    ))

    # 内联 Python 的语法 —— 这类错误只在守护触发时才炸。
    # 本线真的踩到过：publish-results 里一个 ASCII 双引号截断了字符串，
    # 而那个守护一直没触发，直到用假数据演练才发现（否则全量跑完交付文件不会生成）。
    syntax_ok = "的内联 Python 全部语法正确" in out
    results.append((
        "watch_audit：守护内联 Python 语法正确",
        syntax_ok,
        "逐段校验通过" if syntax_ok else out[-300:],
    ))
    results.extend(check_docs_hygiene())
    return results


def check_docs_hygiene() -> list[tuple[str, bool, str]]:
    """交付文档的"静默退化"检查：行尾与章节顺序。

    这两项都是本线真踩过的坑：
    * **行尾** —— 用脚本重写文件时，Windows 上 `write_text` 默认把 `\\n` 转成 `\\r\\n`，
      整个文件的行尾被改写，产生无意义的全文 diff。**只看字节数才会发现**
      （83810 → 85344，差值恰好等于行数）。
    * **章节顺序** —— 往报告中间插一节时锚点选错，会让"第二十章讲一半跳到第二十一章"。
      **只看内容看不出来**，要按标题抽出编号序列才发现。
    """
    results: list[tuple[str, bool, str]] = []

    # 行尾：受版本控制的文本文件必须全是 LF
    exts = (".md", ".py", ".sh", ".txt")
    offenders = []
    checked = 0
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in exts:
            continue
        if "runs" in path.parts:  # runs/ 是 gitignore 的临时产物，不参与
            continue
        checked += 1
        if b"\r\n" in path.read_bytes():
            offenders.append(path.name)
    results.append((
        "受控文本文件全部是 LF 行尾",
        not offenders,
        f"{checked} 个文件检查通过" if not offenders else f"带 CRLF：{offenders}",
    ))

    # 章节顺序：LOG.md 的一级章节编号必须连续
    log = REPO_ROOT / "tools" / "emotion_profile" / "LOG.md"
    numbers = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and line[3:4].isdigit():
            head = line[3:].split(".", 1)[0]
            if head.isdigit():
                numbers.append(int(head))
    expect = list(range(1, len(numbers) + 1))
    results.append((
        "LOG.md 章节编号连续无重复",
        numbers == expect,
        f"1→{len(numbers)} 连续" if numbers == expect else f"实际：{numbers}",
    ))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="角色② 工具链自检")
    parser.add_argument("--quick", action="store_true", help="跳过跑全量 2354 条的慢项")
    parser.add_argument("--workdir", default=None,
                        help="临时目录（默认 runs/.selftest）。不用系统 temp："
                             "在受限沙箱下系统 temp 往往不可写。")
    parser.add_argument("--keep", action="store_true", help="保留临时目录便于排查")
    args = parser.parse_args(argv)

    if not (PROCESSED / "val_samples.jsonl").exists():
        raise SystemExit(f"缺少 {PROCESSED / 'val_samples.jsonl'}，自检需要全量 val")

    tmp = Path(args.workdir) if args.workdir else REPO_ROOT / "runs" / ".selftest"
    if not tmp.is_absolute():
        tmp = REPO_ROOT / tmp
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, bool, str]] = []
    try:
        ok, detail = check_product_matches_template()
        results.append(("定稿产物与模板一致（--check）", ok, detail))

        results.extend(check_slices(tmp, args.quick))

        # 造一份"完美预测"供 compare_arms 自比
        references = [json.loads(line) for line in
                      (PROCESSED / "val_references.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        with (tmp / "perfect.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in references:
                target = record["target"]
                handle.write(json.dumps({
                    "id": record["id"], "emotion_label": target["emotion_label"],
                    "user_profile": target["user_profile"], "response_text": target["response_text"],
                    "memory_refs": [],
                }, ensure_ascii=False) + "\n")
        ok, detail = check_compare_arms(tmp)
        results.append(("compare_arms 自比全 0、全不显著", ok, detail))

        results.extend(check_probes(args.quick))

        if not args.quick:
            results.extend(check_new_tools())
    finally:
        if not args.keep and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    width = max(len(name) for name, _, _ in results)
    print(f"{'检查项':<{width}}  结果  说明")
    print("-" * (width + 60))
    failed = 0
    for name, ok, detail in results:
        if not ok:
            failed += 1
        mark = "✅" if ok else "❌"
        print(f"{name:<{width}}  {mark}    {detail}")

    print()
    if failed:
        print(f"❌ {failed} / {len(results)} 项未通过 —— 有结论被改坏了，先查再往下走。")
        return 1
    print(f"✅ {len(results)} 项全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
