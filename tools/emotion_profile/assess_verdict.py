#!/usr/bin/env python3
"""按报告 §十九 的判读表，对全量结果出机械裁决（不靠手写叙述）。

为什么要有它：§十九 那张表是"守住验收门"的整套判据，但**逐条判定如果靠人写**，
就很容易在"差一点点"的时候被叙述带偏（"基本达标""接近门"这种说法）。
把它做成脚本，**每条判据的结论都由数字直接给出**，并且：

* 点估计是否过门 → 直接比阈值；
* 是否"守住" → **看配对显著性（第 6 条）**，不看点估计；
* 第 0 条解析失败是硬门槛，> 2% 直接判不合格。

同时核对 §19.4 的三条预登记预测（预测本身也由脚本判"命中/被证伪"）。

用法::

    python tools/emotion_profile/assess_verdict.py
    python tools/emotion_profile/assess_verdict.py --tag v3-full
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: §十九 的判据：(编号, 名称, 阈值, 方向, 说明)
CRITERIA = [
    ("0", "parse_failure_count ≤ 2%", 0.02, "le",
     "解析失败会让情绪与画像必定丢；> 2% 即不合格（硬门槛）"),
    ("1", "情绪 Acc > 0.3190", 0.3190, "gt", "一律答 anxiety 的下限"),
    ("2", "画像 Acc > 0.6283", 0.6283, "gt", "三维一律留空的下限"),
    ("3", "personality_traits EMA > 0.6185", 0.6185, "gt", "该维金标留空率"),
    ("4", "interests EMA > 0.7489", 0.7489, "gt", "该维金标留空率"),
    ("5", "style EMA > 0.5174", 0.5174, "gt", "该维金标留空率"),
]

#: §19.4 的预登记预测：(名称, 预测值, 容差)
PREDICTIONS = [
    ("v3 全量画像 ≈ 打平下限 0.6283（±0.01 内算命中）", 0.6283, 0.01),
]

#: 散落在别处、也必须一起核的预测（第 136 轮才发现，第 161 轮补进脚本）
#: —— 埋在论证里的预测如果不登记，跑完就只会去对 §19.4 那张表而把它们忘掉。
BURIED_PREDICTIONS = [
    # (出处, 指标, 判据说明)
    ("§14.3 后半句", "画像应**明显超过 E1 的 0.5615** —— 只打平下限不算违反，低于 0.5615 即被证伪"),
    ("§16.3 表格", "情绪**大概率远低于 0.50**；**能否超过下限 0.3190 未知**（不设成败，只如实报告）"),
]

E1_EMOTION = 0.2464
E1_PROFILE = 0.5615


def load(tag: str) -> tuple[dict, dict] | None:
    run = REPO_ROOT / "runs" / f"eval-{tag}"
    score_file = run / "score.json"
    perf_file = run / "performance_report.json"
    if not score_file.exists() or not perf_file.exists():
        return None
    return (
        json.loads(score_file.read_text(encoding="utf-8"))["metrics"],
        json.loads(perf_file.read_text(encoding="utf-8")),
    )


def paired_significant(tag: str) -> tuple[bool | None, str]:
    """从 FULL-COMPARISON.md 里读"是否显著"那一列。"""
    path = REPO_ROOT / "runs" / "FULL-COMPARISON.md"
    if not path.exists():
        return None, "runs/FULL-COMPARISON.md 不存在"
    text = path.read_text(encoding="utf-8")
    if "本次没有做成配对对比" in text:
        return None, "对比文件是占位说明（后处理失败），无显著性证据"
    rows = [line for line in text.splitlines() if line.startswith("| ") and "|" in line]
    hits = [line for line in rows if "✅ 显著" in line or "显著" in line]
    if not hits:
        return False, "对比表里没有任何一行标为显著"
    return True, f"{len(hits)} 行标为显著"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按 §十九 判读表出机械裁决")
    parser.add_argument("--tag", default="v3-full")
    args = parser.parse_args(argv)

    loaded = load(args.tag)
    if loaded is None:
        print(f"⏳ runs/eval-{args.tag}/ 还没有完整的产物（score.json + performance_report.json）")
        print("   全量数字尚未产出 —— 这与报告里的「未跑」声明一致。")
        return 2
    metrics, perf = loaded

    samples = perf.get("samples") or metrics.get("n_comparable") or 0
    failures = perf.get("parse_failure_count") or 0
    actual = {
        "0": failures / samples if samples else 0.0,
        "1": metrics["emotion_accuracy"],
        "2": metrics["profile_accuracy_mean_3dim"],
        "3": metrics["profile_dimension_scores"]["personality_traits"]["exact_match_acc"],
        "4": metrics["profile_dimension_scores"]["interests"]["exact_match_acc"],
        "5": metrics["profile_dimension_scores"]["style"]["exact_match_acc"],
    }

    print(f"=== 按 §十九 判读表裁决（tag={args.tag}，n={samples}）===")
    print()
    print(f"{'#':<3}{'判据':<34}{'实测':>10}{'阈值':>10}  判定")
    print("-" * 78)
    passed_all = True
    for code, name, threshold, direction, _note in CRITERIA:
        value = actual[code]
        if direction == "le":
            ok = value <= threshold
        else:
            ok = value > threshold
        passed_all = passed_all and ok
        mark = "✅ 过" if ok else "❌ 未过"
        print(f"{code:<3}{name:<34}{value:>10.4f}{threshold:>10.4f}  {mark}")

    significant, why = paired_significant(args.tag)
    print()
    if significant is None:
        print(f"6  相对下限的配对显著性（p < 0.05）        ⏳ {why}")
    else:
        print(f"6  相对下限的配对显著性（p < 0.05）        {'✅ 有显著项' if significant else '❌ 无显著项'} —— {why}")

    print()
    print("=== §19.4 的预登记预测 ===")
    for name, predicted, tol in PREDICTIONS:
        value = actual["2"]
        hit = abs(value - predicted) <= tol
        print(f"  {'✅ 命中' if hit else '⚠️ 与预测不符'}：{name}")
        print(f"      预测 {predicted:.4f} / 实测 {value:.4f}（差 {value - predicted:+.4f}）")

    print()
    print("=== 散落在别处的两条预测（别漏）===")
    profile_value = actual["2"]
    print(f"  §14.3 后半句：画像应明显超过 E1 的 {E1_PROFILE:.4f}")
    print(f"      实测画像 {profile_value:.4f} vs E1 {E1_PROFILE:.4f}（差 {profile_value - E1_PROFILE:+.4f}）"
          f" → {'✅ 超过' if profile_value > E1_PROFILE else '❌ 低于 E1，**这条被证伪**'}")
    emotion_value = actual["1"]
    print(f"  §16.3 表格：情绪「大概率远低于 0.50；能否超过下限 0.3190 未知」")
    print(f"      实测情绪 {emotion_value:.4f}（vs 0.5000 差 {emotion_value - 0.5:+.4f}；"
          f"vs 下限 0.3190 差 {emotion_value - 0.3190:+.4f}）")
    print("      —— 这条是「未知」型预测，**不判成败**，只如实报告。")

    print()
    print("=== 结论 ===")
    if not passed_all:
        print("  ❌ **第 1~5 条里有未过的** —— 按 §十九 不能宣称守住验收门。")
    elif significant is None:
        print("  ⚠️ 第 1~5 条都过了，但**第 6 条（配对显著性）没有证据** ——")
        print("     只能说「点估计过门」，**不能说「守住」**。")
    elif not significant:
        print("  ❌ 点估计过门，但**配对检验不显著** —— 按 §十九 只能说「点估计更好」。")
    else:
        print("  ✅ 七条判据全部满足。")
    print()
    print("> 绝对值口径仅供参考（本地评分器是近似实现，见 score.json 的 `_caveat`）；")
    print("> **相对下限的对比才是强证据** —— 以第 6 条为准。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
