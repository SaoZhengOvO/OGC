#!/usr/bin/env python3
"""把 `src/cli.py eval --per-sample-out` 的逐样本明细加工成写规则要看的报告。

回答的问题（角色②的第一件事就是这个）：
1. 情绪 16 类各自的 P/R/F1 是多少，support 多少；
2. 哪几类**最亏**——按「该类样本数 × 漏检率」排序，也就是还能捞回多少分；
3. 最常混淆的类别对（例如 anxiety/sadness/loneliness/helplessness 之间怎么混）；
4. 画像三个维度各自准确率，以及错误分成「该空却填」「该填却空」「写错」各多少；
   集合精确匹配下「该空却填」比「该填却空」严重得多，必须分开看；
5. 回复长度与参考长度的偏离（虽然回复归角色③，但长度会连带影响情绪/画像的上下文）。

用法::

    python src/cli.py eval --predictions runs/val_v1.jsonl --per-sample-out runs/val_v1.samples.jsonl
    python tools/emotion_profile/report.py --per-sample runs/val_v1.samples.jsonl
    python tools/emotion_profile/report.py --per-sample X.jsonl --json-out runs/val_v1.report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_SPACE = REPO_ROOT / "configs" / "label_space.json"
PROFILE_DIMS = ("personality_traits", "interests", "style")
EMPTY = "<empty>"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取逐样本明细。"""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: JSON 解析失败: {exc.msg}") from exc
    if not rows:
        raise SystemExit(f"{path} 是空的")
    return rows


def load_emotions() -> list[str]:
    """从唯一真相来源读 16 类情绪；读不到就按明细里出现过的金标排序。"""
    if LABEL_SPACE.exists():
        data = json.loads(LABEL_SPACE.read_text(encoding="utf-8"))
        if isinstance(data.get("emotions"), list):
            return [str(item) for item in data["emotions"]]
    return []


def pct(numerator: float, denominator: float) -> float:
    """安全的百分比。"""
    return numerator / denominator if denominator else 0.0


def emotion_section(rows: list[dict[str, Any]], emotions: list[str]) -> dict[str, Any]:
    """情绪：P/R/F1、混淆矩阵、最亏的类。"""
    support = Counter(row["emotion_gold"] for row in rows)
    predicted = Counter(row["emotion_pred"] for row in rows)
    hit = Counter(row["emotion_gold"] for row in rows if row["emotion_hit"])
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        confusion[row["emotion_gold"]][row["emotion_pred"]] += 1

    labels = sorted(set(emotions) | set(support) | set(predicted), key=lambda name: (-support.get(name, 0), name))
    per_class = []
    for name in labels:
        tp = hit.get(name, 0)
        fn = support.get(name, 0) - tp
        fp = predicted.get(name, 0) - tp
        precision = pct(tp, tp + fp)
        recall = pct(tp, tp + fn)
        f1 = pct(2 * precision * recall, precision + recall)
        per_class.append(
            {
                "emotion": name,
                "support": support.get(name, 0),
                "predicted": predicted.get(name, 0),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                # 还能捞回的上限：该类全对时总体 Acc 能涨多少
                "recoverable_acc": pct(fn, len(rows)),
            }
        )

    pairs = Counter()
    for row in rows:
        if not row["emotion_hit"]:
            pairs[(row["emotion_gold"], row["emotion_pred"])] += 1

    return {
        "n": len(rows),
        "accuracy": pct(sum(1 for row in rows if row["emotion_hit"]), len(rows)),
        "empty_pred_rate": pct(sum(1 for row in rows if row["emotion_pred"] == EMPTY), len(rows)),
        "empty_gold_rate": pct(sum(1 for row in rows if row["emotion_gold"] == EMPTY), len(rows)),
        "per_class": per_class,
        "confusion": {gold: dict(counter) for gold, counter in confusion.items()},
        "top_confusions": [{"gold": g, "pred": p, "count": c} for (g, p), c in pairs.most_common(15)],
        # 「最亏的几类」= 漏检数最多，也就是分数还能捞回来的地方
        "worst_classes": sorted(
            (item for item in per_class if item["support"] > 0),
            key=lambda item: (-item["fn"], -item["support"]),
        )[:8],
    }


def profile_section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """画像：逐维度准确率 + 五态分解。"""
    dimensions: dict[str, Any] = {}
    for dim in PROFILE_DIMS:
        stats = Counter()
        for row in rows:
            gold = sorted(row.get("profile_gold", {}).get(dim) or [])
            pred = sorted(row.get("profile_pred", {}).get(dim) or [])
            if not gold and not pred:
                stats["both_empty"] += 1
            elif not gold and pred:
                stats["over_filled"] += 1  # 该空却填：集合精确匹配下最亏
            elif gold and not pred:
                stats["missed"] += 1  # 该填却空
            elif gold == pred:
                stats["exact_hit"] += 1
            else:
                stats["wrong_values"] += 1
        total = len(rows)
        dimensions[dim] = {
            "exact_match_acc": pct(stats["both_empty"] + stats["exact_hit"], total),
            "both_empty": stats["both_empty"],
            "exact_hit": stats["exact_hit"],
            "over_filled": stats["over_filled"],
            "missed": stats["missed"],
            "wrong_values": stats["wrong_values"],
            # 上限：把 over_filled 与 wrong_values 全部改成空数组后的准确率
            "acc_if_only_safe": pct(stats["both_empty"] + stats["exact_hit"] + stats["missed"] + stats["wrong_values"], total),
            # 「该空却填」直接影响的上限
            "recoverable_by_silence": pct(stats["over_filled"], total),
        }
    empty_all = sum(
        1 for row in rows if all(not (row.get("profile_gold", {}).get(dim) or []) for dim in PROFILE_DIMS)
    )
    return {
        "dimensions": dimensions,
        "mean_3dim": sum(item["exact_match_acc"] for item in dimensions.values()) / len(PROFILE_DIMS),
        "gold_all_empty_rate": pct(empty_all, len(rows)),
    }


def length_section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """回复长度分布与偏离（回复归角色③，这里只做背景）。"""
    ratios = []
    for row in rows:
        reference = row.get("reference_chars") or 0
        if reference > 0:
            ratios.append(row.get("response_chars", 0) / reference)
    ratios.sort()
    return {
        "n_with_reference": len(ratios),
        "pred_chars_mean": sum(row.get("response_chars", 0) for row in rows) / len(rows),
        "reference_chars_mean": sum(row.get("reference_chars", 0) for row in rows) / len(rows),
        "ratio_median": ratios[len(ratios) // 2] if ratios else 0.0,
        "ratio_p90": ratios[int(len(ratios) * 0.9)] if ratios else 0.0,
        "empty_response_rate": pct(sum(1 for row in rows if not row.get("response_chars")), len(rows)),
    }


def format_report(samples_path: Path, emotion: dict, profile: dict, length: dict) -> str:
    """渲染给人看的文本报告。"""
    lines: list[str] = []
    lines.append(f"逐样本明细: {samples_path}  ({emotion['n']} 条)\n")

    lines.append("== 情绪 ==")
    lines.append(
        f"  Acc {emotion['accuracy']:.4f}   预测空值率 {emotion['empty_pred_rate']:.4f}   "
        f"金标空值率 {emotion['empty_gold_rate']:.4f}"
    )
    lines.append("  最亏的几类（按漏检数排序，rn = 还能捞回的总体 Acc）")
    lines.append(f"    {'情绪':<16}{'support':>8}{'命中':>6}{'漏检':>6}{'误报':>6}{'P':>7}{'R':>7}{'F1':>7}{'rn':>7}")
    for item in emotion["worst_classes"]:
        lines.append(
            f"    {item['emotion']:<16}{item['support']:>8}{item['tp']:>6}{item['fn']:>6}{item['fp']:>6}"
            f"{item['precision']:>7.3f}{item['recall']:>7.3f}{item['f1']:>7.3f}{item['recoverable_acc']:>7.4f}"
        )
    lines.append("\n  全部类别 per-class")
    lines.append(f"    {'情绪':<16}{'support':>8}{'P':>7}{'R':>7}{'F1':>7}")
    for item in sorted(emotion["per_class"], key=lambda x: -x["support"]):
        lines.append(
            f"    {item['emotion']:<16}{item['support']:>8}{item['precision']:>7.3f}"
            f"{item['recall']:>7.3f}{item['f1']:>7.3f}"
        )

    lines.append("\n  最常混淆的（金标 -> 预测）")
    for pair in emotion["top_confusions"]:
        lines.append(f"    {pair['gold']:<16} -> {pair['pred']:<16} {pair['count']:>5}")

    lines.append("\n== 画像 ==")
    lines.append(f"  三维均值 Acc {profile['mean_3dim']:.4f}   金标三维全空占比 {profile['gold_all_empty_rate']:.4f}")
    lines.append(
        f"    {'维度':<20}{'Acc':>7}{'空对空':>8}{'命中':>7}{'该空却填':>9}{'该填却空':>9}{'写错':>7}{'静默可涨':>9}"
    )
    for dim, item in profile["dimensions"].items():
        lines.append(
            f"    {dim:<20}{item['exact_match_acc']:>7.4f}{item['both_empty']:>8}{item['exact_hit']:>7}"
            f"{item['over_filled']:>9}{item['missed']:>9}{item['wrong_values']:>7}{item['recoverable_by_silence']:>9.4f}"
        )
    lines.append("  注：「该空却填」= 金标为空但模型写了标签，集合精确匹配下这一条直接判错；")
    lines.append("      「静默可涨」= 把这些条目改成空数组能带来的准确率增量上限。")

    lines.append("\n== 回复长度（背景信息，回复归角色③）==")
    lines.append(
        f"  平均长度 预测 {length['pred_chars_mean']:.1f} 字 / 参考 {length['reference_chars_mean']:.1f} 字；"
        f"预测/参考 中位 {length['ratio_median']:.2f}，P90 {length['ratio_p90']:.2f}；空回复率 {length['empty_response_rate']:.4f}"
    )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="情绪/画像逐类分析与 badcase 定位")
    parser.add_argument("--per-sample", required=True, help="src/cli.py eval --per-sample-out 的产物")
    parser.add_argument("--json-out", default=None, help="把完整报告写成 JSON")
    parser.add_argument("--top-bad", type=int, default=20, help="额外打印多少条最差样本（默认 20）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    path = Path(args.per_sample)
    if not path.is_absolute():
        path = REPO_ROOT / path
    rows = read_jsonl(path)

    emotion = emotion_section(rows, load_emotions())
    profile = profile_section(rows)
    length = length_section(rows)
    print(format_report(path, emotion, profile, length))

    if args.top_bad:
        bad = [
            row
            for row in rows
            if not row["emotion_hit"] and not all(row.get("profile_dim_hit", {}).values())
        ][: args.top_bad]
        print(f"\n== 同时错情绪又错画像的样本（前 {len(bad)} 条）==")
        for row in bad:
            worst_dims = [dim for dim, ok in row.get("profile_dim_hit", {}).items() if not ok]
            print(f"  {row['id']}  情绪 {row['emotion_gold']} -> {row['emotion_pred']}  画像错维度 {worst_dims}")

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"emotion": emotion, "profile": profile, "length": length}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON 报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
