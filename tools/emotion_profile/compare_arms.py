#!/usr/bin/env python3
"""多臂对比：把若干次实验的逐样本明细并排成一张表。

为什么要有它：`report.py` 只看单个臂。判断「这一版比上一版好在哪」需要把多个臂并排，
而 v1→v4 那几轮我都是临时脚本手搓的，别人复现不了。这里把它做成正式工具。

输入是 `src/cli.py eval --per-sample-out` 的产物（内部已含金标与预测，
所以不需要再传真值文件）。

用法::

    python tools/emotion_profile/compare_arms.py \\
        --arm official=runs/eval-smoke-official/samples.jsonl \\
        --arm v1=runs/eval-smoke-wired/samples.jsonl \\
        --arm v3=runs/eval-v3-smoke/samples.jsonl \\
        --out runs/comparison.md

    # 只打印不进文件
    python tools/emotion_profile/compare_arms.py --arm a=x.jsonl --arm b=y.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIMS = ("personality_traits", "interests", "style")


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def profile(record: dict, key: str, dim: str) -> list[str]:
    """取某一维的标签集合（已排序，便于比较）。"""
    values = (record.get(key) or {}).get(dim)
    return sorted(values) if isinstance(values, list) else []


def emotion_accuracy(rows: list[dict]) -> float:
    """情绪准确率。"""
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("emotion_hit")) / len(rows)


def dimension_stats(rows: list[dict], dim: str) -> dict:
    """一个维度的五态分解。"""
    stats = Counter()
    labels = Counter()
    for row in rows:
        gold = profile(row, "profile_gold", dim)
        predicted = profile(row, "profile_pred", dim)
        if not gold and not predicted:
            stats["空对空"] += 1
        elif not gold and predicted:
            stats["该空却填"] += 1
        elif gold and not predicted:
            stats["该填却空"] += 1
        elif gold == predicted:
            stats["命中"] += 1
        else:
            stats["填错"] += 1
        labels.update(predicted)
    total = len(rows)
    return {
        "ema": (stats["空对空"] + stats["命中"]) / total if total else 0.0,
        "stats": stats,
        "labels": labels,
        "filled": sum(1 for row in rows if profile(row, "profile_pred", dim)),
    }


def parse_arm(value: str) -> tuple[str, Path]:
    """解析 `名字=路径`。"""
    if "=" not in value:
        raise SystemExit(f"--arm 需要 `名字=路径` 的形式，收到 {value!r}")
    name, raw = value.split("=", 1)
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"找不到 {path}")
    return name.strip(), path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多臂对比（读 per-sample 明细）")
    parser.add_argument("--arm", action="append", required=True, help="`名字=per_sample.jsonl`，可重复")
    parser.add_argument("--out", default=None, help="写入 Markdown（默认只打印）")
    parser.add_argument("--per-sample-diff", action="store_true", help="额外列出各臂与第一个臂的逐条差异")
    args = parser.parse_args(argv)

    arms = [parse_arm(value) for value in args.arm]
    data: dict[str, dict[str, dict]] = {}
    for name, path in arms:
        rows = read_jsonl(path)
        data[name] = {row["id"]: row for row in rows}
        print(f"[读入] {name}: {len(rows)} 条  ({path})", flush=True)

    names = [name for name, _ in arms]
    common_ids = sorted(set.intersection(*(set(data[name]) for name in names)))
    if not common_ids:
        raise SystemExit("各臂之间没有共同的样本 id")
    if any(len(data[name]) != len(common_ids) for name in names):
        print(f"[注意] 各臂样本数不一致，只比较共有的 {len(common_ids)} 条")

    lines: list[str] = []
    lines.append(f"# 多臂对比（{len(common_ids)} 条共有样本）")
    lines.append("")
    lines.append("| 指标 | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")

    def row_of(label: str, value_of) -> str:
        return f"| {label} | " + " | ".join(value_of(name) for name in names) + " |"

    lines.append(row_of("情绪 Acc", lambda n: f"{emotion_accuracy([data[n][i] for i in common_ids]):.4f}"))
    mean_profile = {
        n: sum(dimension_stats([data[n][i] for i in common_ids], dim)["ema"] for dim in DIMS) / len(DIMS)
        for n in names
    }
    lines.append(row_of("画像 Acc（三维均值）", lambda n: f"{mean_profile[n]:.4f}"))
    for dim in DIMS:
        stats = {n: dimension_stats([data[n][i] for i in common_ids], dim) for n in names}
        lines.append(row_of(f"└ {dim}", lambda n, s=stats: f"{s[n]['ema']:.4f}"))
    lines.append(row_of("情绪预测空值率", lambda n: f"{sum(1 for i in common_ids if data[n][i].get('emotion_pred') == '<empty>') / len(common_ids):.4f}"))

    lines.append("")
    lines.append("## 逐维五态分解")
    lines.append("")
    lines.append("| 维度 | 臂 | EMA | 空对空 | 命中 | 该空却填 | 该填却空 | 填错 | 填了 | 输出标签频次 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for dim in DIMS:
        for name in names:
            stats = dimension_stats([data[name][i] for i in common_ids], dim)["stats"]
            item = dimension_stats([data[name][i] for i in common_ids], dim)
            lines.append(
                f"| {dim} | {name} | {item['ema']:.4f} | {stats['空对空']} | {stats['命中']} | "
                f"{stats['该空却填']} | {stats['该填却空']} | {stats['填错']} | {item['filled']} | "
                f"{dict(item['labels'].most_common(6))} |"
            )

    if args.per_sample_diff and len(names) > 1:
        base = names[0]
        lines.append("")
        lines.append(f"## 逐条差异（基准 = {base}）")
        for name in names[1:]:
            lines.append("")
            lines.append(f"### {base} → {name}")
            changed = 0
            for sample_id in common_ids:
                left, right = data[base][sample_id], data[name][sample_id]
                bits = []
                if left.get("emotion_pred") != right.get("emotion_pred"):
                    bits.append(
                        f"情绪 {left.get('emotion_gold')}: {left.get('emotion_pred')} → {right.get('emotion_pred')}"
                    )
                for dim in DIMS:
                    a, b = profile(left, "profile_pred", dim), profile(right, "profile_pred", dim)
                    if a != b:
                        gold = profile(right, "profile_gold", dim)
                        bits.append(f"{dim} 金标{gold}: {a} → {b}")
                if bits:
                    changed += 1
                    lines.append(f"- `{sample_id}`: " + "；".join(bits))
            if not changed:
                lines.append("- （没有差异）")
            lines.append(f"\n共 {changed} 条有差异。")

    text = "\n".join(lines) + "\n"
    print()
    print(text)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
