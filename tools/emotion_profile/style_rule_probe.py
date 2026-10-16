#!/usr/bin/env python3
"""用真值探测「哪些 style 规则真的能过盈亏线」（纯 CPU，零 GPU）。

动机
----
画像走「三维留空」的下限是 **0.6283**；要超过它，就必须在某些维度上填标签，
而集合精确匹配下**填的条目必须超过 50% 完全正确**才比留空强。
逐维的金标为空率是 personality 0.6185 / interests 0.7489 / **style 0.5174** ——
所以 style 是唯一有机会过线的维度（其余两维要求 >62% / >75% 的精度）。

那到底哪条 style 规则能过 52%？不能靠感觉。这个脚本把候选规则直接套在**全量真值**上，
算出每条规则的：

    触发条数 / 精确率 / 该空却填条数 / 净收益 Δ EMA

净收益的定义（就是集合精确匹配下填一条的期望变化）：

    Δ = (触发的条目里金标恰好等于预测集合的条数) − (触发的条目里金标为空的条数)

Δ > 0 才值得写进提示词。这是**上界估计**：它假设模型 100% 照规则执行，
真实模型会更差，所以只有 Δ 明显为正的规则才值得试。

用法::

    python tools/emotion_profile/style_rule_probe.py
    python tools/emotion_profile/style_rule_probe.py --dim interests   # 换一维试试
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_user_turn(sample: dict) -> str:
    """取目标轮之前最后一条用户消息。"""
    turns = [t for t in sample.get("history", []) if t.get("role") == "user"]
    if not turns:
        return ""
    turns.sort(key=lambda t: t.get("turn_id", 0))
    return turns[-1].get("content", "") or ""


def build_rules() -> dict[str, tuple[str, callable]]:
    """候选规则：名字 -> (预测的标签元组, 判定函数)。

    规则只写「明显可判定的」那几条 —— 这一版的目的是**证伪**，
    所以宁可少列，也不要塞进需要语感的规则。
    """
    def any_of(*patterns: str):
        compiled = [re.compile(p) for p in patterns]
        return lambda text: any(p.search(text) for p in compiled)

    return {
        # --- style ---
        "style=direct｜祈使/划边界": (("direct",), any_of(
            r"先别", r"别问", r"别急", r"就这样吧", r"先听我说", r"先听我", r"不用(?=给|急着)",
            r"能不能先", r"我想自己", r"先让我",
        )),
        "style=rational｜因果/对比": (("rational",), any_of(
            r"因为", r"所以", r"之所以", r"其实", r"真正", r"反而", r"与其",
        )),
        "style=high_emotional｜夸张比喻": (("high_emotional_expression",), any_of(
            r"摔", r"就是空气", r"悬在", r"简直", r"煎熬", r"受不了", r"崩溃", r"气死", r"真让人",
        )),
        "style=brief｜≤20 字": (("brief",), lambda text: len(text.strip()) <= 20),
        "style=brief｜≤30 字": (("brief",), lambda text: len(text.strip()) <= 30),
        "style=direct+rational｜同时命中": (("direct", "rational"), lambda text: (
            any_of(r"先别", r"别问", r"别急", r"就这样吧", r"先听我说")(text)
            and any_of(r"因为", r"所以", r"之所以", r"其实")(text)
        )),
        "style=rational｜只在句首出现因果词": (("rational",), lambda text: bool(
            re.search(r"^[^。！？]{0,6}(因为|所以|其实|之所以)", text.strip())
        )),
    }


def probe(dim: str, rule_name: str, labels: tuple[str, ...], predicate, rows: list[dict]) -> dict:
    """把一条规则套在全量真值上，算精度与净收益。"""
    predicted = sorted(labels)
    fired = hits = over_fill = missed = 0
    for row in rows:
        text = row["_text"]
        if not predicate(text):
            continue
        fired += 1
        gold = sorted((row.get("target") or {}).get("user_profile", {}).get(dim) or [])
        if not gold:
            over_fill += 1
        elif gold == predicted:
            hits += 1
        else:
            missed += 1
    total = len(rows)
    delta = (hits - over_fill) / total if total else 0.0
    precision = hits / fired if fired else 0.0
    return {
        "rule": rule_name,
        "fired": fired,
        "precision": precision,
        "hits": hits,
        "over_fill": over_fill,
        "missed": missed,
        "delta_ema": delta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探测 style 规则的精度与净收益")
    parser.add_argument("--dim", default="style", help="要探测的维度（默认 style）")
    args = parser.parse_args(argv)

    references = read_jsonl(PROCESSED / "val_references.jsonl")
    samples = {row["id"]: row for row in read_jsonl(PROCESSED / "val_samples.jsonl")}
    rows = []
    for reference in references:
        sample = samples.get(reference["id"])
        if sample is None:
            continue
        rows.append({**reference, "_text": last_user_turn(sample)})
    total = len(rows)

    gold_empty = sum(1 for row in rows if not ((row.get("target") or {}).get("user_profile", {}).get(args.dim) or []))
    print(f"维度 {args.dim}   全量 {total} 条   金标为空 {gold_empty} ({gold_empty / total:.4f})")
    print(f"→ 要超过「一律留空」的下限 {gold_empty / total:.4f}，填的条目必须精确率 > "
          f"{gold_empty / total:.4f} 才行\n")

    results = [probe(args.dim, name, labels, predicate, rows)
               for name, (labels, predicate) in build_rules().items()]
    results.sort(key=lambda item: -item["delta_ema"])

    print(f"{'规则':<40}{'触发':>6}{'精确率':>9}{'命中':>6}{'该空却填':>9}{'漏':>6}{'ΔEMA':>9}  结论")
    print("-" * 104)
    for item in results:
        verdict = "✅ 值得试" if item["delta_ema"] > 0.002 else (
            "➖ 约等于零" if item["delta_ema"] > -0.002 else "❌ 净亏"
        )
        print(f"{item['rule']:<40}{item['fired']:>6}{item['precision']:>9.4f}"
              f"{item['hits']:>6}{item['over_fill']:>9}{item['missed']:>6}"
              f"{item['delta_ema']:>+9.4f}  {verdict}")

    best = results[0]
    print(f"\n最好的单条规则：{best['rule']}   ΔEMA = {best['delta_ema']:+.4f}")
    print("注意：这是**上界估计**（假设模型 100% 照规则执行）。真实模型会更差，")
    print("      所以只有 Δ 明显为正（比如 > +0.01）才值得占一轮实验。")
    positive = [item for item in results if item["delta_ema"] > 0.01]
    if positive:
        print("\nΔEMA > +0.01 的规则：")
        for item in positive:
            print(f"  · {item['rule']}  Δ={item['delta_ema']:+.4f}  精确率 {item['precision']:.3f}")
    else:
        print("\n没有任何单条规则的 ΔEMA 超过 +0.01 —— "
              "说明「用关键词规则填 style」这条路的天花板很低，")
        print("更该考虑：把 style 留给模型自己判断（v4 正在测），而不是用规则硬填。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
