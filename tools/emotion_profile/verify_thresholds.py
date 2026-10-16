"""独立重算判读表的 5 个阈值（不调用 evaluate.py、不调用本目录的任何其它工具）。

**为什么单独有这个脚本**：报告 §十九 那张判读表是「守住验收门」的整套判据，
而它的地基就是下面这 5 个数。其中任何一个算错，整张表都会给出错误结论 ——
所以值得用**最朴素的定义**从头再算一遍。

判据（报告 §十九）：
  1. 情绪 Acc > 0.3190       ← 一律答 anxiety 的下限
  2. 画像 Acc > 0.6283       ← 三维一律留空的下限
  3. personality EMA > 0.6185 ← 该维金标留空率
  4. interests    EMA > 0.7489
  5. style        EMA > 0.5174

用法::
    python tools/emotion_profile/verify_thresholds.py

真实 schema 是 `{id, conversation_id, target: {...}}` —— 标签在 **target** 里面。
（第一版凭印象写了 `row["emotion_label"]`，全取到 None —— 又一次"猜字段名"。
 教训：期望值必须取自被检查的对象本身，见报告 §87 的同类记录。）
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DIMS = ("personality_traits", "interests", "style")

refs = [
    json.loads(line)
    for line in Path("data/processed/val_references.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(f"全量参考 {len(refs)} 条")
print("target 的字段:", sorted(refs[0]["target"]))
n = len(refs)

gold = Counter((row.get("target") or {}).get("emotion_label") for row in refs)
print()
print("=== 1) 情绪下限（一律答 anxiety）===")
print(f"  金标里 anxiety 占 {gold['anxiety']}/{n} = {gold['anxiety'] / n:.4f}")
for label, count in gold.most_common(3):
    print(f"    {label!s:<16} {count:>5} = {count / n:.4f}")

print()
print("=== 2) 画像下限（三维一律留空）===")
emas = {}
for dim in DIMS:
    empty = sum(1 for row in refs if not ((row.get("target") or {}).get("user_profile") or {}).get(dim))
    emas[dim] = empty / n
    print(f"  {dim:<20} 金标为空 {empty:>5}/{n} = {empty / n:.4f}")
mean3 = sum(emas.values()) / len(emas)
print(f"  三维均值 = {mean3:.4f}")

claim = {
    "情绪 Acc": gold["anxiety"] / n,
    "画像 Acc": mean3,
    "personality_traits": emas["personality_traits"],
    "interests": emas["interests"],
    "style": emas["style"],
}
reported = {
    "情绪 Acc": 0.3190,
    "画像 Acc": 0.6283,
    "personality_traits": 0.6185,
    "interests": 0.7489,
    "style": 0.5174,
}
print()
print("=== 与报告 §十九 的数字对照 ===")
ok = True
for key in claim:
    diff = abs(claim[key] - reported[key])
    mark = "✅" if diff < 5e-5 else "❌"
    ok = ok and diff < 5e-5
    print(f"  {mark} {key:<22} 重算 {claim[key]:.4f}  报告 {reported[key]:.4f}  差 {diff:.1e}")
print()
print("✅ 全部吻合（差 < 5e-5）" if ok else "❌ 有不一致，报告需要更正")
