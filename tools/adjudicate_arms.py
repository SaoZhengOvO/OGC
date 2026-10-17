#!/usr/bin/env python3
"""四臂对照的裁决器：把「该发哪一版 prompt」从叙事变成可复算的算术。

## 为什么需要这个脚本

官方对「回应质量 20 分」**没有公布公式**，只公布了三项指标名（BLEU-4 / ROUGE-L /
BERTScore）和「综合」。任何单一加权都只是假设。所以这里不假装知道公式，而是：

1. 在**四种不同的合成方式**下分别算一遍综合分（含最保守的「只算 BLEU」）；
2. 只有当四种方式**指向同一个赢家**时才说「结论稳健」；
3. 否则明确输出「结论依赖加权假设」，并指出分歧来自哪一项。

画像也同理：旧口径（`set()==set()` 算命中）会把「预测空数组」判成命中，官方口径按
`GivenData/participant/README.md:100-102`「空值作为该项预测失败」判成 0。这里只读
官方口径那个字段，旧口径只用于对照展示。

## 用法

    python3 tools/adjudicate_arms.py runs/eval-clean-A/score.json runs/eval-clean-B/score.json \
        runs/eval-clean-C/score.json runs/eval-clean-D/score.json

也可以只传两个（做单变量对照）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# 官方评分权重（PLAN.md / docs/roles/README.md）
W_EMOTION = 25.0
W_PROFILE = 15.0
W_RESPONSE = 20.0

# 官方公布的门槛值（用于把指标归一化到 [0,1]）
GATE_BLEU = 0.15
GATE_ROUGE = 0.15
# BERTScore 没有公布门槛。0.80 是「同义改写」的经验下界，0.95 附近是人工改写水平。
BERT_LO, BERT_HI = 0.80, 0.95


def _dig(d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if cur is None:
        return default
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def load_arm(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    metrics = raw.get("metrics") or raw
    rq = metrics.get("response_quality") or {}

    # 画像：优先官方口径字段；没有就退回旧口径并标记（避免再犯「拿旧口径当官方」的错）
    prof_off = _dig(metrics, "profile_accuracy_mean_3dim_official", default=-1.0)
    if prof_off < 0:
        prof_off = _dig(metrics, "profile", "accuracy_mean_3dim_official", default=-1.0)
    prof_old = _dig(metrics, "profile", "accuracy_mean_3dim", default=0.0)
    has_official = prof_off >= 0

    return {
        "tag": path.parent.name,
        "n": int(_dig(metrics, "n_predictions", default=_dig(metrics, "num_samples", default=0))),
        "emotion": _dig(metrics, "emotion_accuracy", default=_dig(metrics, "emotion", "accuracy")),
        "bleu4": _dig(rq, "bleu4", default=_dig(rq, "bleu_4")),
        "rouge_l": _dig(rq, "rouge_l_f", default=_dig(rq, "rouge_l")),
        "bertscore": _dig(rq, "bertscore_f1", default=_dig(metrics, "bertscore", "f1")),
        "bertscore_computed": bool((metrics.get("bertscore") or {}).get("computed")),
        "profile_official": prof_off if has_official else 0.0,
        "profile_old": _dig(metrics, "profile_accuracy_mean_3dim", default=prof_old),
        "has_official_profile": has_official,
        "empty_rate": _dig(rq, "empty_response_rate", default=_dig(metrics, "empty_response_rate", default=0.0)),
        "path": str(path),
    }


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def response_score(arm: Dict[str, Any], scheme: str) -> float:
    """把回应质量归一化到 [0,1]。不同 scheme = 不同加权假设。"""
    b = _clip(arm["bleu4"] / GATE_BLEU)
    r = _clip(arm["rouge_l"] / GATE_ROUGE)
    t = _clip((arm["bertscore"] - BERT_LO) / (BERT_HI - BERT_LO))

    if scheme == "S1_three_equal":
        return (b + r + t) / 3.0
    if scheme == "S2_bleu_rouge":
        return (b + r) / 2.0
    if scheme == "S3_bleu_only":
        return b
    if scheme == "S4_bert_heavy":
        return 0.25 * b + 0.25 * r + 0.5 * t
    raise ValueError(scheme)


def composite(arm: Dict[str, Any], scheme: str) -> float:
    return (
        W_EMOTION * arm["emotion"]
        + W_PROFILE * arm["profile_official"]
        + W_RESPONSE * response_score(arm, scheme)
    )


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    arms = [load_arm(Path(p)) for p in argv[1:]]
    schemes = ["S1_three_equal", "S2_bleu_rouge", "S3_bleu_only", "S4_bert_heavy"]

    # ---- 缺口径的臂直接拒绝，不静默按 0 算 ----
    bad = [a["tag"] for a in arms if not a["has_official_profile"]]
    if bad:
        print(f"⚠️  这些臂的 score.json 里没有官方口径画像字段，按 0 参与计算会很危险：{bad}")
        print("   请先用 src/evaluate.py 重算，或确认字段名。下面的画像列对这些臂不可信。\n")

    print("=" * 108)
    print("原始指标")
    print("=" * 108)
    hdr = f"{'臂':<18}{'n':>6}{'情绪':>9}{'BLEU-4':>10}{'ROUGE-L':>10}{'BERT-F1':>10}{'画像(官方)':>12}{'画像(旧)':>10}{'空回复':>9}"
    print(hdr)
    print("-" * 108)
    for a in arms:
        print(
            f"{a['tag']:<18}{a['n']:>6}{a['emotion']:>9.4f}{a['bleu4']:>10.5f}"
            f"{a['rouge_l']:>10.5f}{a['bertscore']:>10.4f}{a['profile_official']:>12.4f}"
            f"{a['profile_old']:>10.4f}{a['empty_rate']:>8.2%}"
        )

    # ---- 四种加权假设下的裁决 ----
    print()
    print("=" * 108)
    print("四种加权假设下的综合分（情感 25 + 画像 15 + 回应 20，满分 60）")
    print("=" * 108)
    print(f"{'臂':<18}" + "".join(f"{s:>22}" for s in schemes))
    print("-" * 108)
    for a in arms:
        print(f"{a['tag']:<18}" + "".join(f"{composite(a, s):>22.3f}" for s in schemes))

    print()
    winners = set()
    for s in schemes:
        best = max(arms, key=lambda a: composite(a, s))
        winners.add(best["tag"])
        runner = sorted(arms, key=lambda a: composite(a, s), reverse=True)[1]
        gap = composite(best, s) - composite(runner, s)
        print(f"{s:<20} 赢家 {best['tag']:<16} 领先 {gap:+.3f} 分（第二名 {runner['tag']}）")

    print()
    if len(winners) == 1:
        print(f"✅ 结论稳健：四种加权假设都指向 {winners.pop()}。发这一臂。")
        return 0
    print(f"⚠️  结论依赖加权假设，赢家集合 = {sorted(winners)}。")
    print("   处置原则：官方公式未公布时，不选「在某种假设下最高」的臂，")
    print("   而选「在所有假设下都不差」的那一个（maximin）。下面是 maximin 结果：")
    scores = {a["tag"]: [composite(a, s) for s in schemes] for a in arms}
    worst = {t: min(v) for t, v in scores.items()}
    pick = max(worst, key=worst.get)
    for t in sorted(worst, key=worst.get, reverse=True):
        print(f"     最差假设下 {t:<18} {worst[t]:>7.3f}")
    print(f"   ⇒ maximin 选择：{pick}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
