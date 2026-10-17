#!/usr/bin/env python3
"""给已有的 submission.jsonl 补上画像兜底，不需要重跑模型。

## 为什么需要「事后补算」这件事

画像兜底（`apply_profile_floor`）只改预测里的三个画像字段，**不碰 response、
不碰 emotion_label**。所以它对 BLEU / ROUGE / BERTScore / 情绪准确率没有任何影响，
只影响画像三项准确率。既然如此，就没必要为了评估兜底再跑一遍 300 条推理
（每臂 75 分钟，四臂 5 小时）——直接对着已有的 `submission.jsonl` 补算即可，
得到的结果与「运行时开着兜底跑出来」**逐字节等价**。

等价性来自一个事实：`run_inference.py` 里的兜底发生在
`normalize_prediction()` 内、写盘之前，是对 prediction dict 的纯函数变换。
事后对同一个 dict 施加同一个函数，结果相同。

## 为什么这件事不能省

2026-10-17 的四臂对照是**在没装兜底的盒子上**跑的（盒子 checkout 是 `97ae406`，
早于兜底那几行），而**镜像里是开着的**（`PROFILE_FLOOR_DEFAULTS` 默认生效）。
不补算的话，A/B 的画像对比描述的是一个我们不发货的系统——这正是「测量环境与
交付环境不一致」这一类错误的又一次复发，必须显式消掉。

## 用法

    python3 tools/apply_profile_floor.py \
        --submission runs/eval-clean-A/submission.jsonl \
        --out runs/eval-clean-A/submission.floored.jsonl

    # 再看分（--skip-inference 不跑模型）
    python3 tools/eval_loop.py --tag clean-A-floor --skip-inference \
        --predictions runs/eval-clean-A/submission.floored.jsonl \
        --references runs/slice300/val_references.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from label_space import normalize_profile  # noqa: E402

# 与 deploy/participant/run_inference.py 的 PROFILE_FLOOR_DEFAULTS 保持一致。
# 这里刻意再写一遍而不是 import：run_inference.py 会 import wiring / torch，
# 在评分机上跑不动；两边一致性由 tests/test_profile_floor_sync.py 守住。
PROFILE_FLOOR_DEFAULTS = {
    "personality_traits": "sensitive",
    "interests": "career_development",
    "style": "rational",
}


def apply_profile_floor(profile: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """只补空维度，非空维度原样保留。"""
    filled = dict(profile)
    for name, prior in PROFILE_FLOOR_DEFAULTS.items():
        if filled.get(name):
            continue
        filled[name] = [prior]
    return filled


# submission.jsonl 里画像字段叫什么，由 run_inference.py 决定。
# 2026-10-17 的教训：这里曾写死 "profile"，而真实字段是 "user_profile"，
# 于是补算写了一整行看不见的键、评分器读不到，A 臂「补兜底前后分数一模一样」，
# 我一度以为兜底无效。两个键都认，读哪个就写哪个。
_PROFILE_KEYS = ("user_profile", "profile")


def _profile_key(row: Dict[str, Any]) -> str:
    for key in _PROFILE_KEYS:
        if key in row:
            return key
    return _PROFILE_KEYS[0]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", required=True, nargs="+")
    parser.add_argument("--out", default="", help="单文件时指定输出；多文件时自动加 .floored 后缀")
    parser.add_argument("--stats", action="store_true", help="打印补了多少维")
    args = parser.parse_args(argv[1:])

    single = len(args.submission) == 1
    for src in args.submission:
        path = Path(src)
        out = Path(args.out) if (single and args.out) else path.with_suffix(".floored.jsonl")

        rows: Dict[str, Dict[str, Any]] = {}
        dupes = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(row.get("id") or "")
                if not sid:
                    continue
                if sid in rows:
                    dupes += 1
                rows[sid] = row  # 保留最后一次

        filled_dims = 0
        total_dims = 0
        for row in rows.values():
            key = _profile_key(row)
            profile = normalize_profile(row.get(key) or {})
            before = {d: bool(profile.get(d)) for d in PROFILE_FLOOR_DEFAULTS}
            profile = apply_profile_floor(profile)
            total_dims += len(PROFILE_FLOOR_DEFAULTS)
            filled_dims += sum(1 for d, had in before.items() if not had)
            row[key] = profile

        with out.open("w", encoding="utf-8") as handle:
            for sid, row in rows.items():
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        msg = f"{path.name} -> {out.name}  {len(rows)} 条"
        if dupes:
            msg += f"（丢弃 {dupes} 行重复 id）"
        if args.stats:
            msg += f"  补空 {filled_dims}/{total_dims} 维 = {filled_dims / max(1, total_dims):.2%}"
        print(msg, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
