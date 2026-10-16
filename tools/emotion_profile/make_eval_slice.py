"""从 val 里取一个**等距**切片，作为"比 n=20 可信、又比 n=2354 便宜"的决策集。

为什么需要它（有数据支撑的动机，见 tools/emotion_profile/LOG.md 第三轮）：

`python3 tools/eval_loop.py --limit 20` 取的是 val 的**前 20 条**，而它严重不代表全量：

| 切片 | 空p | 空i | 空s | 「一律留空」得分 |
|---|---|---|---|---|
| 前 20（smoke） | 0.6000 | 0.6500 | **0.2000** | **0.4833** |
| 全量 2354 | 0.6185 | 0.7489 | **0.5174** | **0.6283** |

style 的金标为空率差了 0.317 —— 「必须填 style」这个结论在 smoke 上的理由强度是全量的 2.6 倍。
val 内部也有位置漂移（中段 200 条的 personality 空率只有 0.39）。

所以在 smoke 上达标的规则，不一定在全量上不掉分。**等距切片**按固定步长跨全 val 取样，
对位置无偏，200 条约 8 分钟就能跑完，适合做"要不要把 v4 推上全量"的决策依据。

用法::

    python tools/emotion_profile/make_eval_slice.py --size 200
    # 产出 runs/slice200/{val_samples.jsonl,val_references.jsonl,slice_manifest.json}

    # 然后直接喂给 eval_loop（--source 与 --references 都指到切片上）
    python3 tools/eval_loop.py --tag v4-slice200 \\
        --source runs/slice200/val_samples.jsonl \\
        --references runs/slice200/val_references.jsonl \\
        --inference-script wired --prompt-dir /mnt/workspace/OGC/runs/p02-v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
DIMS = ("personality_traits", "interests", "style")


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def profile_empty_rates(rows: list[dict]) -> dict[str, float]:
    """逐维金标为空率。"""
    total = len(rows)
    return {
        dim: sum(1 for row in rows if not ((row.get("target") or {}).get("user_profile") or {}).get(dim)) / total
        for dim in DIMS
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成等距 val 切片（决策集）")
    parser.add_argument("--size", type=int, default=200, help="切片条数（默认 200）")
    parser.add_argument("--out", default=None, help="输出目录（默认 runs/slice<size>）")
    args = parser.parse_args(argv)

    samples = read_jsonl(PROCESSED / "val_samples.jsonl")
    references = read_jsonl(PROCESSED / "val_references.jsonl")
    reference_by_id = {row["id"]: row for row in references}

    size = min(args.size, len(samples))
    step = len(samples) / size
    indices = [int(index * step) for index in range(size)]
    picked_samples = [samples[index] for index in indices]
    picked_references = [reference_by_id[row["id"]] for row in picked_samples]

    out_dir = Path(args.out) if args.out else REPO_ROOT / "runs" / f"slice{size}"
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in (("val_samples.jsonl", picked_samples), ("val_references.jsonl", picked_references)):
        with (out_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    slice_rates = profile_empty_rates(picked_samples)
    full_rates = profile_empty_rates(samples)
    manifest = {
        "size": len(picked_samples),
        "val_total": len(samples),
        "step": step,
        "sampling": "evenly spaced（对位置无偏）",
        "slice_empty_rates": slice_rates,
        "full_empty_rates": full_rates,
        "abs_diff": {dim: abs(slice_rates[dim] - full_rates[dim]) for dim in DIMS},
        "first_20_empty_rates": profile_empty_rates(samples[:20]),
    }
    (out_dir / "slice_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"切片: {out_dir}（{len(picked_samples)} 条，步长 {step:.2f}，等距取样）")
    print(f"{'维度':<20}{'切片':>9}{'全量':>9}{'|差|':>9}{'前20(对比)':>12}")
    for dim in DIMS:
        print(
            f"{dim:<20}{slice_rates[dim]:>9.4f}{full_rates[dim]:>9.4f}"
            f"{manifest['abs_diff'][dim]:>9.4f}{manifest['first_20_empty_rates'][dim]:>12.4f}"
        )
    worst = max(manifest["abs_diff"].values())
    print(f"\n切片与全量的最大逐维偏差: {worst:.4f}"
          f"（对比：前 20 条 style 的偏差是 "
          f"{abs(manifest['first_20_empty_rates']['style'] - full_rates['style']):.4f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
