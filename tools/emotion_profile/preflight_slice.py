#!/usr/bin/env python3
"""预检「切片 + eval_loop --source/--references」这条决策路径（纯 CPU，不占卡卡位）。

为什么要有它：`tools/emotion_profile/make_eval_slice.py` 造出的切片要靠
`eval_loop.py --source/--references` 喂进去，这条参数组合之前没用过。
不先验证就会在 GPU 卡位上跑出一个报错、白占一轮。本脚本只做输入构造与打分两步。

它同时给出切片上的**零成本下限**，也就是任何候选都应当超过的那条线：

    情绪：一律答多数类（anxiety）
    画像：三维一律留空

用法::

    python tools/emotion_profile/preflight_slice.py            # 默认 runs/slice200
    python tools/emotion_profile/preflight_slice.py --slice runs/slice400
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_KEYS = {"id", "conversation_id", "target_user_turn_id", "history"}


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(predictions: Path, references: Path, out: Path) -> dict:
    """调 src/cli.py eval 打分，返回 metrics。"""
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "cli.py"),
            "eval",
            "--predictions",
            str(predictions),
            "--references",
            str(references),
            "--json-out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        raise SystemExit(f"打分失败，退出码 {completed.returncode}")
    return json.loads(out.read_text(encoding="utf-8"))["metrics"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="预检切片决策路径")
    parser.add_argument("--slice", default="runs/slice200", help="切片目录（相对仓库根）")
    args = parser.parse_args(argv)

    slice_dir = Path(args.slice)
    if not slice_dir.is_absolute():
        slice_dir = REPO_ROOT / slice_dir
    samples = slice_dir / "val_samples.jsonl"
    references = slice_dir / "val_references.jsonl"
    if not samples.exists() or not references.exists():
        raise SystemExit(f"切片不存在，请先跑 make_eval_slice.py（缺 {samples} 或 {references}）")

    # 1) 输入构造（复用 eval_loop 自己的函数，保证与正式跑法完全一致）
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import eval_loop  # noqa: PLC0415

    generated = slice_dir / "val_input.jsonl"
    count = eval_loop.build_val_input(samples, generated, None, 0)
    rows = read_jsonl(generated)
    if count != len(rows):
        raise SystemExit(f"build_val_input 返回 {count} 但写出 {len(rows)} 行")
    for row in rows:
        if set(row) != INPUT_KEYS:
            raise SystemExit(f"输入里出现了非预期字段：{set(row)}（答案可能泄漏）")
        if not row["history"]:
            raise SystemExit(f"{row['id']} 的 history 为空")
    print(f"[1] 输入构造 {count} 条，只含 {sorted(INPUT_KEYS)} —— 无答案泄漏 ✅")

    # 2) id 顺序必须与真值一一对应
    reference_ids = [row["id"] for row in read_jsonl(references)]
    if [row["id"] for row in rows] != reference_ids:
        raise SystemExit("输入顺序与真值顺序不一致")
    print(f"[2] {len(reference_ids)} 条 id 与真值顺序一致 ✅")

    # 3) 完美预测应得 1.0000（证明打分链路自洽）
    perfect = slice_dir / "preflight_perfect.jsonl"
    with perfect.open("w", encoding="utf-8", newline="\n") as handle:
        for record in read_jsonl(references):
            target = record["target"]
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "response_text": target["response_text"],
                        "emotion_label": target["emotion_label"],
                        "user_profile": target["user_profile"],
                        "memory_refs": [],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    metrics = score(perfect, references, slice_dir / "preflight_perfect_score.json")
    print(
        f"[3] 完美预测：n={metrics['n_comparable']} "
        f"情绪 {metrics['emotion_accuracy']:.4f} 画像 {metrics['profile_accuracy_mean_3dim']:.4f} ✅"
    )
    if abs(metrics["emotion_accuracy"] - 1.0) > 1e-9 or abs(metrics["profile_accuracy_mean_3dim"] - 1.0) > 1e-9:
        raise SystemExit("完美预测没有拿到 1.0000，打分链路有问题")

    # 4) 零成本下限（任何候选都该超过它）
    floor = slice_dir / "preflight_floor.jsonl"
    with floor.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "response_text": "嗯，我在听。",
                        "emotion_label": "anxiety",
                        "user_profile": {"personality_traits": [], "interests": [], "style": []},
                        "memory_refs": [],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    floor_metrics = score(floor, references, slice_dir / "preflight_floor_score.json")
    print(
        f"[4] 零成本下限：情绪 {floor_metrics['emotion_accuracy']:.4f} "
        f"（一律答 anxiety）/ 画像 {floor_metrics['profile_accuracy_mean_3dim']:.4f}（三维留空）"
    )
    for dim, value in floor_metrics["profile_dimension_scores"].items():
        print(f"      {dim:20s} 留空得分 {value['exact_match_acc']:.4f}")
    print("\n候选的合格线 = 同时超过这两个下限。预检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
