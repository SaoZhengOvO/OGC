"""量化「解析层抢救正则」那个修法值多少分。

背景（issue #3 第 4 条评论）：
`recover_response_text` 的抢救正则要求键**带引号**（`"response_text"`），
而失败形态是无引号的 `response_text: …` → 匹配不上 → 整段原文（含前缀与 user_profile 尾巴）
被当回复提交。

这个脚本拿已有产物做**离线对照**：
  1. 找出被污染的回复（以 `response_text:` 开头）
  2. 造一份"剥掉格式残留"的变体
  3. 两份都过一遍 src/cli.py eval，比 BLEU-4 / ROUGE-L

这样就能回答"那条修法值多少"，而不是凭感觉说"值得修"。

用法（在 box 上跑）::

    python tools/emotion_profile/measure_recovery_gain.py --tag mi300x-v4-smoke
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 回复开头形如 `response_text:` / `"response_text":` / `response_text：`
PREFIX = re.compile(r'^\s*"?response_text"?\s*[:：]\s*')
#: 尾巴形如 `user_profile: [...]`（以及其后的一切）
TAIL = re.compile(r'\n?\s*"?user_profile"?\s*[:：]', re.S)


def strip_pollution(text: str) -> str:
    """剥掉抢救回来的文本里混入的字段名与尾部片段。"""
    cleaned = PREFIX.sub("", text)
    cleaned = TAIL.split(cleaned)[0]
    return cleaned.strip()


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(predictions: Path, references: Path, out: Path) -> dict:
    """调 src/cli.py eval 打分。"""
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "cli.py"), "eval",
         "--predictions", str(predictions), "--references", str(references), "--json-out", str(out)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if completed.returncode != 0:
        raise SystemExit(f"打分失败：{completed.stdout[-500:]}{completed.stderr[-500:]}")
    return json.loads(out.read_text(encoding="utf-8"))["metrics"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="量化解析层抢救正则的修法价值")
    parser.add_argument("--tag", required=True, help="被污染的评测 tag（runs/<tag>）")
    parser.add_argument("--references", default=None, help="真值文件（默认用该目录的 val_references_subset.jsonl）")
    args = parser.parse_args(argv)

    run_dir = REPO_ROOT / "runs" / args.tag
    submission = run_dir / "submission.jsonl"
    references = Path(args.references) if args.references else run_dir / "val_references_subset.jsonl"
    if not submission.exists() or not references.exists():
        raise SystemExit(f"缺 {submission} 或 {references}")

    rows = read_jsonl(submission)
    polluted = [row for row in rows if PREFIX.match(row.get("response_text") or "")]
    print(f"{args.tag}：{len(rows)} 条，其中被污染的 {len(polluted)} 条"
          f"（{len(polluted) / len(rows):.2%}）")
    if not polluted:
        print("没有被污染的回复 —— 这次运行不是我们要找的形态。")
        return 0

    for row in polluted[:3]:
        original = row["response_text"]
        print(f"\n  id={row['id']}")
        print(f"    原始（被污染）: {original[:100]!r}")
        print(f"    剥掉残留后    : {strip_pollution(original)[:100]!r}")

    cleaned_rows = []
    for row in rows:
        text = row.get("response_text") or ""
        cleaned_rows.append({**row, "response_text": strip_pollution(text) if PREFIX.match(text) else text})
    cleaned_path = run_dir / "submission.cleaned.jsonl"
    with cleaned_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in cleaned_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    before = score(submission, references, run_dir / "score.before-clean.json")
    after = score(cleaned_path, references, run_dir / "score.after-clean.json")

    print(f"\n{'指标':<26}{'现状':>10}{'剥掉残留':>11}{'差':>12}")
    print("-" * 62)
    for label, key in (("BLEU-4", "bleu4"), ("ROUGE-L", "rouge_l_f")):
        a, b = before["response_quality"][key], after["response_quality"][key]
        print(f"{label:<26}{a:>10.4f}{b:>11.4f}{b - a:>+12.4f}")
    for label, key in (("情绪 Acc", "emotion_accuracy"), ("画像 Acc(三维均值)", "profile_accuracy_mean_3dim")):
        print(f"{label:<26}{before[key]:>10.4f}{after[key]:>11.4f}{after[key] - before[key]:>+12.4f}")

    print()
    print("读法：情绪/画像在两次之间**不应变化**（解析层没动它们）——若变了说明我做错了对照。")
    print("      回复两项的差值就是「抢救正则修好」能拿回来的分（在这条 tag 上）。")
    print("      注意：这只量化了回复那一项；情绪与画像的损失要靠提示词侧减少畸形输出（v8）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
