"""`tools/eval_loop.py::dedupe_submission` 的回归测试。

这段逻辑不是洁癖：2026-10-16 那次两个 run_inference 并发写同一个 submission.jsonl，
2545 行里只剩 2354 个唯一 id、164 个 id 的重复行内容还互相冲突，情绪分从 0.4146
掉到 0.1619，而 `src/cli.py validate` 只查 schema，一个字都没报。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.eval_loop import dedupe_submission, read_jsonl, write_jsonl  # noqa: E402


def _row(sample_id: str, text: str = "好的", emotion: str = "anxiety") -> dict:
    return {
        "id": sample_id,
        "response_text": text,
        "emotion_label": emotion,
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "memory_refs": [],
    }


def test_clean_file_is_untouched(tmp_path: Path) -> None:
    path = tmp_path / "submission.jsonl"
    write_jsonl(path, [_row("a"), _row("b")])
    before = path.read_bytes()

    stats = dedupe_submission(path)

    assert stats == {"rows_total": 2, "unique_ids": 2, "duplicates": 0, "conflicting": 0}
    assert path.read_bytes() == before, "没有重复就不该重写文件"


def test_duplicates_are_dropped_keeping_the_last_one(tmp_path: Path) -> None:
    path = tmp_path / "submission.jsonl"
    write_jsonl(path, [_row("a", "旧的"), _row("b"), _row("a", "新的")])

    stats = dedupe_submission(path)

    assert stats["rows_total"] == 3
    assert stats["unique_ids"] == 2
    assert stats["duplicates"] == 1
    assert stats["conflicting"] == 1, "两次内容不同，要算冲突"

    kept = list(read_jsonl(path))
    assert [row["id"] for row in kept] == ["a", "b"], "保持首次出现的顺序"
    assert kept[0]["response_text"] == "新的", "追加写时后写的才是完整那条"


def test_identical_duplicates_are_not_counted_as_conflicting(tmp_path: Path) -> None:
    path = tmp_path / "submission.jsonl"
    write_jsonl(path, [_row("a"), _row("a")])

    stats = dedupe_submission(path)

    assert stats["duplicates"] == 1
    assert stats["conflicting"] == 0


def test_backup_keeps_the_original_bytes(tmp_path: Path) -> None:
    path = tmp_path / "submission.jsonl"
    write_jsonl(path, [_row("a", "旧"), _row("a", "新")])
    original = path.read_bytes()

    dedupe_submission(path)

    backup = path.with_suffix(path.suffix + ".dedupe.bak")
    assert backup.exists(), "去重前必须留一份原始文件"
    assert backup.read_bytes() == original


def test_missing_file_is_a_noop(tmp_path: Path) -> None:
    stats = dedupe_submission(tmp_path / "nope.jsonl")
    assert stats["rows_total"] == 0
    assert stats["duplicates"] == 0


def test_truncated_line_surfaces_instead_of_being_silently_skipped(tmp_path: Path) -> None:
    """半截行是并发写的另一种症状，不该被当成「空行」吞掉。"""
    path = tmp_path / "submission.jsonl"
    path.write_text(
        json.dumps(_row("a"), ensure_ascii=False) + "\n" + '{"id": "b", "response_',
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        dedupe_submission(path)
