"""`tools/eval_loop.py` 里真值裁剪的回归测试。

背景：`src/evaluate.py` 的主指标分母刻意取**完整验证集**（少交很多条不该反而分高）。
这个选择对正式提交是对的，但会让 `--limit 20` 的冒烟跑分永远显示约 0.001，
队友根本看不出自己改的东西有没有用。所以部分样本跑分时，真值要一起收窄。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_loop  # noqa: E402  （要先把 tools/ 挂进 sys.path）


def write_lines(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


class WriteReferenceSubsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)

        self.references = self.root / "references.jsonl"
        write_lines(
            self.references,
            [
                {"id": "val_000001_t1", "target": {"response_text": "甲", "emotion_label": "joy"}},
                {"id": "val_000001_t3", "target": {"response_text": "乙", "emotion_label": "sadness"}},
                {"id": "val_000002_t5", "target": {"response_text": "丙", "emotion_label": "anxiety"}},
            ],
        )
        self.scope = self.root / "val_input.jsonl"
        write_lines(
            self.scope,
            [
                {"id": "val_000001_t1", "conversation_id": "val_conv_000001", "history": []},
                {"id": "val_000002_t5", "conversation_id": "val_conv_000002", "history": []},
            ],
        )
        self.out = self.root / "subset.jsonl"

    def test_keeps_only_scoped_ids(self) -> None:
        kept = eval_loop.write_reference_subset(self.references, self.scope, self.out)
        self.assertEqual(kept, 2)
        ids = [row["id"] for row in eval_loop.read_jsonl(self.out)]
        self.assertEqual(ids, ["val_000001_t1", "val_000002_t5"])

    def test_preserves_original_lines_verbatim(self) -> None:
        eval_loop.write_reference_subset(self.references, self.scope, self.out)
        original = [line for line in self.references.read_text(encoding="utf-8").splitlines() if "val_000002_t5" in line]
        self.assertEqual(self.out.read_text(encoding="utf-8").splitlines()[1], original[0])

    def test_raises_when_nothing_matches(self) -> None:
        other = self.root / "other_input.jsonl"
        write_lines(other, [{"id": "test1_000001_t5", "history": []}])
        with self.assertRaises(SystemExit):
            eval_loop.write_reference_subset(self.references, other, self.out)

    def test_raises_when_scope_has_no_ids(self) -> None:
        empty = self.root / "empty.jsonl"
        write_lines(empty, [{"conversation_id": "val_conv_000001"}])
        with self.assertRaises(SystemExit):
            eval_loop.write_reference_subset(self.references, empty, self.out)


class RealDataTest(unittest.TestCase):
    """拿仓库里真实的 val 数据核对一次：20 条的部分跑分应该能对上 20 条真值。"""

    def test_first_twenty_scope_matches_twenty_references(self) -> None:
        source = REPO_ROOT / "data" / "processed" / "val_samples.jsonl"
        references = REPO_ROOT / "data" / "processed" / "val_references.jsonl"
        if not source.is_file() or not references.is_file():
            self.skipTest("还没跑过 python src/cli.py build")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scope = root / "val_input.jsonl"
            eval_loop.build_val_input(source, scope, 20, 0)
            kept = eval_loop.write_reference_subset(references, scope, root / "subset.jsonl")
        self.assertEqual(kept, 20)


if __name__ == "__main__":
    unittest.main()
