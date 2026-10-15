"""`tools/eval_loop.py` 选脚本 / 探接线层 / 台账迁移的回归测试。

这三件事都属于「同一把尺子」：跑错脚本、探不到提示词、台账错位，都会让台账上的
数字看起来可比，其实不可比。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_loop  # noqa: E402  （要先把 tools/ 挂进 sys.path）


class TestResolveScript(unittest.TestCase):
    """默认必须是我们真正会提交的那棵树，不是官方原件。"""

    def test_presets(self):
        self.assertEqual(
            eval_loop.resolve_inference_script("wired"),
            eval_loop.WIRED_PARTICIPANT / "run_inference.py",
        )
        self.assertEqual(
            eval_loop.resolve_inference_script("official"),
            eval_loop.OFFICIAL_PARTICIPANT / "run_inference.py",
        )

    def test_default_is_the_wired_tree(self):
        self.assertEqual(
            eval_loop.INFERENCE_SCRIPT,
            eval_loop.WIRED_PARTICIPANT / "run_inference.py",
        )
        parser_default = eval_loop.build_parser().get_default("inference_script")
        self.assertEqual(parser_default, "wired")

    def test_preset_is_case_and_space_insensitive(self):
        self.assertEqual(
            eval_loop.resolve_inference_script("  WIRED "),
            eval_loop.WIRED_PARTICIPANT / "run_inference.py",
        )

    def test_relative_path_is_repo_root_relative(self):
        self.assertEqual(
            eval_loop.resolve_inference_script("deploy/participant/run_inference.py"),
            eval_loop.WIRED_PARTICIPANT / "run_inference.py",
        )

    def test_absolute_path_is_untouched(self):
        absolute = (REPO_ROOT / "tools" / "eval_loop.py").resolve()
        self.assertEqual(eval_loop.resolve_inference_script(str(absolute)), absolute)


class TestScriptKind(unittest.TestCase):
    def test_known_trees(self):
        self.assertEqual(eval_loop.script_kind(eval_loop.SCRIPT_PRESETS["wired"]), "wired")
        self.assertEqual(eval_loop.script_kind(eval_loop.SCRIPT_PRESETS["official"]), "official")

    def test_missing_script(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(eval_loop.script_kind(Path(tmp) / "nope.py"), "missing")

    def test_unknown_copy_is_detected_by_content(self):
        with TemporaryDirectory() as tmp:
            copy = Path(tmp) / "run_inference.py"
            copy.write_text(
                eval_loop.SCRIPT_PRESETS["wired"].read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            self.assertEqual(eval_loop.script_kind(copy), "wired")

    def test_official_original_really_has_no_wiring(self):
        """官方原件里不能有接线层的 import，否则「官方裸基线」这个对照就名不副实。"""
        text = eval_loop.SCRIPT_PRESETS["official"].read_text(encoding="utf-8")
        self.assertNotIn("import wiring", text)


class TestProbeWiring(unittest.TestCase):
    """探测必须在子进程里做，否则父进程的 sys.path 会给出假阳性。"""

    def test_reports_team_prompt_block(self):
        info = eval_loop.probe_wiring(eval_loop.DEFAULT_PROMPT_DIR, eval_loop.WIRED_PARTICIPANT)
        if not info.get("available"):
            self.skipTest(f"接线层探测不可用：{info.get('reason')}")
        self.assertIn("emotion_profile", info["prompt_blocks_present"])
        self.assertEqual(info["rag_fusion"], "ok")
        self.assertEqual(Path(info["prompt_dir"]), eval_loop.DEFAULT_PROMPT_DIR)

    def test_empty_prompt_dir_reports_none(self):
        """空提示词目录不能报成「加载了」—— 那正是静默退化成官方基线的样子。"""
        with TemporaryDirectory() as tmp:
            info = eval_loop.probe_wiring(Path(tmp), eval_loop.WIRED_PARTICIPANT)
        if not info.get("available"):
            self.skipTest(f"接线层探测不可用：{info.get('reason')}")
        self.assertEqual(info["prompt_blocks_present"], [])

    def test_official_tree_has_no_wiring_layer(self):
        info = eval_loop.probe_wiring(eval_loop.DEFAULT_PROMPT_DIR, eval_loop.OFFICIAL_PARTICIPANT)
        self.assertFalse(info["available"])


class TestLeaderboardMigration(unittest.TestCase):
    """台账加列之后，老行必须按列名搬家，不能整体错位。"""

    OLD_HEADER = ("time", "tag", "backend", "model", "n_samples", "n_comparable", "emotion_acc")

    def _write(self, path: Path, header, rows) -> None:
        lines = ["\t".join(header)] + ["\t".join(row) for row in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    def test_old_rows_are_padded_not_shifted(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaderboard.tsv"
            self._write(
                path, self.OLD_HEADER, [["t1", "smoke", "transformers", "config", "5", "5", "0.6"]]
            )
            self.assertTrue(
                eval_loop.migrate_leaderboard_header(path, eval_loop.LEADERBOARD_COLUMNS)
            )
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0].split("\t"), list(eval_loop.LEADERBOARD_COLUMNS))
            row = dict(zip(lines[0].split("\t"), lines[1].split("\t")))
            self.assertEqual(row["emotion_acc"], "0.6")  # 老列按名字搬过去
            self.assertEqual(row["tag"], "smoke")
            self.assertEqual(row["script"], "n/a")  # 新列补 n/a
            self.assertEqual(row["prompts"], "n/a")

    def test_migration_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaderboard.tsv"
            self._write(
                path, self.OLD_HEADER, [["t1", "smoke", "transformers", "config", "5", "5", "0.6"]]
            )
            eval_loop.migrate_leaderboard_header(path, eval_loop.LEADERBOARD_COLUMNS)
            once = path.read_text(encoding="utf-8")
            self.assertFalse(
                eval_loop.migrate_leaderboard_header(path, eval_loop.LEADERBOARD_COLUMNS)
            )
            self.assertEqual(path.read_text(encoding="utf-8"), once)

    def test_empty_file_is_left_alone(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaderboard.tsv"
            path.write_text("", encoding="utf-8")
            self.assertFalse(
                eval_loop.migrate_leaderboard_header(path, eval_loop.LEADERBOARD_COLUMNS)
            )

    def test_append_writes_header_once_then_appends(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaderboard.tsv"
            original = eval_loop.LEADERBOARD
            eval_loop.LEADERBOARD = path
            try:
                eval_loop.append_leaderboard({"tag": "a", "script": "wired", "prompts": "(none)"})
                eval_loop.append_leaderboard({"tag": "b", "script": "official"})
            finally:
                eval_loop.LEADERBOARD = original
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0].split("\t"), list(eval_loop.LEADERBOARD_COLUMNS))
            self.assertEqual(lines[1].split("\t")[1], "a")
            self.assertEqual(lines[1].split("\t")[4], "wired")
            self.assertEqual(lines[2].split("\t")[1], "b")


if __name__ == "__main__":
    unittest.main()
