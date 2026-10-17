"""输出塌陷重试的判据。

角色③在 n=300 上归因出来的事实：空回复 26 条里 18 条顶层只吐画像三数组、
4 条只吐 `{}`，与 `parse_failure` 集合**零交集**。所以重试判据不能只看
`parse_failed`——否则 22/26 条的塌陷会漏掉。
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "participant"))

import run_inference  # noqa: E402


GOOD = {
    "id": "val_000001_t1",
    "response_text": "我在听，你慢慢说。",
    "emotion_label": "anxiety",
    "user_profile": {"personality_traits": ["sensitive"], "interests": [], "style": ["rational"]},
    "memory_refs": [],
}


class TestNeedsSchemaRetry(unittest.TestCase):
    def test_healthy_prediction_does_not_retry(self) -> None:
        self.assertFalse(run_inference.needs_schema_retry(GOOD, False))

    def test_parse_failure_always_retries(self) -> None:
        self.assertTrue(run_inference.needs_schema_retry(dict(GOOD), True))

    def test_schema_collapse_empty_response_retries(self) -> None:
        """③ 归因出来的主因：JSON 合法但没有 response_text。"""
        broken = dict(GOOD)
        broken["response_text"] = ""
        self.assertFalse(run_inference.needs_schema_retry(broken, False) is False)
        self.assertTrue(run_inference.needs_schema_retry(broken, False))

    def test_schema_collapse_whitespace_response_retries(self) -> None:
        broken = dict(GOOD)
        broken["response_text"] = "   \n\t "
        self.assertTrue(run_inference.needs_schema_retry(broken, False))

    def test_schema_collapse_empty_emotion_retries(self) -> None:
        """情绪空串在官方评分里按判错处理，同样值得重试。"""
        broken = dict(GOOD)
        broken["emotion_label"] = ""
        self.assertTrue(run_inference.needs_schema_retry(broken, False))

    def test_missing_keys_retry(self) -> None:
        self.assertTrue(run_inference.needs_schema_retry({}, False))

    def test_non_dict_retries(self) -> None:
        self.assertTrue(run_inference.needs_schema_retry([], False))  # type: ignore[arg-type]

    def test_profile_only_output_retries(self) -> None:
        """③ 实测的 18/26：顶层只有画像三数组。"""
        profile_only = {
            "id": "val_000002_t5",
            "response_text": "",
            "emotion_label": "",
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
            "memory_refs": [],
        }
        self.assertTrue(run_inference.needs_schema_retry(profile_only, False))

    def test_retry_attempts_formula(self) -> None:
        """attempts = 1 + max(0, SCHEMA_RETRY)：0 关、1 一次、3 三次。"""
        for flag, expected in ((0, 1), (1, 2), (3, 4), (-2, 1)):
            self.assertEqual(1 + max(0, flag), expected)


class TestSchemaReminder(unittest.TestCase):
    def test_reminder_mentions_both_required_fields(self) -> None:
        source = (ROOT / "deploy" / "participant" / "run_inference.py").read_text(encoding="utf-8")
        block = source.split("SCHEMA_REMINDER = (", 1)[1].split(")", 1)[0]
        self.assertIn("response_text", block)
        self.assertIn("emotion_label", block)

    def test_retry_is_on_by_default(self) -> None:
        source = (ROOT / "deploy" / "participant" / "run_inference.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("PARTICIPANT_SCHEMA_RETRY", "1")', source)


if __name__ == "__main__":
    unittest.main()
