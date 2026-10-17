"""画像输出策略（floor / force / off）的用例。

背景：官方口径下画像留空必判 0 分（`GivenData/participant/README.md:100-102`），
所以「交空数组」不是保守，是直接放弃这一项。`floor` 只补空、`force` 一律常量。

`force` 的正当性来自实测，不是偏好：2026-10-17 四臂数据里，模型真正给出非空
标签的那部分样本命中率低于拿常量去撞（B 臂 0.0667/0.0467/0.1533 vs 常量
0.1233/0.0367/0.1967）。所以需要能把两种策略都跑出来对照。
"""

from __future__ import annotations

import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "participant"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_inference  # noqa: E402


def _clean_env() -> dict[str, str]:
    """去掉两个相关环境变量后的环境副本，避免宿主环境影响用例。"""
    env = dict(os.environ)
    env.pop("PARTICIPANT_PROFILE_MODE", None)
    env.pop("PARTICIPANT_PROFILE_FLOOR", None)
    return env


class ProfileModeResolutionTests(unittest.TestCase):
    def _mode(self, **env: str) -> str:
        with mock.patch.dict(os.environ, env, clear=True):
            return run_inference.profile_mode()

    def test_default_is_floor(self) -> None:
        self.assertEqual(self._mode(), "floor")

    def test_explicit_force(self) -> None:
        self.assertEqual(self._mode(PARTICIPANT_PROFILE_MODE="force"), "force")
        self.assertEqual(self._mode(PARTICIPANT_PROFILE_MODE="FORCE"), "force")
        self.assertEqual(self._mode(PARTICIPANT_PROFILE_MODE=" constant "), "force")

    def test_explicit_off(self) -> None:
        self.assertEqual(self._mode(PARTICIPANT_PROFILE_MODE="off"), "off")
        self.assertEqual(self._mode(PARTICIPANT_PROFILE_MODE="0"), "off")

    def test_legacy_floor_switch_still_works(self) -> None:
        self.assertEqual(self._mode(PARTICIPANT_PROFILE_FLOOR="0"), "off")
        self.assertEqual(self._mode(PARTICIPANT_PROFILE_FLOOR="false"), "off")
        self.assertEqual(self._mode(PARTICIPANT_PROFILE_FLOOR="1"), "floor")

    def test_new_mode_wins_over_legacy_switch(self) -> None:
        """两个都设时以 PARTICIPANT_PROFILE_MODE 为准，避免语义互相打架。"""
        self.assertEqual(
            self._mode(PARTICIPANT_PROFILE_MODE="force", PARTICIPANT_PROFILE_FLOOR="0"),
            "force",
        )
        self.assertEqual(
            self._mode(PARTICIPANT_PROFILE_MODE="off", PARTICIPANT_PROFILE_FLOOR="1"),
            "off",
        )


class ApplyProfileTests(unittest.TestCase):
    KEEP = {"personality_traits": ["anxious"], "interests": ["health"], "style": ["emotional"]}

    def test_floor_keeps_model_labels(self) -> None:
        self.assertEqual(run_inference.apply_profile_floor(self.KEEP), self.KEEP)

    def test_force_overwrites_model_labels(self) -> None:
        forced = run_inference.apply_profile_constant(self.KEEP)
        self.assertNotEqual(forced, self.KEEP)
        for name, prior in run_inference.PROFILE_FLOOR_DEFAULTS.items():
            self.assertEqual(forced[name], [prior], f"{name} 没被换成常量")

    def test_both_modes_fill_empty_dimensions(self) -> None:
        empty = {"personality_traits": [], "interests": [], "style": []}
        floor = run_inference.apply_profile_floor(empty)
        force = run_inference.apply_profile_constant(empty)
        self.assertEqual(floor, force, "全空时两种策略应当等价")
        for name, prior in run_inference.PROFILE_FLOOR_DEFAULTS.items():
            self.assertEqual(floor[name], [prior])

    def test_no_dimension_is_left_empty(self) -> None:
        """无论模型交什么，两个策略都不能留下空数组——空数组等于 0 分。"""
        for profile in (self.KEEP, {"personality_traits": [], "interests": ["career_development"], "style": []}, {}):
            for fn in (run_inference.apply_profile_floor, run_inference.apply_profile_constant):
                out = fn(dict(profile))
                for name in run_inference.PROFILE_FLOOR_DEFAULTS:
                    self.assertTrue(out.get(name), f"{fn.__name__} 留下了空的 {name}")


class ToolParityTests(unittest.TestCase):
    """补算工具的两个策略必须和发货代码同语义。"""

    def test_tool_constant_matches_deployed(self) -> None:
        from tools.apply_profile_floor import apply_profile_constant as tool_force

        sample = {"personality_traits": ["anxious"], "interests": [], "style": ["emotional"]}
        self.assertEqual(tool_force(dict(sample)), run_inference.apply_profile_constant(dict(sample)))

    def test_tool_floor_matches_deployed(self) -> None:
        from tools.apply_profile_floor import apply_profile_floor as tool_floor

        sample = {"personality_traits": ["anxious"], "interests": [], "style": ["emotional"]}
        self.assertEqual(tool_floor(dict(sample)), run_inference.apply_profile_floor(dict(sample)))


if __name__ == "__main__":
    unittest.main()
