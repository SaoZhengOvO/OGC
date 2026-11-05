"""`EVALUATION_MAX_NEW_TOKENS` 覆盖的用例。

为什么需要这个开关：CI 里的离线自检在纯 CPU 上跑真模型，不设上限时一条样本
实测 **928 秒**（2026-11-05 的构建 run `42043416912`），3 条样例必然撑爆 1800 秒
超时，于是自检拿不到产物、整条构建判红。压到几十个 token 后每条约 1 分钟，
「能不能离线起、产物合不合规」这件事照样验得到。

这个覆盖只服务于自检：官方评测不会设这个变量，所以线上行为不受影响。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "participant"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_inference  # noqa: E402

_BASE_CONFIG = {
    "backend": "transformers",
    "model_path": "./model",
    "performance_rounds": 100,
    "max_new_tokens": 1200,
    "hardware_label": "unspecified",
    "generation": {
        "max_new_tokens": 1200,
        "do_sample": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    },
}


def _clean_env(**overrides: str) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("EVALUATION_")
    }
    environment.update(overrides)
    return environment


class ConfigEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config_path = pathlib.Path(self._tmp.name) / "inference_config.json"
        self.config_path.write_text(json.dumps(_BASE_CONFIG), encoding="utf-8")
        patcher = mock.patch.object(run_inference, "CONFIG_PATH", self.config_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _load(self, **overrides: str) -> dict:
        with mock.patch.dict(
            os.environ, _clean_env(**overrides), clear=True
        ):
            return run_inference.load_config()

    def test_未设置时保持配置文件里的值(self) -> None:
        config = self._load()
        self.assertEqual(config["generation"]["max_new_tokens"], 1200)
        self.assertEqual(config["generation"]["do_sample"], False)

    def test_设置后覆盖且不动其它生成参数(self) -> None:
        config = self._load(EVALUATION_MAX_NEW_TOKENS="96")
        self.assertEqual(config["generation"]["max_new_tokens"], 96)
        # 只改上限，其余解码参数必须原样保留 —— 否则自检验的就不是同一条链路了
        self.assertEqual(config["generation"]["do_sample"], False)
        self.assertEqual(config["generation"]["temperature"], 0.7)
        self.assertEqual(config["generation"]["top_p"], 0.9)
        self.assertEqual(config["generation"]["repetition_penalty"], 1.05)

    def test_空字符串视为未设置(self) -> None:
        config = self._load(EVALUATION_MAX_NEW_TOKENS="   ")
        self.assertEqual(config["generation"]["max_new_tokens"], 1200)

    def test_非整数与非正数都报错(self) -> None:
        for bad in ("abc", "0", "-5", "9.5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self._load(EVALUATION_MAX_NEW_TOKENS=bad)

    def test_缺_generation_段时仍按配置校验报错(self) -> None:
        broken = dict(_BASE_CONFIG)
        broken.pop("generation")
        self.config_path.write_text(json.dumps(broken), encoding="utf-8")
        # 没有 generation 段时配置文件本身就不合法，这里只确认注入逻辑不吞掉这个错误
        with self.assertRaises(ValueError):
            self._load(EVALUATION_MAX_NEW_TOKENS="96")


if __name__ == "__main__":
    unittest.main()
