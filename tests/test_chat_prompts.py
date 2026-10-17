"""`tools/chat.py` 复制的重试提示词必须与交付脚本逐字一致。

为什么要有这条测试：`SCHEMA_REMINDER` 定义在 `deploy/participant/run_inference.py` 的
`main()` 函数内部，模块级 import 拿不到，所以 `tools/chat.py` 复制了一份。复制就意味着
会漂 —— 哪天有人改了交付脚本的重试提示词，本地聊天看到的就不再是提交上去的行为。
这条测试把两边的字面量钉在一起。
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_INFERENCE = REPO_ROOT / "deploy" / "participant" / "run_inference.py"


def _literal_from_source(path: Path, name: str) -> str:
    """在源码里找 `name = <字符串字面量>`（可以是多段相加），把值拼出来。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        if isinstance(value, str):
            return value
    raise AssertionError(f"{path.name} 里找不到字符串常量 {name}")


def test_schema_reminder_matches_delivery_script() -> None:
    from tools.chat import SCHEMA_REMINDER as chat_copy

    canonical = _literal_from_source(RUN_INFERENCE, "SCHEMA_REMINDER")
    assert chat_copy == canonical, (
        "tools/chat.py 里的 SCHEMA_REMINDER 与 deploy/participant/run_inference.py 不一致，"
        "本地聊天已不再等价于提交行为"
    )
