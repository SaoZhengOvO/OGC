"""守住两处 PROFILE_FLOOR_DEFAULTS 的一致性。

画像兜底的正本在 `deploy/participant/run_inference.py`（真正发货的那份），
副本在 `tools/apply_profile_floor.py`（事后补算用）。副本是**必须**存在的：
`run_inference.py` 会 import wiring 与 torch，在评分机上根本 import 不动，
不能直接复用。

两份逻辑一旦漂移，后果很隐蔽——事后补算出来的画像分与镜像实际跑出来的不一样，
而两边都不会报错。所以用测试把「先验标签表」钉成同一个东西。

只比对标签表这一份数据，不比对函数体：函数体的语义
（只补空、不覆盖）由 `tools/apply_profile_floor.py` 自己的用例覆盖。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEPLOY_SRC = REPO_ROOT / "deploy" / "participant" / "run_inference.py"
TOOL_SRC = REPO_ROOT / "tools" / "apply_profile_floor.py"


def _extract_dict_literal(path: Path, name: str) -> dict[str, str]:
    """从源码里抠出 ``NAME = {...}`` 这个纯字面量赋值，不 import 模块。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                value = ast.literal_eval(node.value)
                assert isinstance(value, dict), f"{path.name} 里的 {name} 不是 dict"
                return {str(k): str(v) for k, v in value.items()}
    raise AssertionError(f"{path.name} 里找不到 {name} 的字面量赋值")


def test_profile_floor_defaults_are_identical() -> None:
    deployed = _extract_dict_literal(DEPLOY_SRC, "PROFILE_FLOOR_DEFAULTS")
    local = _extract_dict_literal(TOOL_SRC, "PROFILE_FLOOR_DEFAULTS")
    assert deployed == local, (
        "两处 PROFILE_FLOOR_DEFAULTS 漂移了。\n"
        f"  deploy: {deployed}\n"
        f"  tools : {local}\n"
        "改一处必须同步另一处，否则事后补算的画像分会和镜像里跑出来的不一样。"
    )


def test_floor_covers_exactly_the_three_profile_dimensions() -> None:
    from label_space import PROFILE_DIMENSIONS, PROFILE_VALUES

    deployed = _extract_dict_literal(DEPLOY_SRC, "PROFILE_FLOOR_DEFAULTS")
    assert set(deployed) == set(PROFILE_DIMENSIONS), (
        f"兜底覆盖的维度 {sorted(deployed)} 与官方画像维度 {sorted(PROFILE_DIMENSIONS)} 不一致"
    )
    for dim, prior in deployed.items():
        assert prior in PROFILE_VALUES[dim], f"{dim} 的先验 {prior!r} 不在官方标签空间里"


def test_floor_only_fills_empty_dimensions() -> None:
    """只补空、不覆盖——模型判断对的那部分样本必须一个字节都不动。"""
    from tools.apply_profile_floor import apply_profile_floor

    untouched = {
        "personality_traits": ["anxious"],
        "interests": ["health"],
        "style": ["emotional"],
    }
    assert apply_profile_floor(untouched) == untouched

    partly = {"personality_traits": ["anxious"], "interests": [], "style": []}
    filled = apply_profile_floor(partly)
    assert filled["personality_traits"] == ["anxious"], "非空维度被覆盖了"
    assert filled["interests"] == ["career_development"]
    assert filled["style"] == ["rational"]

    empty = {"personality_traits": [], "interests": [], "style": []}
    assert apply_profile_floor(empty) == {
        "personality_traits": ["sensitive"],
        "interests": ["career_development"],
        "style": ["rational"],
    }


def test_tool_reads_the_same_profile_key_as_the_deployed_writer() -> None:
    """补算工具读写的字段名，必须和 run_inference.py 真正写盘的那个一致。

    2026-10-17 的实际事故：工具写死 ``profile``，而 ``run_inference.py:346``
    写的是 ``"user_profile"``。补算于是往每行塞了一个评分器不认识的键，
    改前改后分数逐字段相同，看起来像「兜底没有效果」，实际是补算打空了。
    这类错误没有任何报错，只能靠测试钉住。
    """
    from tools.apply_profile_floor import _PROFILE_KEYS

    source = DEPLOY_SRC.read_text(encoding="utf-8")
    # run_inference.py 里真正被 dump 进 submission.jsonl 的键
    assert '"user_profile": clean_profile' in source, (
        "run_inference.py 写盘的画像字段变了，请同步 tools/apply_profile_floor.py 的 _PROFILE_KEYS"
    )
    assert _PROFILE_KEYS[0] == "user_profile", (
        f"补算工具的首选字段是 {_PROFILE_KEYS[0]!r}，与 run_inference.py 写的 'user_profile' 不一致"
    )
