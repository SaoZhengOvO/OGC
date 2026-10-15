"""官方标签空间（唯一真相来源）。

所有标签枚举都从 ``GivenData`` 里的官方 schema 抄录而来，硬编码在此处而不是运行时读
schema 文件，原因有两个：

1. 评测容器里 ``GivenData`` 不一定存在于同一路径，硬编码可以避免运行时依赖；
2. README.md 第六节明确警告过"字段名拼错直接 0 分"，把枚举集中到一处可以防止
   训练侧、推理侧、评分侧各自维护一份而逐渐漂移。

改动本文件前请先对照：

- ``GivenData/训练-验证-数据集/train/train_public.schema.json``
- ``GivenData/participant/templates/submission_schema.json``

两类标签的取值范围：验证/训练数据里 emotion 是 16 类闭集；提交 schema 额外允许空字符串
``""``，表示该项预测失败（官方评分脚本会把空值按"未命中"处理）。
"""

from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# 情绪标签（16 类闭集）
# --------------------------------------------------------------------------- #
EMOTIONS: tuple[str, ...] = (
    "joy",
    "gratitude",
    "relaxed",
    "care",
    "pride",
    "neutral",
    "surprise",
    "mixed",
    "sadness",
    "loneliness",
    "anxiety",
    "anger",
    "fear",
    "disgust",
    "shame",
    "helplessness",
)

# 提交 schema 允许空串，表示该样本情绪字段解析失败。
EMOTION_VALUES_WITH_EMPTY: tuple[str, ...] = ("",) + EMOTIONS

# --------------------------------------------------------------------------- #
# 用户画像三个子维度
# --------------------------------------------------------------------------- #
PROFILE_DIMENSIONS: tuple[str, ...] = ("personality_traits", "interests", "style")

PERSONALITY_TRAITS: tuple[str, ...] = (
    "extroverted",
    "introverted",
    "open",
    "conservative",
    "high_conscientiousness",
    "casual",
    "agreeable",
    "assertive",
    "emotionally_stable",
    "sensitive",
)

INTERESTS: tuple[str, ...] = (
    "study_exam",
    "programming_technology",
    "reading_writing",
    "film_animation",
    "music",
    "games",
    "sports_fitness",
    "travel_outdoor",
    "pets",
    "social",
    "career_development",
    "art_design",
)

STYLE: tuple[str, ...] = (
    "brief",
    "detailed",
    "colloquial",
    "formal",
    "direct",
    "indirect",
    "humorous",
    "rational",
    "high_emotional_expression",
    "low_emotional_expression",
    "emoji_user",
)

PROFILE_VALUES: dict[str, tuple[str, ...]] = {
    "personality_traits": PERSONALITY_TRAITS,
    "interests": INTERESTS,
    "style": STYLE,
}

# --------------------------------------------------------------------------- #
# 标签规范化
# --------------------------------------------------------------------------- #


def _build_alias_map(values: tuple[str, ...]) -> dict[str, str]:
    """为每个合法标签建立别名表：大小写、连字符、空格都不该导致 0 分。"""
    alias: dict[str, str] = {}
    for value in values:
        alias[value] = value
        alias[value.lower()] = value
        alias[value.replace("_", "-")] = value
        alias[value.replace("_", " ")] = value
        alias[value.replace("_", "")] = value
    return alias


_EMOTION_ALIAS = _build_alias_map(EMOTIONS)
_PROFILE_ALIAS = {dim: _build_alias_map(vals) for dim, vals in PROFILE_VALUES.items()}


def normalize_emotion(raw: object) -> str:
    """把模型输出归一到官方情绪标签；无法归一返回 ``""``（计为预测失败）。"""
    if not isinstance(raw, str):
        return ""
    key = raw.strip().lower()
    if not key:
        return ""
    return _EMOTION_ALIAS.get(key, _EMOTION_ALIAS.get(key.replace("-", "_"), ""))


def normalize_profile_dimension(dimension: str, raw: object) -> list[str]:
    """把某一画像维度的模型输出归一为去重、有序的合法标签列表。

    非列表输入、非法标签一律丢弃；输出顺序按官方枚举顺序，保证同一集合的序列化结果
    稳定（便于 diff 与逐样本核对）。
    """
    if dimension not in PROFILE_VALUES:
        raise KeyError(f"未知画像维度: {dimension}")
    if isinstance(raw, str):
        candidates: list[object] = [raw]
    elif isinstance(raw, (list, tuple, set)):
        candidates = list(raw)
    else:
        return []

    canonical_order = PROFILE_VALUES[dimension]
    alias = _PROFILE_ALIAS[dimension]
    hits: set[str] = set()
    for item in candidates:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if not key:
            continue
        hit = alias.get(key) or alias.get(key.replace("-", "_"))
        if hit:
            hits.add(hit)
    return [label for label in canonical_order if label in hits]


def normalize_profile(raw: object) -> dict[str, list[str]]:
    """归一整个 ``user_profile`` 对象；缺失维度补空列表。"""
    source = raw if isinstance(raw, dict) else {}
    return {dim: normalize_profile_dimension(dim, source.get(dim)) for dim in PROFILE_DIMENSIONS}


def export_label_space(path: str | Path) -> Path:
    """把标签空间落盘成 ``configs/label_space.json``。

    训练侧（构造 Prompt 里的候选枚举）和推理侧（受限解码、后处理映射）都应该读这个
    文件，而不是各自硬编码，避免多次复制后漂移。
    """
    payload = {
        "_comment": "由 src/label_space.py 导出，请勿手工编辑；改动请改 src/label_space.py。",
        "emotions": list(EMOTIONS),
        "profile_dimensions": list(PROFILE_DIMENSIONS),
        "profile_values": {dim: list(vals) for dim, vals in PROFILE_VALUES.items()},
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "configs/label_space.json"
    written = export_label_space(target)
    print(f"已导出标签空间 -> {written}")
    print(f"  emotion: {len(EMOTIONS)} 类")
    for dim in PROFILE_DIMENSIONS:
        print(f"  {dim}: {len(PROFILE_VALUES[dim])} 类")
