"""提交格式校验器：在把结果送进评测容器之前，先在本机把 schema 层面的错误全部挡掉。

为什么单独写一个模块
--------------------
README.md 第六节点名过："字段名若拼错（例如 ``emotion`` 写成 ``emotions``）将直接导致对应项
0 分"。这类错误不会让脚本崩溃，只会静默丢分，因此必须在提交前用机器逐一核验，而不是靠肉眼。

校验分两层
----------
1. **单条记录**：必需字段、字段类型、枚举取值、数组去重、正则格式。
2. **整文件**：ID 唯一性、与输入测试文件的 ID 集合是否一致（漏条 / 多条都是致命的）、
   空值率统计。

关于 ID 正则的一处已知冲突
--------------------------
官方 ``submission_schema.json`` 里 ``id`` 的正则是 ``^test_[0-9]{6}_t[0-9]+$``，
但官方推理样例的 ID 是 ``test1_000001_t5``，多了一个 ``1``，**不匹配该正则**。

本校验器因此采用"样例优先"策略：默认接受 ``^test[0-9]*_[0-9]{6}_t[0-9]+$``（即允许
``test1_``），并在报告中记录发现的真实前缀。若拿到官方评分脚本确认必须严格匹配
schema，用 ``--strict-schema-id`` 切成严格模式即可，不需要改代码。

另外，``train_/val_`` 前缀只在本地自测（拿自己的验证集划分跑闭环）时放行，
见 ``ID_PATTERN_LOCAL``。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

try:  # 允许 ``python src/validate_submission.py`` 直接运行
    from label_space import EMOTION_VALUES_WITH_EMPTY, PROFILE_DIMENSIONS, PROFILE_VALUES
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from label_space import EMOTION_VALUES_WITH_EMPTY, PROFILE_DIMENSIONS, PROFILE_VALUES

REQUIRED_FIELDS = ("id", "response_text", "emotion_label", "user_profile", "memory_refs")

ID_PATTERN_LENIENT = re.compile(r"^test[0-9]*_[0-9]{6}_t[0-9]+$")
ID_PATTERN_STRICT = re.compile(r"^test_[0-9]{6}_t[0-9]+$")
# 本地自测专用：我们自己的 train/val 划分也用同一套 ``<prefix>_<6位>_t<n>`` 编号，
# 拿它们跑闭环自测时不该被判成格式错——格式错只对"要交出去的文件"成立。
ID_PATTERN_LOCAL = re.compile(r"^(?:train|val)_[0-9]{6}_t[0-9]+$")
MEMORY_PATTERN = re.compile(r"^mem_[0-9]{6}$")


@dataclass
class ValidationIssue:
    line: int
    sample_id: str
    rule: str
    message: str

    def render(self) -> str:
        return f"  行 {self.line} [{self.rule}] {self.sample_id or '<无 id>'}: {self.message}"


@dataclass
class ValidationResult:
    total: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    id_prefixes: Counter = field(default_factory=Counter)
    empty_emotion: int = 0
    empty_response: int = 0
    empty_profile_all: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues

    def render(self, limit: int = 40) -> str:
        lines: list[str] = []
        lines.append(f"共校验 {self.total} 条记录")
        lines.append(f"  id 前缀分布: {dict(self.id_prefixes)}")
        lines.append(
            f"  空情绪 {self.empty_emotion} | 空回复 {self.empty_response} | "
            f"画像三维全空 {self.empty_profile_all}"
        )
        if self.ok:
            lines.append("结果: 通过（未发现 schema 层面的问题）")
            return "\n".join(lines)
        lines.append(f"结果: 发现 {len(self.issues)} 处问题")
        for issue in self.issues[:limit]:
            lines.append(issue.render())
        if len(self.issues) > limit:
            lines.append(f"  ... 另有 {len(self.issues) - limit} 处未显示")
        return "\n".join(lines)


def _iter_records(path: Path) -> Iterator[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                yield lineno, exc


def validate_submission(
    submission_path: str | Path,
    *,
    input_path: str | Path | None = None,
    strict_schema_id: bool = False,
) -> ValidationResult:
    """校验一个提交 JSONL 文件。"""
    result = ValidationResult()
    seen_ids: set[str] = set()
    id_pattern = ID_PATTERN_STRICT if strict_schema_id else ID_PATTERN_LENIENT

    for lineno, record in _iter_records(Path(submission_path)):
        if isinstance(record, json.JSONDecodeError):
            result.issues.append(
                ValidationIssue(lineno, "", "json", f"JSON 解析失败: {record}")
            )
            continue
        if not isinstance(record, dict):
            result.issues.append(
                ValidationIssue(lineno, "", "type", f"每行必须是 JSON object，当前是 {type(record).__name__}")
            )
            continue

        result.total += 1
        sample_id = record.get("id") if isinstance(record.get("id"), str) else ""
        if not sample_id:
            result.issues.append(ValidationIssue(lineno, "", "id.missing", "缺少字符串 id"))
        else:
            if not id_pattern.match(sample_id) and not (
                not strict_schema_id and ID_PATTERN_LOCAL.match(sample_id)
            ):
                result.issues.append(
                    ValidationIssue(
                        lineno,
                        sample_id,
                        "id.pattern",
                        f"id 不匹配 {'严格' if strict_schema_id else '宽松'}正则 {id_pattern.pattern}"
                        "（本地 train_/val_ 前缀仅在自测时放行）",
                    )
                )
            if sample_id in seen_ids:
                result.issues.append(
                    ValidationIssue(lineno, sample_id, "id.duplicate", "id 重复出现")
                )
            seen_ids.add(sample_id)
            result.id_prefixes[sample_id.split("_")[0]] += 1

        extra = set(record) - set(REQUIRED_FIELDS)
        if extra:
            result.issues.append(
                ValidationIssue(
                    lineno, sample_id, "field.extra", f"出现未定义字段 {sorted(extra)}（additionalProperties=false）"
                )
            )
        missing = [name for name in REQUIRED_FIELDS if name not in record]
        if missing:
            result.issues.append(
                ValidationIssue(lineno, sample_id, "field.missing", f"缺少必需字段 {missing}")
            )

        response = record.get("response_text")
        if not isinstance(response, str):
            result.issues.append(
                ValidationIssue(lineno, sample_id, "response_text.type", "response_text 必须是字符串")
            )
        elif not response.strip():
            result.empty_response += 1

        emotion = record.get("emotion_label")
        if not isinstance(emotion, str) or emotion not in EMOTION_VALUES_WITH_EMPTY:
            result.issues.append(
                ValidationIssue(
                    lineno, sample_id, "emotion_label.enum", f"emotion_label 非法: {emotion!r}"
                )
            )
        elif emotion == "":
            result.empty_emotion += 1

        profile = record.get("user_profile")
        if not isinstance(profile, dict):
            result.issues.append(
                ValidationIssue(lineno, sample_id, "user_profile.type", "user_profile 必须是 object")
            )
            continue
        profile_missing = [d for d in PROFILE_DIMENSIONS if d not in profile]
        profile_extra = sorted(set(profile) - set(PROFILE_DIMENSIONS))
        if profile_missing:
            result.issues.append(
                ValidationIssue(
                    lineno, sample_id, "user_profile.missing", f"缺少画像维度 {profile_missing}"
                )
            )
        if profile_extra:
            result.issues.append(
                ValidationIssue(
                    lineno, sample_id, "user_profile.extra", f"出现未定义画像维度 {profile_extra}"
                )
            )
        all_empty = True
        for dimension in PROFILE_DIMENSIONS:
            values = profile.get(dimension)
            if values is None:
                continue
            if not isinstance(values, list):
                result.issues.append(
                    ValidationIssue(
                        lineno, sample_id, f"user_profile.{dimension}.type", "必须是数组"
                    )
                )
                all_empty = False
                continue
            if values:
                all_empty = False
            if len(values) != len(set(map(str, values))):
                result.issues.append(
                    ValidationIssue(
                        lineno, sample_id, f"user_profile.{dimension}.unique", "数组元素重复"
                    )
                )
            illegal = [v for v in values if v not in PROFILE_VALUES.get(dimension, ())]
            if illegal:
                result.issues.append(
                    ValidationIssue(
                        lineno,
                        sample_id,
                        f"user_profile.{dimension}.enum",
                        f"非法取值 {illegal[:5]}",
                    )
                )
        if all_empty:
            result.empty_profile_all += 1

        memories = record.get("memory_refs")
        if not isinstance(memories, list):
            result.issues.append(
                ValidationIssue(lineno, sample_id, "memory_refs.type", "memory_refs 必须是数组")
            )
        else:
            if len(memories) != len(set(map(str, memories))):
                result.issues.append(
                    ValidationIssue(lineno, sample_id, "memory_refs.unique", "数组元素重复")
                )
            bad = [m for m in memories if not isinstance(m, str) or not MEMORY_PATTERN.match(m)]
            if bad:
                result.issues.append(
                    ValidationIssue(
                        lineno, sample_id, "memory_refs.pattern", f"非法引用 {bad[:5]}"
                    )
                )

    # 与输入测试文件的 ID 集合比对
    if input_path is not None:
        expected: list[str] = []
        with Path(input_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                expected.append(json.loads(line).get("id", ""))
        expected_set = set(expected)
        missing_ids = sorted(expected_set - seen_ids)
        extra_ids = sorted(seen_ids - expected_set)
        if missing_ids:
            result.issues.append(
                ValidationIssue(
                    0,
                    "",
                    "coverage.missing",
                    f"输入中有 {len(missing_ids)} 条没有对应预测（示例 {missing_ids[:5]}）",
                )
            )
        if extra_ids:
            result.issues.append(
                ValidationIssue(
                    0,
                    "",
                    "coverage.extra",
                    f"提交中出现 {len(extra_ids)} 条输入里没有的 id（示例 {extra_ids[:5]}）",
                )
            )
        if len(expected) != result.total:
            result.issues.append(
                ValidationIssue(
                    0,
                    "",
                    "coverage.count",
                    f"输入 {len(expected)} 条 vs 提交 {result.total} 条，数量不一致",
                )
            )
    return result


def validate_performance_report(path: str | Path) -> list[str]:
    """轻量检查官方 ``performance_report.json`` 是否含计时所需的字段。"""
    problems: list[str] = []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    text = json.dumps(payload, ensure_ascii=False).lower()
    for key in ("latency", "mean", "95", "count"):
        if key not in text:
            problems.append(f"performance_report.json 里没找到与 {key!r} 相关的字段，请人工确认计时口径")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验提交 JSONL 是否符合官方 schema")
    parser.add_argument("submission", help="待校验的 submission.jsonl")
    parser.add_argument("--input", default=None, help="对应的输入测试 jsonl，用于核对 ID 覆盖")
    parser.add_argument(
        "--strict-schema-id",
        action="store_true",
        help="按官方 schema 的 ^test_[0-9]{6}_t[0-9]+$ 严格校验 id（默认宽松，接受 test1_ 前缀）",
    )
    parser.add_argument("--json-out", default=None, help="把问题清单另存为 JSON")
    args = parser.parse_args(argv)

    result = validate_submission(
        args.submission, input_path=args.input, strict_schema_id=args.strict_schema_id
    )
    print(result.render())
    if args.json_out:
        payload = {
            "total": result.total,
            "ok": result.ok,
            "id_prefixes": dict(result.id_prefixes),
            "empty_emotion": result.empty_emotion,
            "empty_response": result.empty_response,
            "empty_profile_all": result.empty_profile_all,
            "issues": [issue.__dict__ for issue in result.issues],
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"问题清单已写入 {args.json_out}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
