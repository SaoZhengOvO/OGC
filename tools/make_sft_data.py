#!/usr/bin/env python3
"""把已处理好的样本转成 SFT 训练数据（chat 格式）。

**训练与推理必须用同一份提示词**，否则微调学到的是另一个任务。所以本模块
不自己抄一份提示词，而是直接加载 `deploy/participant/run_inference.py`，
复用它的 `SYSTEM_PROMPT`、`EMOTIONS`、`PROFILE_LABELS` 和
`conversation_text()`——那份脚本的模块级 import 全是标准库，加载很便宜。

切分按 `conversation_id` 而不是按条：同一段对话的多轮共享同一个用户，
按条切会让验证集里出现训练时见过的用户，分数虚高。

用法：
    python tools/make_sft_data.py
    python tools/make_sft_data.py --dev-per-mille 0     # 全部当训练数据
    python tools/make_sft_data.py --source data/processed/val_samples.jsonl --train-out runs/sft/val_as_train.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
INFERENCE_SCRIPT = REPO_ROOT / "deploy" / "participant" / "run_inference.py"

PROFILE_DIMENSIONS = ("personality_traits", "interests", "style")

DEFAULT_SOURCE = PROCESSED / "train_samples.jsonl"
DEFAULT_TRAIN_OUT = PROCESSED / "sft_train.jsonl"
DEFAULT_DEV_OUT = PROCESSED / "sft_dev.jsonl"


def load_inference_module() -> ModuleType:
    """加载部署脚本本体，拿到官方提示词与枚举（唯一真源）。"""
    if not INFERENCE_SCRIPT.is_file():
        raise SystemExit(f"找不到推理脚本 {INFERENCE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("_deployed_run_inference", INFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载 {INFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_system_prompt(module: ModuleType) -> str:
    """把官方模板里的枚举占位符填成推理时真正会发给模型的那一段。"""
    return module.SYSTEM_PROMPT.format(
        emotions=", ".join(module.EMOTIONS),
        personality=", ".join(sorted(module.PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(module.PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(module.PROFILE_LABELS["style"])),
    )


def clean_labels(module: ModuleType, dimension: str, values: Any) -> tuple[list[str], int]:
    """按枚举过滤某一维画像标签，去重保序；返回 (干净标签, 被丢掉的个数)。"""
    allowed = module.PROFILE_LABELS[dimension]
    if not isinstance(values, list):
        return [], 0
    kept: list[str] = []
    dropped = 0
    for value in values:
        if not isinstance(value, str) or value not in allowed:
            dropped += 1
            continue
        if value not in kept:
            kept.append(value)
    return kept, dropped


def build_target_json(module: ModuleType, target: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """把金标拼成模型该吐出的那一个 JSON object（单行、无空格）。"""
    stats = {"dropped_profile_labels": 0, "bad_emotion": 0}
    profile = target.get("user_profile")
    profile = profile if isinstance(profile, dict) else {}
    clean_profile: dict[str, list[str]] = {}
    for dimension in PROFILE_DIMENSIONS:
        kept, dropped = clean_labels(module, dimension, profile.get(dimension))
        clean_profile[dimension] = kept
        stats["dropped_profile_labels"] += dropped

    emotion = target.get("emotion_label")
    if emotion not in module.EMOTIONS:
        emotion = ""
        stats["bad_emotion"] = 1

    payload = {
        "response_text": str(target.get("response_text") or "").strip(),
        "emotion_label": emotion,
        "user_profile": clean_profile,
        # 官方提示词要求固定输出 []：测试集没有公开 memory bank。
        "memory_refs": [],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), stats


def to_record(module: ModuleType, sample: dict[str, Any], system_prompt: str) -> dict[str, Any] | None:
    """单条样本转成 chat 记录；不合格返回 None。"""
    target = sample.get("target")
    history = sample.get("history")
    if not isinstance(target, dict) or not isinstance(history, list) or not history:
        return None
    try:
        user_text = module.conversation_text(sample)
    except (KeyError, ValueError, TypeError):
        return None
    answer, stats = build_target_json(module, target)
    if not json.loads(answer)["response_text"]:
        return None
    return {
        "id": sample["id"],
        "conversation_id": sample.get("conversation_id", ""),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ],
        "stats": stats,
    }


def is_dev_conversation(conversation_id: str, dev_per_mille: int) -> bool:
    """按 conversation_id 稳定切分：同一段对话永远落在同一侧。"""
    if dev_per_mille <= 0:
        return False
    digest = hashlib.sha1(conversation_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 1000 < dev_per_mille


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """逐行读 JSONL，跳过空行。"""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """写 JSONL；记录里的 `stats` 只在内存里用于统计，不落盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            row = {key: value for key, value in row.items() if key != "stats"}
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="把样本转成 SFT 训练数据")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="输入 JSONL（默认训练集）")
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--dev-out", type=Path, default=DEFAULT_DEV_OUT)
    parser.add_argument(
        "--dev-per-mille",
        type=int,
        default=20,
        help="千分之几的对话划给验证集（默认 20，即 2%%）；0 表示全部当训练数据",
    )
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试用，0 表示全部）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    args = build_parser().parse_args(argv)
    module = load_inference_module()
    system_prompt = build_system_prompt(module)

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    totals = {"read": 0, "skipped": 0, "dropped_profile_labels": 0, "bad_emotion": 0}
    dev_conversations: set[str] = set()
    train_conversations: set[str] = set()

    for sample in read_jsonl(args.source):
        if args.limit and totals["read"] >= args.limit:
            break
        totals["read"] += 1
        record = to_record(module, sample, system_prompt)
        if record is None:
            totals["skipped"] += 1
            continue
        totals["dropped_profile_labels"] += record["stats"]["dropped_profile_labels"]
        totals["bad_emotion"] += record["stats"]["bad_emotion"]
        conversation = record["conversation_id"]
        if is_dev_conversation(conversation, args.dev_per_mille):
            dev_rows.append(record)
            dev_conversations.add(conversation)
        else:
            train_rows.append(record)
            train_conversations.add(conversation)

    if not train_rows:
        raise SystemExit(f"{args.source} 没有产出任何训练记录，先确认数据已展开")

    written_train = write_jsonl(args.train_out, train_rows)
    written_dev = write_jsonl(args.dev_out, dev_rows) if dev_rows else 0

    print(f"读入      {totals['read']} 条")
    print(f"跳过      {totals['skipped']} 条（缺 target / history 为空 / 回复为空）")
    print(f"丢标签    {totals['dropped_profile_labels']} 个（不在枚举里或重复）")
    print(f"坏情绪    {totals['bad_emotion']} 条（金标情绪不在 16 类里，已写成空串）")
    print(f"训练集    {written_train} 条 / {len(train_conversations)} 段对话 -> {args.train_out}")
    print(f"验证集    {written_dev} 条 / {len(dev_conversations)} 段对话 -> {args.dev_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
