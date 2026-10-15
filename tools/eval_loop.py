#!/usr/bin/env python3
"""一键评测闭环：造验证输入 -> 官方推理脚本 -> 格式校验 -> 本地打分 -> 实验台账。

这是五条线共用的「裁判」：任何提示词 / RAG / 模型改动，都用同一条命令、同一份验证集打分，
结果追加到 runs/leaderboard.tsv，避免各人报不同的数、互相没法比。

用法
----
    # 1) 小样本冒烟（先跑这个，确认链路通）
    python tools/eval_loop.py --tag smoke --limit 20

    # 2) 全量（默认 transformers 后端，模型目录来自 inference_config.json）
    python tools/eval_loop.py --tag baseline

    # 3) 指向共享推理服务（vLLM / Ollama，AMD 机器或咪咕平台上的都行）
    python tools/eval_loop.py --tag r1 --backend openai_compatible \
        --base-url http://127.0.0.1:8000/v1 --api-model Qwen2.5-3B-Instruct

    # 4) 已经跑出 submission.jsonl，只重新打分
    python tools/eval_loop.py --tag r1 --skip-inference \
        --predictions runs/eval-r1/submission.jsonl

产物
----
    runs/eval-<tag>/val_input.jsonl        喂给官方脚本的输入（只含 history，不含答案）
    runs/eval-<tag>/submission.jsonl       官方脚本的输出
    runs/eval-<tag>/performance_report.json
    runs/eval-<tag>/score.json             完整评分报告
    runs/leaderboard.tsv                   一行一次实验

注意
----
* 验证集的 target **绝不**写进 val_input.jsonl（防泄漏，脚本会自己断言这件事）。
* --limit 小于全量时 BLEU/ROUGE 是语料级指标，样本少波动大，**不要**拿小样本的数跟全量比。
* 延迟只有在目标硬件（RTX 4090 或本地卡）上量才有意义；走网络的服务不具参考性。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
PARTICIPANT = REPO_ROOT / "GivenData" / "participant"
INFERENCE_SCRIPT = PARTICIPANT / "run_inference.py"
RUNS = REPO_ROOT / "runs"
DEFAULT_SOURCE = PROCESSED / "val_samples.jsonl"
DEFAULT_REFERENCES = PROCESSED / "val_references.jsonl"
LEADERBOARD = RUNS / "leaderboard.tsv"

INPUT_KEYS = ("id", "conversation_id", "target_user_turn_id", "history")
LEADERBOARD_COLUMNS = (
    "time",
    "tag",
    "backend",
    "model",
    "n_samples",
    "n_comparable",
    "emotion_acc",
    "profile_acc",
    "bleu4",
    "rouge_l",
    "memory_f1",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def build_val_input(source: Path, out_path: Path, limit: int | None, offset: int) -> int:
    """把带标签的验证样本削成推理输入（只留 id 与 history），并断言没夹带答案。"""
    if not source.exists():
        raise SystemExit(f"找不到 {source}，请先运行: python src/cli.py build")

    rows = []
    for index, sample in enumerate(read_jsonl(source)):
        if index < offset:
            continue
        if limit is not None and len(rows) >= limit:
            break
        payload = {key: sample[key] for key in INPUT_KEYS if key in sample}
        missing = [key for key in INPUT_KEYS if key not in payload]
        if missing:
            raise SystemExit(f"{source} 第 {index + 1} 条缺字段 {missing}")
        rows.append(payload)

    if not rows:
        raise SystemExit(f"{source} 在 offset={offset} 处没有取到样本")

    forbidden = [key for key in ("target", "response_text", "emotion_label") if any(key in row for row in rows)]
    if forbidden:
        raise SystemExit(f"输入里混进了答案字段 {forbidden}，这会造成泄漏，已中止")

    return write_jsonl(out_path, rows)


def run_step(cmd, *, env=None, title: str) -> int:
    print(f"\n$ {' '.join(str(part) for part in cmd)}")
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    print(f"[{title}] 退出码 {completed.returncode}")
    return completed.returncode


def score(predictions: Path, references: Path, score_path: Path, *, tokenizer: str, include_memory_refs: bool) -> dict:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "cli.py"),
        "eval",
        "--predictions",
        str(predictions),
        "--references",
        str(references),
        "--json-out",
        str(score_path),
    ]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    if include_memory_refs:
        cmd.append("--include-memory-refs")
    code = run_step(cmd, title="eval")
    if code != 0:
        raise SystemExit(f"打分失败（退出码 {code}）")

    payload = json.loads(score_path.read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def extract_row(metrics: dict) -> dict:
    quality = metrics.get("response_quality", {})
    profile = metrics.get("profile_dimension_scores", {})
    return {
        "n_comparable": metrics.get("n_comparable"),
        "emotion_acc": metrics.get("emotion_accuracy"),
        "profile_acc": metrics.get("profile_accuracy_mean_3dim"),
        "bleu4": quality.get("bleu4"),
        "rouge_l": quality.get("rouge_l_f"),
        "memory_f1": metrics.get("memory_refs_scores", {}).get("micro_f1"),
        "profile_dims": {
            name: score_.get("exact_match_acc") for name, score_ in profile.items()
        },
    }


def format_cell(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def append_leaderboard(row: dict) -> None:
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LEADERBOARD.exists()
    with LEADERBOARD.open("a", encoding="utf-8", newline="\n") as handle:
        if is_new:
            handle.write("\t".join(LEADERBOARD_COLUMNS) + "\n")
        handle.write("\t".join(format_cell(row.get(column)) for column in LEADERBOARD_COLUMNS) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一键评测闭环（验证集推理 + 格式校验 + 本地打分 + 台账）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tag", required=True, help="这次实验的名字，决定 runs/eval-<tag>/ 目录")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="带标签的验证样本（默认 data/processed/val_samples.jsonl）")
    parser.add_argument("--references", default=str(DEFAULT_REFERENCES), help="真值文件（默认 data/processed/val_references.jsonl）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（小样本冒烟用）")
    parser.add_argument("--offset", type=int, default=0, help="从第几条开始取")
    parser.add_argument("--backend", default="transformers", choices=("transformers", "openai_compatible"))
    parser.add_argument("--model-path", default=None, help="transformers 后端的模型目录（相对路径以 GivenData/participant 为基准）")
    parser.add_argument("--base-url", default=None, help="openai_compatible 后端的服务地址，例如 http://127.0.0.1:8000/v1")
    parser.add_argument("--api-model", default=None, help="openai_compatible 后端请求里带的模型名")
    parser.add_argument("--api-key", default=None, help="openai_compatible 后端的密钥（没有就留空）")
    parser.add_argument("--hardware-label", default=None, help="写进 performance_report.json 的硬件标签")
    parser.add_argument("--tokenizer", default="char", choices=("char", "jieba"), help="BLEU/ROUGE 分词口径（默认 char）")
    parser.add_argument("--include-memory-refs", action="store_true", help="把 memory_refs 并入画像总分（默认只作参考）")
    parser.add_argument("--skip-inference", action="store_true", help="不跑推理，直接给已有 submission 打分")
    parser.add_argument("--skip-validate", action="store_true", help="跳过提交格式校验")
    parser.add_argument("--predictions", default=None, help="配合 --skip-inference：已有的 submission.jsonl")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    result_dir = RUNS / f"eval-{args.tag}"
    result_dir.mkdir(parents=True, exist_ok=True)
    submission = result_dir / "submission.jsonl"
    score_path = result_dir / "score.json"

    if args.skip_inference:
        if not args.predictions:
            raise SystemExit("--skip-inference 需要同时给 --predictions")
        submission = Path(args.predictions).resolve()
        if not submission.exists():
            raise SystemExit(f"找不到 {submission}")
        print(f"[1/4] 跳过推理，直接用 {submission}")
        input_path = None
    else:
        input_path = result_dir / "val_input.jsonl"
        count = build_val_input(Path(args.source).resolve(), input_path, args.limit, args.offset)
        print(f"[1/4] 验证输入已生成: {input_path}  ({count} 条，只含 history)")

        env = os.environ.copy()
        env["EVALUATION_BACKEND"] = args.backend
        env["EVALUATION_HARDWARE_LABEL"] = args.hardware_label or f"eval-loop-{args.tag}"
        if args.base_url:
            env["EVALUATION_BASE_URL"] = args.base_url
        if args.api_model:
            env["EVALUATION_API_MODEL"] = args.api_model
        if args.api_key is not None:
            env["EVALUATION_API_KEY"] = args.api_key
        if args.model_path:
            env["EVALUATION_MODEL_PATH"] = args.model_path

        model_label = args.api_model or args.model_path or "(config 默认)"
        print(f"[2/4] 跑推理: backend={args.backend} model={model_label}")
        code = run_step(
            [sys.executable, str(INFERENCE_SCRIPT), str(input_path), str(result_dir)],
            env=env,
            title="inference",
        )
        if code != 0:
            raise SystemExit(
                f"推理失败（退出码 {code}）。常见原因：模型目录不存在、缺 torch/transformers、"
                "共享服务没起来、或 Python 版本没有对应轮子。"
            )
        if not submission.exists():
            raise SystemExit(f"推理跑完了但没有 {submission}")

    if args.skip_validate:
        print("[3/4] 跳过格式校验")
    else:
        cmd = [sys.executable, str(REPO_ROOT / "src" / "cli.py"), "validate", "--submission", str(submission)]
        if input_path is not None:
            cmd += ["--input", str(input_path)]
        run_step(cmd, title="validate")

    print("[4/4] 本地打分")
    metrics = score(
        submission,
        Path(args.references).resolve(),
        score_path,
        tokenizer=args.tokenizer,
        include_memory_refs=args.include_memory_refs,
    )
    row = extract_row(metrics)
    row.update(
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": args.tag,
            "backend": args.backend,
            "model": args.api_model or args.model_path or "config",
            "n_samples": None if args.skip_inference else count,
        }
    )
    append_leaderboard(row)

    print("\n== 本次结果 ==")
    print(f"  可比对样本        {row['n_comparable']}")
    print(f"  情绪 Acc          {format_cell(row['emotion_acc'])}")
    print(f"  画像 Acc(三维均值) {format_cell(row['profile_acc'])}")
    for name, value in (row.get("profile_dims") or {}).items():
        print(f"      {name:<20} {format_cell(value)}")
    print(f"  回复 BLEU-4       {format_cell(row['bleu4'])}   ROUGE-L {format_cell(row['rouge_l'])}")
    print(f"  [参考] memory_refs micro-F1 {format_cell(row['memory_f1'])}")
    print(f"\n台账 -> {LEADERBOARD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
