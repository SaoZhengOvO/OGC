#!/usr/bin/env python3
"""一键评测闭环：造验证输入 -> 官方推理脚本 -> 格式校验 -> 本地打分 -> 实验台账。

这是全队共用的「裁判」：任何提示词 / RAG / 模型改动，都用同一条命令、同一份验证集打分，
结果追加到 runs/leaderboard.tsv，避免各人报不同的数、互相没法比。

跑的到底是哪份脚本（重要）
--------------------------
主办方唯一会执行的是 `bash /root/participant/start.sh TEST_FILE RESULT_DIR`，见
`GivenData/participant/README.md`「唯一运行命令」一节 —— 也就是**我们容器里的那份**，
不是 `GivenData/` 里的官方原件。所以默认跑的是 `deploy/participant/run_inference.py`
（带接线层，会加载队友的 `prompts/` 与 `rag/`）；`GivenData/participant/` 那份**没有接线层**，
拿它当默认会让你以为提示词没效果。

    --inference-script wired        # 默认：我们要提交的那棵树
    --inference-script official     # 官方裸基线，做对照用
    --inference-script <路径>       # 其它，相对路径以仓库根为基准

用法
----
    # 1) 小样本冒烟（先跑这个，确认链路通）
    python tools/eval_loop.py --tag smoke --limit 20

    # 2) 全量（默认 transformers 后端）
    python tools/eval_loop.py --tag baseline

    # 3) 官方裸基线对照跑
    python tools/eval_loop.py --tag official-baseline --inference-script official

    # 4) 指向共享推理服务（vLLM / Ollama，AMD 机器或咪咕平台上的都行）
    python tools/eval_loop.py --tag r1 --backend openai_compatible \
        --base-url http://127.0.0.1:8000/v1 --api-model Qwen2.5-3B-Instruct

    # 5) 已经跑出 submission.jsonl，只重新打分
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
RUNS = REPO_ROOT / "runs"

# 两棵参赛树。主办方跑的是**我们容器里的**那份（见 GivenData/participant/README.md
# 「唯一运行命令」：`bash /root/participant/start.sh TEST_FILE RESULT_DIR`，
# 「确保脚本位置无更改」），所以默认用 deploy 那棵；GivenData 那棵是**没有接线层**
# 的官方原件，只在做裸基线对照时用。
OFFICIAL_PARTICIPANT = REPO_ROOT / "GivenData" / "participant"
WIRED_PARTICIPANT = REPO_ROOT / "deploy" / "participant"
PARTICIPANT = WIRED_PARTICIPANT  # 旧引用兼容；别再拿它去找模型
SCRIPT_PRESETS = {
    "wired": WIRED_PARTICIPANT / "run_inference.py",
    "official": OFFICIAL_PARTICIPANT / "run_inference.py",
}
INFERENCE_SCRIPT = SCRIPT_PRESETS["wired"]
# deploy 树里没有 model/（5.9 GB 不复制第二份），两份 inference_config.json 却都写
# "./model" 且以**工具目录**为基准 —— 换了脚本不同时指定模型，模型就会找不到。
DEFAULT_MODEL_PATH = OFFICIAL_PARTICIPANT / "model"
# 镜像里 prompts/ 是 deploy/build.sh 从仓库根拷进 /root/participant/prompts 的，
# 本地没有这一步；不显式指回来就会**静默**退化成官方提示词，量到的数不对应提交的那棵树。
DEFAULT_PROMPT_DIR = REPO_ROOT / "prompts"

DEFAULT_SOURCE = PROCESSED / "val_samples.jsonl"
DEFAULT_REFERENCES = PROCESSED / "val_references.jsonl"
LEADERBOARD = RUNS / "leaderboard.tsv"

INPUT_KEYS = ("id", "conversation_id", "target_user_turn_id", "history")
LEADERBOARD_COLUMNS = (
    "time",
    "tag",
    "backend",
    "model",
    # 这两列是「同一条命令量到的到底是什么」的证据：没有它们，一行数字分不出
    # 是官方裸基线还是带队友提示词/RAG 的那棵树。
    "script",
    "prompts",
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


def resolve_repo_path(value: str | Path) -> Path:
    """绝对路径原样，相对路径以**仓库根**为基准（不是当前目录）。"""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def resolve_inference_script(value: str) -> Path:
    """把 `wired` / `official` 两个简写或一个路径解析成推理脚本。"""
    key = str(value).strip()
    if key.lower() in SCRIPT_PRESETS:
        return SCRIPT_PRESETS[key.lower()]
    return resolve_repo_path(key)


def script_kind(script: Path) -> str:
    """这份推理脚本会不会加载队友的提示词与 RAG：`wired` / `official` / `missing`。

    先认两棵已知的树；认不出来才看内容里有没有接线层的 import，这样有人把 deploy
    那份拷到别处也还能认对。
    """
    resolved = script.resolve()
    for name, preset in SCRIPT_PRESETS.items():
        if resolved == preset.resolve():
            return name
    try:
        text = script.read_text(encoding="utf-8")
    except OSError:
        return "missing"
    return "wired" if "import wiring" in text else "official"


def probe_wiring(prompt_dir: Path, participant_dir: Path, *, timeout: int = 60) -> dict:
    """问接线层自己「这一趟走了哪条路」，而不是我们在外面猜。

    必须在**子进程**里 import wiring，两个原因：`PROMPT_DIR` 是模块级常量，import
    那一刻就从环境变量求值，同一进程里再 import 不会重算；而且父进程（尤其 pytest）
    的 sys.path 通常已经含仓库根，`rag.*` 会假性 import 成功，报出根本走不通的路。
    """
    wiring_py = participant_dir / "wiring.py"
    if not wiring_py.is_file():
        return {"available": False, "reason": f"{wiring_py} 不存在"}

    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(participant_dir)!r}); "
        "import wiring; "
        "print(json.dumps(wiring.describe(), ensure_ascii=False))"
    )
    env = os.environ.copy()
    env["PARTICIPANT_PROMPT_DIR"] = str(prompt_dir)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": f"探测接线层失败：{error}"}

    if completed.returncode != 0:
        stderr_lines = [line for line in (completed.stderr or "").splitlines() if line.strip()]
        return {
            "available": False,
            "reason": stderr_lines[-1] if stderr_lines else "接线层 import 失败",
        }

    stdout_lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    if not stdout_lines:
        return {"available": False, "reason": "接线层没有输出"}
    try:
        info = json.loads(stdout_lines[-1])
    except ValueError:
        return {"available": False, "reason": "接线层输出了非 JSON"}
    if not isinstance(info, dict):
        return {"available": False, "reason": "接线层输出的不是对象"}
    info["available"] = True
    return info


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


def write_reference_subset(references: Path, scope_file: Path, out_path: Path) -> int:
    """把真值裁剪成和输入完全相同的 id 集合，返回保留条数。

    只在 ``--limit`` / ``--offset`` 跑部分样本时用。`src/evaluate.py` 的分母刻意取
    **完整验证集**（少交很多条不该反而分高），所以 20 条冒烟样本除以 2354 之后
    分数永远接近 0，看不出任何改进。部分样本跑分时把真值一起收窄，数才有意义。
    """
    wanted = {row.get("id") for row in read_jsonl(scope_file)}
    wanted.discard(None)
    if not wanted:
        raise SystemExit(f"{scope_file} 里没读到可用的 id")

    kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with references.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("id") in wanted:
                dst.write(stripped + "\n")
                kept += 1

    if not kept:
        raise SystemExit(f"{references} 里没有一条能对上 {scope_file} 的 id，确认两边是不是同一份数据")
    return kept


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


def migrate_leaderboard_header(path: Path, columns: tuple[str, ...]) -> bool:
    """台账列变过就按**列名**重排，老行缺的列补 n/a。返回是否改动了文件。

    不这么做的话，老表头 11 格、新行 13 格，整个台账会错位 —— 而它正是「五个人用
    同一把尺子」的证据，错位等于没有证据。按列名而不是按位置对齐，是为了让重排
    幂等：再跑一次不会二次搬运。
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    if not lines:
        return False
    old_header = lines[0].split("\t")
    if tuple(old_header) == tuple(columns):
        return False

    rebuilt = ["\t".join(columns)]
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        record = {
            name: cells[index] for index, name in enumerate(old_header) if index < len(cells)
        }
        rebuilt.append("\t".join(record.get(name, "n/a") for name in columns))
    path.write_text("\n".join(rebuilt) + "\n", encoding="utf-8", newline="\n")
    return True


def append_leaderboard(row: dict) -> None:
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    if LEADERBOARD.exists():
        migrate_leaderboard_header(LEADERBOARD, LEADERBOARD_COLUMNS)
    else:
        LEADERBOARD.write_text(
            "\t".join(LEADERBOARD_COLUMNS) + "\n", encoding="utf-8", newline="\n"
        )
    with LEADERBOARD.open("a", encoding="utf-8", newline="\n") as handle:
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
    parser.add_argument(
        "--inference-script",
        default="wired",
        help="跑哪份推理脚本：wired（默认，deploy/participant/，带接线层）"
        "| official（GivenData 官方原件，裸基线对照）| 其它路径（相对路径以仓库根为基准）",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"transformers 后端的模型目录（相对路径以仓库根为基准；默认 {DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--prompt-dir",
        default=str(DEFAULT_PROMPT_DIR),
        help="队友提示词目录，作为 PARTICIPANT_PROMPT_DIR 传给推理脚本（默认仓库根 prompts/）",
    )
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

    script = resolve_inference_script(args.inference_script)
    if not script.is_file():
        raise SystemExit(
            f"找不到推理脚本 {script}；--inference-script 可以给 wired / official / 一个路径"
        )
    participant_dir = script.parent
    prompt_dir = resolve_repo_path(args.prompt_dir).resolve()
    kind = script_kind(script)

    result_dir = RUNS / f"eval-{args.tag}"
    result_dir.mkdir(parents=True, exist_ok=True)
    submission = result_dir / "submission.jsonl"
    score_path = result_dir / "score.json"
    prompts_cell = "(skip)"

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

        model_path = resolve_repo_path(args.model_path) if args.model_path else None
        if model_path is None and args.backend == "transformers" and DEFAULT_MODEL_PATH.is_dir():
            model_path = DEFAULT_MODEL_PATH
        if model_path is not None:
            env["EVALUATION_MODEL_PATH"] = str(model_path)
        # 恒指回仓库根 prompts/：镜像里这一步由 deploy/build.sh 完成，本地不做就会
        # 悄悄退化成官方提示词，量出来的数不对应我们要提交的那棵树。
        env["PARTICIPANT_PROMPT_DIR"] = str(prompt_dir)

        wiring_info: dict = {}
        print(f"[2/4] 跑推理: backend={args.backend}")
        print(f"      脚本       {script}  [{kind}]")
        print(f"      模型       {model_path or '(由 inference_config.json 决定)'}")
        print(f"      提示词目录 {prompt_dir}")
        if kind == "wired":
            wiring_info = probe_wiring(prompt_dir, participant_dir)
            if wiring_info.get("available"):
                prompts_cell = ",".join(wiring_info.get("prompt_blocks_present") or []) or "(none)"
                print(
                    "      接线层     提示词={} rag.retrieve={} rag.fusion={}".format(
                        prompts_cell,
                        wiring_info.get("rag_retrieve"),
                        wiring_info.get("rag_fusion"),
                    )
                )
                if prompts_cell == "(none)":
                    print("      [注意] 一个提示词块都没加载 —— 这趟等于官方提示词基线。")
            else:
                prompts_cell = "?"
                print(f"      [警告] 接线层探测失败：{wiring_info.get('reason')}")
        else:
            prompts_cell = "(官方内置)"
            print("      接线层     无（官方原件；队友的提示词与 RAG 不会生效）")

        code = run_step(
            [sys.executable, str(script), str(input_path), str(result_dir)],
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
    references_path = Path(args.references).resolve()
    if args.limit is not None or args.offset:
        # 部分样本跑分：真值跟着收窄，否则分母永远是完整的 2354 条。
        scope_file = input_path if input_path is not None else submission
        subset_path = result_dir / "val_references_subset.jsonl"
        kept = write_reference_subset(references_path, scope_file, subset_path)
        print(f"      部分样本跑分，真值同步收窄到 {kept} 条；全量跑才是官方口径。")
        references_path = subset_path

    metrics = score(
        submission,
        references_path,
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
            "script": "(skip)" if args.skip_inference else kind,
            "prompts": prompts_cell,
            "n_samples": None if args.skip_inference else count,
        }
    )
    append_leaderboard(row)

    print("\n== 本次结果 ==")
    print(f"  推理脚本          {row['script']}  ({script})")
    print(f"  加载的提示词      {row['prompts']}")
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
