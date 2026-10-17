"""统一命令行入口：把数据转换、离线评分、提交校验串成一条可复现的命令链。

用法速览（在仓库根目录执行）::

    # 1) 把官方对话级数据展开成逐预测点样本，并输出数据质量报告
    python src/cli.py build

    # 2) 拿任一预测文件在验证集上打分
    python src/cli.py eval --predictions runs/baseline/submission.jsonl

    # 3) 提交前校验格式（对着输入测试文件核对 ID 覆盖）
    python src/cli.py validate --submission runs/final/submission.jsonl \\
        --input GivenData/test_inference_data.jsonl

所有子命令都不联网、不申请额外依赖，产物全部落在仓库内的相对路径下，便于留档与复现。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:  # 允许 python src/cli.py 直接运行
    sys.path.insert(0, str(SRC))


def display_path(path: str | Path) -> str:
    """打印用的路径。

    自定义 ``--out-dir`` / ``--report-dir`` 可能是相对路径、也可能在仓库外，
    直接 ``relative_to(REPO_ROOT)`` 会抛 ValueError，所以这里退回绝对路径。
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)

from data_processor import (  # noqa: E402
    convert_conversations,
    filter_samples,
    load_inference_samples,
    read_jsonl,
    write_json,
    write_jsonl,
)
from evaluate import evaluate_predictions, load_predictions, load_references  # noqa: E402
from label_space import export_label_space  # noqa: E402
from validate_submission import validate_submission  # noqa: E402

GIVEN = REPO_ROOT / "GivenData"
TRAIN_RAW = GIVEN / "训练-验证-数据集" / "train" / "train_public.jsonl"
VAL_RAW = GIVEN / "训练-验证-数据集" / "val" / "val_public.jsonl"
TEST_RAW = GIVEN / "test_inference_data.jsonl"

PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "data" / "reports"


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def cmd_build(args: argparse.Namespace) -> int:
    """展开训练/验证集，落盘样本、真值对照表、标签空间与数据质量报告。"""
    out_dir = Path(args.out_dir) if args.out_dir else PROCESSED
    report_dir = Path(args.report_dir) if args.report_dir else REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    label_path = export_label_space(REPO_ROOT / "configs" / "label_space.json")
    print(f"[1/4] 标签空间 -> {display_path(label_path)}")

    summary: dict[str, object] = {}
    produced: dict[str, list[dict]] = {}

    for name, raw_path in (("train", TRAIN_RAW), ("val", VAL_RAW)):
        if not raw_path.exists():
            print(f"  跳过 {name}: 找不到 {raw_path}")
            continue
        conversations = read_jsonl(raw_path)
        samples, quality = convert_conversations(
            conversations,
            drop_turn_mismatch=not args.keep_turn_mismatch,
            drop_true_leaks=not args.keep_true_leaks,
            leak_check=not args.no_leak_check,
        )
        produced[name] = samples
        sample_path = out_dir / f"{name}_samples.jsonl"
        write_jsonl(sample_path, samples)
        refs_path = out_dir / f"{name}_references.jsonl"
        references = [
            {"id": s["id"], "target": s["target"], "conversation_id": s["conversation_id"]}
            for s in samples
        ]
        write_jsonl(refs_path, references)
        quality_payload = quality.summary()
        write_json(report_dir / f"{name}_quality.json", quality_payload)
        summary[name] = {
            "raw_conversations": len(conversations),
            "samples": len(samples),
            "quality_report": display_path(report_dir / f"{name}_quality.json"),
        }
        print(
            f"[2/4] {name}: {len(conversations)} 段对话 -> {len(samples)} 个样本 "
            f"(丢弃错位 {quality.skipped_turn_mismatch}, "
            f"正常重复回应 {quality.repeat_response_samples}, "
            f"疑似真泄漏 {quality.leaked_samples})"
        )

    # 训练集的两个对照子集：画像非空 / 画像全空
    if "train" in produced:
        with_profile = filter_samples(produced["train"], with_profile_only=True)
        without_profile = filter_samples(produced["train"], without_profile_only=True)
        write_jsonl(out_dir / "train_with_profile.jsonl", with_profile)
        write_jsonl(out_dir / "train_without_profile.jsonl", without_profile)
        summary["train_subsets"] = {
            "with_profile": len(with_profile),
            "without_profile": len(without_profile),
        }
        print(
            f"[3/4] 训练集画像子集: 非空 {len(with_profile)} | 全空 {len(without_profile)}"
        )

    # 官方推理样例：确认我们的读取路径与官方字段一致
    if TEST_RAW.exists():
        test_samples = load_inference_samples(TEST_RAW)
        summary["test_inference_data"] = {"samples": len(test_samples)}
        sample_dump = out_dir / "test_inference_samples.jsonl"
        write_jsonl(sample_dump, test_samples)
        print(f"[4/4] 官方推理样例 {len(test_samples)} 条 -> {display_path(sample_dump)}")

    write_json(report_dir / "dataset_summary.json", summary)
    print(f"\n数据质量报告: {display_path(report_dir / 'dataset_summary.json')}")
    if "train" in produced:
        _print_quality_highlights(produced.get("train", []))
    return 0


def _print_quality_highlights(samples: list[dict]) -> None:
    """把报告里最该被五个人看到的几件事直接打到终端。"""
    if not samples:
        return
    total = len(samples)
    empty = sum(
        1
        for s in samples
        if not any(
            s["target"]["user_profile"].get(dim)
            for dim in ("personality_traits", "interests", "style")
        )
    )
    lengths = sorted(len(s["target"]["response_text"]) for s in samples)
    print("\n数据要点（训练集）")
    print(f"  画像三维全空: {empty}/{total} = {100 * empty / total:.1f}%  <- 决定了画像训练策略")
    print(
        f"  回复长度: 中位 {lengths[len(lengths) // 2]} 字, "
        f"P90 {lengths[int(len(lengths) * 0.9)]} 字, 最大 {lengths[-1]} 字"
    )


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def cmd_eval(args: argparse.Namespace) -> int:
    reference_path = Path(args.references) if args.references else PROCESSED / "val_references.jsonl"
    if not reference_path.exists():
        print(f"找不到真值文件 {reference_path}，请先运行: python src/cli.py build")
        return 2

    references = load_references(reference_path)
    predictions = load_predictions(args.predictions)
    report = evaluate_predictions(
        predictions,
        references,
        tokenizer_name=args.tokenizer,
        include_memory_refs=args.include_memory_refs,
    )

    metrics = report.metrics
    quality = metrics["response_quality"]
    print(f"预测文件: {args.predictions}")
    print(f"真值文件: {reference_path}  ({metrics['n_references']} 条)")
    print(f"可比对: {metrics['n_comparable']} 条\n")
    print("== 自动评测三项 ==")
    print(f"  情绪 Acc              {metrics['emotion_accuracy']:.4f}   (空值率 {metrics['emotion_empty_rate']:.4f})")
    print(
        f"  画像 Acc(三维均值)     {metrics['profile_accuracy_mean_3dim']:.4f}   "
        f"计入维度 {metrics['profile_dimensions_included']}"
    )
    print(
        f"  ↳ 官方口径(空值判错)   {metrics['profile_accuracy_mean_3dim_official']:.4f}"
        f"   ← 这个才是要拿去对标官方的数"
    )
    for dim, score in metrics["profile_dimension_scores"].items():
        print(
            f"      {dim:<20} Acc {score['exact_match_acc']:.4f}  "
            f"官方 {score['exact_match_acc_official']:.4f}  "
            f"空预测率 {score['empty_pred_rate']:.4f}  "
            f"micro-F1 {score['micro_f1']:.4f}  Jaccard {score['macro_jaccard']:.4f}"
        )
    print(
        f"  回复 BLEU-4           {quality['bleu4']:.4f}   "
        f"ROUGE-L {quality['rouge_l_f']:.4f}   分词器 {quality['tokenizer']}"
    )
    bert = metrics["bertscore"]
    if bert.get("computed"):
        print(f"  BERTScore F1          {bert['f1']:.4f}")
    else:
        print(f"  BERTScore             未计算（{bert.get('reason')}）")
    if not args.include_memory_refs:
        mem = metrics["memory_refs_scores"]
        print(
            f"\n  [参考] memory_refs      Acc {mem['exact_match_acc']:.4f}  "
            f"micro-F1 {mem['micro_f1']:.4f}  (未计入画像总分，官方口径待确认)"
        )

    if report.warnings:
        print("\n警告")
        for warning in report.warnings:
            print(f"  - {warning}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"\n完整报告 -> {args.json_out}")
    if args.per_sample_out:
        write_jsonl(args.per_sample_out, report.per_sample)
        print(f"逐样本明细 -> {args.per_sample_out}")
    return 0


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_submission(
        args.submission, input_path=args.input, strict_schema_id=args.strict_schema_id
    )
    print(f"提交文件: {args.submission}")
    print(result.render())
    if args.json_out:
        payload = {
            "total": result.total,
            "ok": result.ok,
            "issues": [issue.__dict__ for issue in result.issues],
        }
        write_json(args.json_out, payload)
        print(f"问题清单 -> {args.json_out}")
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py", description="OGC 情感陪伴赛题的本地数据/评分/校验工具链"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="展开官方数据，落盘样本与数据质量报告")
    p_build.add_argument("--out-dir", default=None, help=f"样本输出目录（默认 {PROCESSED}）")
    p_build.add_argument("--report-dir", default=None, help=f"报告输出目录（默认 {REPORTS}）")
    p_build.add_argument(
        "--keep-turn-mismatch",
        action="store_true",
        help="保留 assistant_response_turn_id 错位的样本（默认丢弃）",
    )
    p_build.add_argument(
        "--keep-true-leaks",
        action="store_true",
        help="保留目标回复已出现在输入历史里的样本（默认丢弃，仅用于人工复核）",
    )
    p_build.add_argument(
        "--no-leak-check",
        action="store_true",
        help="跳过泄漏自检（1.8 万样本约几秒，追求速度时用）",
    )
    p_build.set_defaults(func=cmd_build)

    p_eval = sub.add_parser("eval", help="在验证集上给预测文件打分")
    p_eval.add_argument("--predictions", required=True, help="预测 JSONL 路径")
    p_eval.add_argument("--references", default=None, help="真值 JSONL（默认用 build 产物）")
    p_eval.add_argument(
        "--tokenizer", default="char", choices=("char", "jieba"), help="BLEU/ROUGE 分词器"
    )
    p_eval.add_argument(
        "--include-memory-refs",
        action="store_true",
        help="把 memory_refs 也计入画像维度均值（官方口径未确认，默认不计入）",
    )
    p_eval.add_argument("--json-out", default=None, help="完整指标报告输出路径")
    p_eval.add_argument("--per-sample-out", default=None, help="逐样本明细输出路径")
    p_eval.set_defaults(func=cmd_eval)

    p_validate = sub.add_parser("validate", help="校验提交文件是否符合官方 schema")
    p_validate.add_argument("--submission", required=True, help="待校验的 submission.jsonl")
    p_validate.add_argument("--input", default=None, help="对应输入测试 jsonl，用于核对 ID 覆盖")
    p_validate.add_argument("--strict-schema-id", action="store_true", help="严格按官方 id 正则校验")
    p_validate.add_argument("--json-out", default=None, help="问题清单输出路径")
    p_validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
