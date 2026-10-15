#!/usr/bin/env python3
"""把专家评审材料收集成一个目录，并生成带 sha256 的 MANIFEST.md。

用法::

    python docs/reports/pack_review.py --team 队名 --out deploy/dist/review-package

设计要点：

* **缺失的项显式打印 `[缺]`**，不静默跳过——材料不全会直接影响「方案完整性」得分，
  静默跳过会让人以为已经齐了。
* 每个文件都算 sha256，写进 MANIFEST.md。专家评审关心可复现性；我们自己回看时也能确认
  没有拿旧报告冒充新结果。
* 只用标准库，不引入任何新依赖（全队铁律）。

不复制权重、不复制 `data/processed/`：那些是运行期产物，不属于评审材料。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# (目标目录内的相对路径, 仓库内的来源路径, 说明, 是否必需)
ITEMS: tuple[tuple[str, str, str, bool], ...] = (
    ("01-technical-plan.md", "docs/reports/technical-plan.md", "技术方案", True),
    ("02-experiments.md", "docs/reports/experiments.md", "实验设计与结果", True),
    ("03-reproduce.md", "docs/reports/reproduce.md", "复现步骤", True),
    ("04-review-package.md", "docs/reports/review-package.md", "材料清单", True),
    ("05-data-quality/dataset_summary.json", "data/reports/dataset_summary.json", "数据规模", True),
    ("05-data-quality/train_quality.json", "data/reports/train_quality.json", "训练集质量", True),
    ("05-data-quality/val_quality.json", "data/reports/val_quality.json", "验证集质量", True),
    ("06-leaderboard.tsv", "runs/leaderboard.tsv", "评测台账（由 eval_loop.py 追加）", False),
    ("08-code-review.md", "docs/reports/code-review.md", "代码规范巡检结论", False),
)


def sha256(path: Path) -> str:
    """分块算摘要，避免把大文件整个读进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}GB"


def collect(out_dir: Path) -> tuple[list[tuple[str, str, int, str]], list[str], int]:
    """复制文件，返回 (清单, 缺失项描述, 必需项缺失数量)。

    清单项为 (相对路径, 说明, 字节数, sha256)。
    必需项缺失才让脚本以退出码 1 结束——可选项（如尚未产出的巡检报告）只提示、不阻塞。
    """
    copied: list[tuple[str, str, int, str]] = []
    missing: list[str] = []
    required_missing = 0
    for target_name, source_name, description, required in ITEMS:
        source = REPO_ROOT / source_name
        if not source.is_file():
            tag = "[缺]" if required else "[可选未生成]"
            if required:
                required_missing += 1
            missing.append(
                f"{tag} {target_name}  <-  {source_name}  （{description}）"
            )
            continue
        target = out_dir / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append((target_name, description, target.stat().st_size, sha256(target)))
    return copied, missing, required_missing


def copy_platform_log(out_dir: Path) -> int:
    """把平台运行记录整目录复制过去，返回复制的文件数。"""
    source = REPO_ROOT / "docs" / "platform-log"
    if not source.is_dir():
        return 0
    count = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = out_dir / "07-platform-log" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def render_manifest(
    team: str, out_dir: Path, copied: list[tuple[str, str, int, str]], missing: list[str], log_count: int
) -> str:
    lines = [
        f"# 专家评审材料包 — {team}",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "- 赛道：AI 技术赛道 / 数字人综合情感陪伴对话模型",
        f"- 打包命令：`python docs/reports/pack_review.py --team {team}`",
        "",
        "## 文件清单（sha256 用于确认材料与跑出这些数字的代码版本对应）",
        "",
        "| 文件 | 说明 | 大小 | sha256 |",
        "|---|---|---|---|",
    ]
    for name, description, size, digest in copied:
        lines.append(f"| `{name}` | {description} | {human(size)} | `{digest}` |")
    lines += ["", f"平台运行记录：`07-platform-log/`（{log_count} 个文件）", ""]
    if missing:
        lines += ["## 缺失项", ""]
        lines += [f"- {item}" for item in missing]
        lines += [""]
    lines += [
        "## 说明",
        "",
        "- 权重与 `data/processed/` 不在此目录：它们是运行期产物，不属于评审材料。",
        "- `06-leaderboard.tsv` 由 `tools/eval_loop.py` 每次评测追加一行，**不手写**。",
        "- 平台运行记录必须来自真实上机，不得补造（全队铁律：不许伪造任何数字、日志、截图）。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="收集专家评审材料并生成 MANIFEST.md")
    parser.add_argument("--team", required=True, help="队名（官方报名信息里的名称）")
    parser.add_argument(
        "--out",
        default="deploy/dist/review-package",
        help="输出目录（默认 deploy/dist/review-package）",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    copied, missing, required_missing = collect(out_dir)
    log_count = copy_platform_log(out_dir)
    if log_count == 0:
        missing.append("[缺] 07-platform-log/  （平台运行记录，硬性要求，必须从第一次上机开始记）")
        required_missing += 1

    manifest = render_manifest(args.team, out_dir, copied, missing, log_count)
    (out_dir / "MANIFEST.md").write_text(manifest, encoding="utf-8")

    print(f"材料包目录: {out_dir}")
    for name, description, size, _ in copied:
        print(f"  [有] {name:<38} {human(size):>8}  {description}")
    for item in missing:
        print(f"  {item}")
    if required_missing:
        print(f"\nMANIFEST.md 已写出，但**必需项缺 {required_missing} 个**，材料包不完整。")
    elif missing:
        print(f"\nMANIFEST.md 已写出，必需项齐全（另有 {len(missing)} 个可选项未生成）。")
    else:
        print("\nMANIFEST.md 已写出，材料齐全。")
    return 1 if required_missing else 0


if __name__ == "__main__":
    sys.exit(main())
