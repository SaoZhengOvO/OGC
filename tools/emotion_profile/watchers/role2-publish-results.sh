#!/bin/bash
# 全量数字落地后，自动写成**仓库内**的交付文件（runs/ 是 gitignore 的，不进版本库）。
#
# 缺口：postprocess 与 full-compare 两个守护都把产物写进 runs/，而 runs/ 被 gitignore。
# 目标是「交付全量 2354 条的确认数字」，那就必须落进 docs/reports/ 里可提交的文件。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "结果发布守护启动：等 runs/FULL-COMPARISON.md"

for _ in $(seq 1 900); do
    [ -f runs/FULL-COMPARISON.md ] && break
    sleep 60
done

if [ ! -f runs/FULL-COMPARISON.md ]; then
    say "结果发布守护超时"
    exit 1
fi

say "全量对比已就绪，开始写仓库内交付文件"

python3 - <<'PY'
import json
from pathlib import Path

repo = Path("/mnt/workspace/OGC")
full = repo / "runs" / "eval-v3-full"
out = repo / "docs" / "reports" / "role2-full-val-results.md"

score = json.loads((full / "score.json").read_text(encoding="utf-8"))["metrics"]
perf = json.loads((full / "performance_report.json").read_text(encoding="utf-8"))
comparison = (repo / "runs" / "FULL-COMPARISON.md").read_text(encoding="utf-8")

# 从对比表里抽出「是否显著」那一列，方便一眼看结论
lines = [line for line in comparison.splitlines() if line.startswith("| ") and "|" in line]

body = []
body.append("# 角色② 全量 2354 条确认数字（v3）")
body.append("")
body.append("> 由 `role2-publish-results.sh` 在全量跑完后自动生成。")
body.append("> 产物目录 `runs/eval-v3-full/`（gitignore，不随仓库提交）；本文件是**可提交的证据副本**。")
body.append("")
body.append("## 一、v3 在 val 全量上的成绩")
body.append("")
body.append("| 指标 | 值 |")
body.append("|---|---|")
body.append(f"| 样本数 | {score['n_comparable']} |")
body.append(f"| 情绪 Acc | **{score['emotion_accuracy']:.4f}** |")
body.append(f"| 画像 Acc（三维均值） | **{score['profile_accuracy_mean_3dim']:.4f}** |")
for dim, value in score["profile_dimension_scores"].items():
    body.append(
        f"| └ {dim} | EMA {value['exact_match_acc']:.4f}"
        f"（P {value['micro_precision']:.4f} / F1 {value['micro_f1']:.4f}） |"
    )
quality = score["response_quality"]
body.append(f"| 回复 BLEU-4 / ROUGE-L | {quality['bleu4']:.4f} / {quality['rouge_l_f']:.4f} |")
body.append(f"| 空回复率 | {quality['empty_response_rate']:.4f} |")
body.append(f"| 端到端延迟（前 {perf.get('rounds')} 条） | 平均 {perf.get('average_latency_ms', 0):.0f} ms |")
body.append("")
body.append("> ⚠️ **回复质量那两行不代表最终系统。** 这次跑的是 `--prompt-dir runs/p02-only`，")
body.append("> 只带角色② 的 `emotion_profile.md`，**没有**角色③ 的 `response.md`。")
body.append("> 这么跑是为了让情绪/画像的成绩**只归因于②**（控制变量）——")
body.append("> 回复类指标要等带上 `response.md` 的全量跑才有意义，不要拿这里的数字当系统分。")
body.append("")
body.append("## 二、相对零成本下限的配对显著性")
body.append("")
body.append("零成本下限：情绪一律 `anxiety`（0.3190）、画像三维一律留空（0.6283）。")
body.append("")
body.extend(lines)
body.append("")
body.append("## 三、判读规则（跑之前就定好，避免事后解释）")
body.append("")
body.append("| 条件 | 结论 |")
body.append("|---|---|")
body.append("| 情绪 > 0.3190 且 画像 > 0.6283 | 相对零成本下限**站住了** |")
body.append("| 其中一项不达标 | **如实报告未达标**，并说明是哪一维 |")
body.append("| 与下限的差值为正但 `p ≥ 0.05` | 只能说「点估计更好」，**不能说"守住"** |")
body.append("")
body.append("## 四、参考：其它口径")
body.append("")
body.append("| 对照 | 情绪 Acc | 画像 Acc |")
body.append("|---|---|---|")
body.append("| E1 裸模型（全量 2354，`docs/reports/experiments.md` §E1） | 0.2464 | 0.5615 |")
body.append("| 零成本下限 | 0.3190 | 0.6283 |")
body.append("| n=20 smoke 上的 v3（**偏易约 25pp，仅供参考**） | 0.6000 | 0.5000 |")
body.append("")

out.write_text("\n".join(body) + "\n", encoding="utf-8")
print(f"已写入 {out}")
PY

say "仓库内交付文件 -> docs/reports/role2-full-val-results.md（下一步：git add 后提交）"
