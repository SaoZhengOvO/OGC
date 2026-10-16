#!/bin/bash
# 全量 v3 跑完后自动做后处理（纯 CPU，不占卡）。
#
# 为什么要有它：eval_loop.py 只写 score.json，不含逐样本明细；
# 而后处理（逐类 P/R/F1、画像五态分解、badcase 清单）是角色②交付要用的东西。
# 挂成守护进程，跑完立刻出证据，不依赖我这边的笔记本是否还连着。
cd /mnt/workspace/OGC || exit 1

OUT=runs/eval-v3-full
LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "后处理守护启动：等 $OUT/performance_report.json"

for _ in $(seq 1 720); do   # 最长等 12 小时（每 60 秒看一次）
    [ -f "$OUT/performance_report.json" ] && break
    sleep 60
done

if [ ! -f "$OUT/performance_report.json" ]; then
    say "后处理守护超时退出（全量还没跑完）"
    exit 1
fi

say "全量已跑完，开始后处理"

python3 src/cli.py eval \
    --predictions "$OUT/submission.jsonl" \
    --per-sample-out "$OUT/samples.jsonl" \
    >> "$OUT/postprocess.log" 2>&1
say "逐样本明细 -> $OUT/samples.jsonl"

python3 tools/emotion_profile/report.py \
    --per-sample "$OUT/samples.jsonl" \
    --json-out "$OUT/report.json" \
    --top-bad 30 \
    >> "$OUT/postprocess.log" 2>&1
say "逐类报告 -> $OUT/report.json"

python3 tools/emotion_profile/badcase_pack.py \
    --per-sample "$OUT/samples.jsonl" \
    --per-class 8 \
    --out "$OUT/badcase.md" \
    >> "$OUT/postprocess.log" 2>&1
say "badcase 清单 -> $OUT/badcase.md"

# 顺手把 20 条 smoke 与全量的关键数字并排放一份摘要
python3 - "$OUT" <<'PY' >> "$OUT/postprocess.log" 2>&1
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
full = json.loads((out / "score.json").read_text(encoding="utf-8"))["metrics"]
print("=== 全量 2354 条 v3 ===")
print("emotion_accuracy      ", full["emotion_accuracy"])
print("profile_mean_3dim     ", full["profile_accuracy_mean_3dim"])
for dim, s in full["profile_dimension_scores"].items():
    print(f"  {dim:20s} EMA={s['exact_match_acc']:.4f} P={s['micro_precision']:.4f} F1={s['micro_f1']:.4f}")
print("bleu4 / rouge_l       ", full["response_quality"]["bleu4"], full["response_quality"]["rouge_l_f"])
PY
say "摘要已写入 postprocess.log"

# 释放自己的 lock
rm -f runs/.gpu-lock.d/role2-v3-full.lock runs/.gpu-lock.d/role2-queued.lock
say "后处理完成，已释放 lock"
