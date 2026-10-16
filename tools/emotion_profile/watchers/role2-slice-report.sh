#!/bin/bash
# 切片跑完后自动出对比表（纯 CPU，不占卡）。等 v4-slice200 出分即触发。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "切片报告守护启动：等 runs/eval-v4-slice200/score.json"

for _ in $(seq 1 360); do
    [ -f runs/eval-v4-slice200/score.json ] && break
    sleep 30
done

if [ ! -f runs/eval-v4-slice200/score.json ]; then
    say "切片报告守护超时（v4-slice200 没出分）"
    exit 1
fi

say "切片跑完，开始出对比"
exec >> runs/.slice-report.log 2>&1

SLICE=/mnt/workspace/OGC/runs/slice200

# 0) 先跑预检：它会造出 val_input.jsonl / 完美预测 / 下限预测
python3 tools/emotion_profile/preflight_slice.py --slice runs/slice200

# 1) 三份逐样本明细（compare_arms 读的是 per-sample，不是 predictions）
python3 src/cli.py eval --predictions "$SLICE/preflight_floor.jsonl" \
    --references "$SLICE/val_references.jsonl" \
    --per-sample-out "$SLICE/floor.samples.jsonl"
for tag in v3-slice200 v4-slice200; do
    python3 src/cli.py eval --predictions "runs/eval-$tag/submission.jsonl" \
        --references "$SLICE/val_references.jsonl" \
        --per-sample-out "runs/eval-$tag/samples.jsonl"
done

# 2) 多臂对比：下限 / v3 / v4
python3 tools/emotion_profile/compare_arms.py \
    --arm "下限=全anxiety+画像留空=runs/slice200/floor.samples.jsonl" \
    --arm "v3=runs/eval-v3-slice200/samples.jsonl" \
    --arm "v4=runs/eval-v4-slice200/samples.jsonl" \
    --per-sample-diff \
    --out runs/SLICE-COMPARISON.md

echo "[$(date '+%m-%d %H:%M:%S')] 对比表 -> runs/SLICE-COMPARISON.md"
