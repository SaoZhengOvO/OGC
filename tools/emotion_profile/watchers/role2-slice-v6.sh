#!/bin/bash
# v6 切片：等 v5a-slice200 出分后，跑 v6 的 200 条切片，并出一份含 v6 的对比表。
#
# v6 是唯一改动 = 修正 fear 词表的候选（证据见报告 §十八）。
# 同样用独立守护串在链尾，不改动正在执行的其它脚本。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "v6 切片守护启动：等 runs/eval-v5a-slice200/score.json"

for _ in $(seq 1 900); do
    [ -f runs/eval-v5a-slice200/score.json ] && break
    sleep 30
done

if [ ! -f runs/eval-v5a-slice200/score.json ]; then
    say "v6 切片守护超时"
    exit 1
fi

sleep 30
say "开始 v6 切片（200 条，唯一改动 = 修正 fear 词表）"
python3 tools/eval_loop.py \
    --tag v6-slice200 \
    --source /mnt/workspace/OGC/runs/slice200/val_samples.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02-v6 \
    >> runs/v6-slice200.log 2>&1
say "v6-slice200 结束（退出码 $?）"

python3 src/cli.py eval \
    --predictions runs/eval-v6-slice200/submission.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --per-sample-out runs/eval-v6-slice200/samples.jsonl \
    >> runs/.slice-report.log 2>&1

python3 tools/emotion_profile/compare_arms.py \
    --arm "下限=runs/slice200/floor.samples.jsonl" \
    --arm "v3=runs/eval-v3-slice200/samples.jsonl" \
    --arm "v4=runs/eval-v4-slice200/samples.jsonl" \
    --arm "v5a=runs/eval-v5a-slice200/samples.jsonl" \
    --arm "v6=runs/eval-v6-slice200/samples.jsonl" \
    --paired v3 \
    --out runs/SLICE-COMPARISON-v6.md \
    >> runs/.slice-report.log 2>&1

say "含 v6 的对比表 -> runs/SLICE-COMPARISON-v6.md"
