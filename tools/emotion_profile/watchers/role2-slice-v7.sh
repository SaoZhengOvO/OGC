#!/bin/bash
# v7 切片：等 v6-slice200 出分后跑。v7 的唯一主题 = 补第二步的情境通路（见报告 §二十）。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "v7 切片守护启动：等 runs/eval-v6-slice200/score.json"

for _ in $(seq 1 900); do
    [ -f runs/eval-v6-slice200/score.json ] && break
    sleep 30
done

if [ ! -f runs/eval-v6-slice200/score.json ]; then
    say "v7 切片守护超时"
    exit 1
fi

sleep 30
say "开始 v7 切片（200 条，主题 = 补第二步通路）"
python3 tools/eval_loop.py \
    --tag v7-slice200 \
    --source /mnt/workspace/OGC/runs/slice200/val_samples.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02-v7 \
    >> runs/v7-slice200.log 2>&1
say "v7-slice200 结束（退出码 $?）"

python3 src/cli.py eval \
    --predictions runs/eval-v7-slice200/submission.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --per-sample-out runs/eval-v7-slice200/samples.jsonl \
    >> runs/.slice-report.log 2>&1

python3 tools/emotion_profile/compare_arms.py \
    --arm "下限=runs/slice200/floor.samples.jsonl" \
    --arm "v3=runs/eval-v3-slice200/samples.jsonl" \
    --arm "v4=runs/eval-v4-slice200/samples.jsonl" \
    --arm "v5a=runs/eval-v5a-slice200/samples.jsonl" \
    --arm "v6=runs/eval-v6-slice200/samples.jsonl" \
    --arm "v7=runs/eval-v7-slice200/samples.jsonl" \
    --paired v3 \
    --out runs/SLICE-COMPARISON-v7.md \
    >> runs/.slice-report.log 2>&1

say "含 v7 的对比表 -> runs/SLICE-COMPARISON-v7.md"
