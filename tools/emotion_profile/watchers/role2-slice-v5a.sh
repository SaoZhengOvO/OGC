#!/bin/bash
# v5a 切片：等 v4-slice200 出分后，在同一个卡位跑 v5a 的 200 条切片，并出一份含 v5a 的对比表。
#
# 为什么要单独挂：切片守护（role2-slice.sh）已经在跑 v3/v4 两轮，
# 中途改它的脚本会让正在执行的 bash 读到半截文件。所以用独立守护串在后面。
# 输出文件名与 role2-slice-report.sh 的不同，避免互相覆盖。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "v5a 切片守护启动：等 runs/eval-v4-slice200/score.json"

for _ in $(seq 1 720); do
    [ -f runs/eval-v4-slice200/score.json ] && break
    sleep 30
done

if [ ! -f runs/eval-v4-slice200/score.json ]; then
    say "v5a 切片守护超时"
    exit 1
fi

# 等对比表先出完，避免两个脚本同时写 runs/ 下的东西
for _ in $(seq 1 40); do
    [ -f runs/SLICE-COMPARISON.md ] && break
    sleep 30
done
sleep 30

say "开始 v5a 切片（200 条）"
python3 tools/eval_loop.py \
    --tag v5a-slice200 \
    --source /mnt/workspace/OGC/runs/slice200/val_samples.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02-v5a \
    >> runs/v5a-slice200.log 2>&1
say "v5a-slice200 结束（退出码 $?）"

python3 src/cli.py eval \
    --predictions runs/eval-v5a-slice200/submission.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --per-sample-out runs/eval-v5a-slice200/samples.jsonl \
    >> runs/.slice-report.log 2>&1

python3 tools/emotion_profile/compare_arms.py \
    --arm "下限=runs/slice200/floor.samples.jsonl" \
    --arm "v3=runs/eval-v3-slice200/samples.jsonl" \
    --arm "v4=runs/eval-v4-slice200/samples.jsonl" \
    --arm "v5a=runs/eval-v5a-slice200/samples.jsonl" \
    --per-sample-diff \
    --out runs/SLICE-COMPARISON-v5a.md \
    >> runs/.slice-report.log 2>&1

say "含 v5a 的对比表 -> runs/SLICE-COMPARISON-v5a.md"
