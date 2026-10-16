#!/bin/bash
# 决策切片守护：等 v3 全量起来之后，在「第二个 GPU 卡位」上跑 v3 vs v4 的切片对比。
#
# 为什么要有它：n=20 smoke 严重不代表全量（style 金标为空率 0.20 vs 全量 0.5174），
# 在 smoke 上达标的规则可能在全量上是负收益。等距 200 条切片与全量的偏差只有 0.0024，
# 8 分钟就能跑完，适合做「v4 要不要推上全量」的决策依据。
#
# 占用说明：v3 全量（排队器起的，占 1 个卡位）+ 本脚本（占第 2 个卡位）= 2 个，符合
# runs/GPU-RULES.md 的「同时最多 2 个 GPU 任务」。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "切片守护启动：等 train_sft.py 结束"

while pgrep -f "train_sft.py" >/dev/null 2>&1; do
    sleep 30
done
say "SFT 已结束"

# 等排队器把 v3 全量起起来（最多等 5 分钟），确认它真的在跑
started=0
for _ in $(seq 1 30); do
    if pgrep -f "tag v3-full" >/dev/null 2>&1; then
        say "v3 全量已在跑，开始切片对比"
        started=1
        break
    fi
    sleep 10
done
# 等不到也要往下走（不能永远阻塞），但**必须留下痕迹** ——
# 否则"v3 全量没起来"这件事在日志里完全看不见，事后无从判断当时到底跑了几个任务。
if [ "$started" -eq 0 ]; then
    say "⚠️ 5 分钟内没看到 v3 全量在跑 —— 仍然继续跑切片（不阻塞），"
    say "    但请事后确认排队器是否正常，这会影响『当时几个任务』的记载"
fi

cat > runs/.gpu-lock.d/role2-slice.lock <<EOF
状态: 占用
谁: ② zjdhhds
跑什么: eval_loop --tag v3-slice200 / v4-slice200（等距 200 条决策切片）
起: $(date '+%m-%d %H:%M')
止: 预计 20 分钟（与 v3 全量共享卡，会慢一些）
显存: 3B fp16 约 8-10 GB
PID: $$
EOF

for tag_dir in "v3-slice200:runs/p02-only" "v4-slice200:runs/p02-v4"; do
    tag="${tag_dir%%:*}"
    pdir="/mnt/workspace/OGC/${tag_dir##*:}"
    say "开始 $tag（提示词 $pdir）"
    python3 tools/eval_loop.py \
        --tag "$tag" \
        --source /mnt/workspace/OGC/runs/slice200/val_samples.jsonl \
        --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
        --inference-script wired \
        --prompt-dir "$pdir" \
        >> "runs/$tag.log" 2>&1
    say "$tag 结束（退出码 $?），结果见 runs/eval-$tag/score.json 与 runs/leaderboard.tsv"
done

rm -f runs/.gpu-lock.d/role2-slice.lock
say "切片对比完成，已释放 lock"
