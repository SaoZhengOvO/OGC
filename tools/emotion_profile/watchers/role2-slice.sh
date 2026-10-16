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

# === 并发闸门：确认卡位上真的只有「我的 v3 全量」这一个任务 ===
# 为什么需要：队长的 SFT 一结束，可能**别人也会起评测**。此时我这边
# queue（v3 全量）+ slice（本脚本）= 2 个任务，若再有第三方就是 3 个 ——
# 违反 runs/GPU-RULES.md 的「同时最多 2 个」。而这类越界**不会报错**，
# 只会让所有人的跑都变慢、并且事后说不清当时到底几个任务。
# 判据：数 run_inference.py 进程数。我自己那一个算 1，超过 1 说明还有别人在跑。
for _ in $(seq 1 60); do
    RUNNING=$(pgrep -f "run_inference.py" | wc -l)
    if [ "$RUNNING" -le 1 ]; then
        say "并发闸门通过：当前 run_inference 进程数 = $RUNNING（只有我自己的）"
        break
    fi
    say "并发闸门等待：run_inference 进程数 = $RUNNING（>1，说明有第三方在跑），等 60 秒再看"
    sleep 60
done
FINAL=$(pgrep -f "run_inference.py" | wc -l)
if [ "$FINAL" -gt 1 ]; then
    say "⚠️ 等了一小时仍是 $FINAL 个 run_inference —— 仍然起切片，但**这会让并发超过 2**，"
    say "    请事后核对 runs/GPU-RULES.md 的规矩；已如实记录在这里。"
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
    # 每个臂起跑前复查一次并发（防止中途有第三方加入）
    HERE=$(pgrep -f "run_inference.py" | wc -l)
    if [ "$HERE" -gt 1 ]; then
        say "⚠️ 起 $tag 前 run_inference 进程数 = $HERE（>1）—— 再等一轮"
        for _ in $(seq 1 30); do
            HERE=$(pgrep -f "run_inference.py" | wc -l)
            [ "$HERE" -le 1 ] && break
            sleep 60
        done
    fi
    say "开始 $tag（提示词 $pdir），起跑时 run_inference 进程数 = $(pgrep -f 'run_inference.py' | wc -l)"
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
