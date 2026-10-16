#!/bin/bash
# 集成切片：把②的 emotion_profile.md 与③的 response.md **同时**挂上跑一次 200 条切片。
#
# 为什么要跑：②所有评测都是 --prompt-dir runs/p02-* 隔离跑的（为了控制变量），
# 所以「两篇提示词同时生效」时的真实表现**至今没人测过**。
# 而最终提交的镜像里两篇都会拼进去（deploy/build.sh 拷贝 repo 根 prompts/）。
# 这一轮就是量出「集成之后情绪/画像会不会被③的回复提示词带偏」。
#
# 挂在②的链路最尾（等 v6-slice200 出分），仍然只占第 2 个卡位。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "集成切片守护启动：等 runs/eval-v6-slice200/score.json"

for _ in $(seq 1 900); do
    [ -f runs/eval-v6-slice200/score.json ] && break
    sleep 30
done

if [ ! -f runs/eval-v6-slice200/score.json ]; then
    say "集成切片守护超时"
    exit 1
fi

sleep 30

# 组装集成提示词目录：只在 runs/ 下建，不动 prompts/
INTEG=runs/p02+p03
mkdir -p "$INTEG"
cp prompts/emotion_profile.md "$INTEG/emotion_profile.md"
if [ -f prompts/response.md ]; then
    cp prompts/response.md "$INTEG/response.md"
else
    say "⚠️ prompts/response.md 不存在 —— ③ 还没交付或被删了，集成跑无意义，退出"
    exit 1
fi
say "集成提示词目录 $INTEG：$(ls -1 "$INTEG" | tr '\n' ' ')"

say "开始集成切片（200 条，②+③ 两篇提示词）"
python3 tools/eval_loop.py \
    --tag integ-slice200 \
    --source /mnt/workspace/OGC/runs/slice200/val_samples.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02+p03 \
    >> runs/integ-slice200.log 2>&1
say "integ-slice200 结束（退出码 $?）"

python3 src/cli.py eval \
    --predictions runs/eval-integ-slice200/submission.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --per-sample-out runs/eval-integ-slice200/samples.jsonl \
    >> runs/.slice-report.log 2>&1

# 关键对比：隔离的 v3  vs  集成的 v3+③
python3 tools/emotion_profile/compare_arms.py \
    --arm "隔离v3=runs/eval-v3-slice200/samples.jsonl" \
    --arm "集成v3+③=runs/eval-integ-slice200/samples.jsonl" \
    --paired "隔离v3" \
    --per-sample-diff \
    --out runs/INTEGRATION-COMPARISON.md \
    >> runs/.slice-report.log 2>&1

say "集成对比 -> runs/INTEGRATION-COMPARISON.md（看③的回复提示词有没有带偏情绪/画像）"
