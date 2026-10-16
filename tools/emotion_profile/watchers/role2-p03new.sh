#!/bin/bash
# 第三个参照跑：②的 v3 + ③的【新版】response.md，直接回答"③ 的格式修复有没有效"。
#
# 为什么需要：
#   box 上 prompts/response.md 还是旧版（sha f3769150，23:52）——
#   ③ 把新版（sha 1b48983a，含"分段必须用转义符 \n\n，不能写真实换行"）推到了 GitHub，
#   但没同步到 box。而我的集成守护是运行时 cp prompts/response.md → 只会用到旧版。
#
#   ② 不改 prompts/（那是③的文件）。所以这里把③的新版**只读复制**到 runs/p03-new/，
#   另起一个臂跑，与"旧版集成跑"形成对照 —— 直接量出③这次修改的效果。
#
# 对照关系：
#   integ-slice200    = ② v3  + ③ 旧版   （预期：10% 解析失败）
#   p03new-slice200   = ② v3  + ③ 新版   （若③修对了：解析失败应接近 0）
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "③新版参照守护启动：等 runs/eval-v8+p03-slice200/score.json"

for _ in $(seq 1 900); do
    [ -f runs/eval-v8+p03-slice200/score.json ] && break
    sleep 30
done

if [ ! -f runs/eval-v8+p03-slice200/score.json ]; then
    say "③新版参照守护超时"
    exit 1
fi

sleep 30

PAIR=runs/p02-v3+p03new
mkdir -p "$PAIR"
cp runs/p02-only/emotion_profile.md "$PAIR/emotion_profile.md"
if [ ! -f runs/p03-new/response.md ]; then
    say "⚠️ 缺 runs/p03-new/response.md（③的新版）—— 退出"
    exit 1
fi
cp runs/p03-new/response.md "$PAIR/response.md"
# 组装后校验：两篇必须都在（缺一篇 → 标签正确、内容错的静默结果）
for f in emotion_profile.md response.md; do
    if [ ! -s "$PAIR/$f" ]; then
        say "❌ $PAIR/$f 缺失或为空 —— 拒绝用不完整的提示词跑"
        exit 1
    fi
done
say "③新版参照目录 $PAIR：$(ls -1 "$PAIR" | tr '\n' ' ')"
say "  ③新版 response.md sha256: $(sha256sum runs/p03-new/response.md | cut -c1-16)"

say "开始 ③新版参照跑（200 条）"
python3 tools/eval_loop.py \
    --tag p03new-slice200 \
    --source /mnt/workspace/OGC/runs/slice200/val_samples.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02-v3+p03new \
    >> runs/p03new-slice200.log 2>&1
say "③新版参照跑结束（退出码 $?）"

echo "=== ③ 的格式修复值多少：旧版 vs 新版对照 ===" | tee -a "$LOG"
python3 tools/emotion_profile/format_health.py \
    --tag integ-slice200 --tag p03new-slice200 --tag v8+p03-slice200 2>&1 | tee -a "$LOG"

python3 src/cli.py eval \
    --predictions runs/eval-p03new-slice200/submission.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --per-sample-out runs/eval-p03new-slice200/samples.jsonl \
    >> runs/.slice-report.log 2>&1

python3 tools/emotion_profile/compare_arms.py \
    --arm "集成②③旧=runs/eval-integ-slice200/samples.jsonl" \
    --arm "集成②③新=runs/eval-p03new-slice200/samples.jsonl" \
    --paired "集成②③旧" \
    --out runs/P03-OLD-VS-NEW.md \
    >> runs/.slice-report.log 2>&1

say "③旧版 vs 新版对照 -> runs/P03-OLD-VS-NEW.md"
