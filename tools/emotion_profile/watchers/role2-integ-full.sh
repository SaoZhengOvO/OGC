#!/bin/bash
# 「集成系统全量」——**门控式**：只有队长明确批准后才跑。
#
# 为什么门控：
#   issue #3 里②问过队长「最终提交前是否要以集成数字为准重跑全量」，**尚未答复**。
#   ②不擅自占 1.6 小时卡；但如果队长同意，希望能立刻开跑、不要再等一轮沟通。
#   所以这里用旗标文件做闸门：**队长只需 touch 一下，它就会自动跑。**
#
# 启用方式（队长）：
#   touch /mnt/workspace/OGC/runs/.go-integrated-full
#   （或者让②代跑，等价于同一步）
#
# 跑什么：
#   ②的 v3 emotion_profile.md + ③的【新版】response.md，全量 2354 条。
#   这是目前已知的最佳组合（③新版修掉了 10% 解析失败，见 LOG §21）。
#   —— 若 ③ 届时已把新版同步到 prompts/，也可以把 --prompt-dir 改成仓库根 prompts/
#      与镜像行为完全一致；这里用 runs/p02-v3+p03new 是为了不依赖别人是否同步。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
FLAG=runs/.go-integrated-full

say "集成全量守护启动（门控式）：等旗标 $FLAG，且等 v3 全量结束"

# 第一阶段：等旗标（最多 8 小时；不点就一直等，不占卡）
for _ in $(seq 1 96); do
    [ -f "$FLAG" ] && break
    sleep 300
done
if [ ! -f "$FLAG" ]; then
    say "集成全量守护退出：8 小时内未收到旗标（队长未批准）—— 未占用任何卡"
    exit 0
fi
say "收到旗标 $FLAG，等待 v3 全量结束"

# 第二阶段：等卡位 A 空出来（v3 全量结束）
for _ in $(seq 1 240); do
    [ -f runs/eval-v3-full/performance_report.json ] && break
    sleep 60
done
if [ ! -f runs/eval-v3-full/performance_report.json ]; then
    say "集成全量守护退出：v3 全量一直没结束"
    exit 1
fi

# 组装提示词目录（只读复制；不改 prompts/）
PAIR=runs/p02-v3+p03new
mkdir -p "$PAIR"
cp runs/p02-only/emotion_profile.md "$PAIR/emotion_profile.md"
cp runs/p03-new/response.md "$PAIR/response.md"
say "集成全量提示词目录：$(ls -1 "$PAIR" | tr '\n' ' ')"
say "  ② emotion_profile.md sha: $(sha256sum "$PAIR/emotion_profile.md" | cut -c1-16)"
say "  ③ response.md        sha: $(sha256sum "$PAIR/response.md" | cut -c1-16)"

cat > runs/.gpu-lock.d/role2-integ-full.lock <<EOF
状态: 占用
谁: ② zjdhhds（队长批准的集成全量）
跑什么: tools/eval_loop.py --tag integ-full（全量 2354，②v3 + ③新版）
起: $(date '+%m-%d %H:%M')
止: 预计 1.5-2 小时
显存: 3B fp16 约 8-10 GB
EOF

say "开始集成全量（2354 条）"
# 注意：全量跑**不传 --source / --references**，用默认值（data/processed/val_*.jsonl = 全量 2354）。
# 切片才需要显式指到 runs/slice200/。
python3 tools/eval_loop.py \
    --tag integ-full \
    --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02-v3+p03new \
    >> runs/integ-full.log 2>&1
say "集成全量结束（退出码 $?）"

rm -f runs/.gpu-lock.d/role2-integ-full.lock
say "已释放 lock"
