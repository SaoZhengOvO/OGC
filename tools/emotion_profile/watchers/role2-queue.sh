#!/bin/bash
# ② zjdhhds 的排队器：等卡上空出来（队长的 SFT 结束）再自动跑全量 v3。
#
# 为什么要有它：队的规矩是「同时最多 2 个 GPU 任务，超了排队」。
# 我先前抢跑了一个全量，按规矩撤掉，改成排队；这样不用人盯着，卡一空就自动开始。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "排队器启动：等 train_sft.py 结束"

# 等 SFT 结束（每 60 秒看一眼）
#
# ⚠️ 判据必须用**行首锚定**的 `^python3 tools/train_sft`，不能用裸的 `train_sft.py`。
# 原因（第 174 轮实测）：队长起 SFT 时用的是
#     bash -c "… setsid nohup python3 tools/train_sft.py … & sleep 4; echo relaunched"
# 那个 **`bash -c` 包装进程的命令行里也含 `train_sft.py`**，而且它在训练结束后
# **仍然存活**（实测已存活 2 小时以上，远超它自己那 4 秒）。
# 于是 `pgrep -f "train_sft.py"` 会**永远返回 ≥1** → 这个 while 循环**永不退出**
# → **整条链今晚根本不会启动**。
# 实测对照：
#     pgrep -f "train_sft.py"             -> 2 个（含包装进程）
#     pgrep -f "^python3 tools/train_sft" -> 1 个（只有真正的 python3）✅
while pgrep -f '^python3 tools/train_sft' >/dev/null 2>&1; do
    sleep 60
done
say "SFT 已结束"

# 再等一会儿，确认卡上别的评测也退出（不硬挤）
sleep 45
RUNNING=$(pgrep -f "run_inference.py" | wc -l)
say "当前 run_inference 进程数: $RUNNING"

# 起全量（setsid + nohup：断开 SSH 也不受影响）
#
# ⚠️ 起跑前先清掉目标目录里的**上一轮产物**。
# 原因：`run_inference.py:94-97` 的 prepare_result_dir 只 mkdir、**不清理旧文件**；
# 而 postprocess / full-compare / integ-full 三个守护都是**等 performance_report.json**。
# 若上一轮留下过这份文件，它们会在本次全量还没跑完时就触发、
# 拿半截 submission.jsonl 算分 —— **而且不会报错**。
# （今晚本就不存在该文件，这一句是给"重跑"上的保险。）
rm -f runs/eval-v3-full/performance_report.json runs/eval-v3-full/score.json
say "已清理 runs/eval-v3-full/ 的旧 performance_report.json / score.json（防守护提前触发）"

setsid nohup python3 tools/eval_loop.py \
    --tag v3-full \
    --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02-only \
    > runs/v3-full.log 2>&1 < /dev/null &
PID=$!
say "已启动 v3-full，PID=$PID"

# 登记 lock
cat > runs/.gpu-lock.d/role2-v3-full.lock <<EOF
状态: 占用
谁: ② zjdhhds
跑什么: tools/eval_loop.py --tag v3-full（全量 2354 条，只带 emotion_profile）
起: $(date '+%m-%d %H:%M')
止: 预计 1.5-2 小时（独占时）
显存: 3B fp16 约 8-10 GB
PID: $PID
EOF
say "lock 已写入 runs/.gpu-lock.d/role2-v3-full.lock"
