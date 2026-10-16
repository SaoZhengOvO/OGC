#!/bin/bash
# 第三个参照跑：②的 v3 + ③的【旧版】response.md，
# 与 integration 臂（②的 v3 + ③的【现行版】）构成 A/B，
# 直接回答"③ 对『分段必须用 \n\n』这条的**强调**值多少"。
#
# 为什么需要（这一段的来龙去脉，含两次走偏）：
#   ① 最初 box 上 prompts/response.md 是 ③ 的早期版（f3769150，1901 字节），
#      ③ 的修改版还没同步 → 我另起一臂，把修改版只读复制到 runs/p03-new/。
#   ② 但 ③ 后来把 prompts/response.md 同步成了 639ce8c4（2118 字节），
#      于是两个臂用的**是同一份**，对照失效（会白占一次卡）。
#   ③ 现在改成：本臂用**旧版**（runs/p03-old/response.md = f3769150），
#      integration 臂用 prompt/ 里的**现行版**（639ce8c4）。
#
# 两版的真正差别：对「分段要写成 \n\n 转义、不能写真实换行」这条，
#   旧版：顺带一句"…需要分段时写转义符 \n\n"
#   现行版：单列一条并注明"**这是本任务最容易出错的地方**"（强调 + 解释）
# 所以本臂测的是**强调**的效果，不是"有没有这条要求"。
#
# 对照关系：
#   integ-slice200    = ② v3  + ③ **现行版**（强调版 \n\n）
#   p03new-slice200   = ② v3  + ③ **旧版**  （简述版 \n\n）
# 预登记预测#2 相应表述为：若"强调"有效 → p03new（旧版）的解析失败率应**高于**
# integ（现行版）；反之说明真实换行不是主因，§19.2d 的机制判断要收回。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "③旧版参照守护启动：等 runs/eval-v8+p03-slice200/score.json"

for _ in $(seq 1 900); do
    [ -f runs/eval-v8+p03-slice200/score.json ] && break
    sleep 30
done

if [ ! -f runs/eval-v8+p03-slice200/score.json ]; then
    say "③旧版参照守护超时"
    exit 1
fi

sleep 30

PAIR=runs/p02-v3+p03new
mkdir -p "$PAIR"
cp runs/p02-only/emotion_profile.md "$PAIR/emotion_profile.md"
# ⚠️ 这里故意用 **③ 的旧版**（runs/p03-old/response.md = f3769150, 1901 字节），
# 而不是 ③ 的现行版（prompts/response.md = 639ce8c4, 2118 字节）。
# 原因：两者对「分段必须用转义符 \n\n」这条的**强调程度**不同 ——
# 旧版是顺带一句"需要分段时写转义符 \n\n"，现行版单列一条并注明
# "这是本任务最容易出错的地方"。而 integration 臂用的就是 prompt/ 里的现行版，
# 所以本臂用旧版，两者才构成 **③旧 vs ③现行** 的 A/B，
# 才能检验预登记预测#2（"③ 的格式修复把解析失败压到 ≈0"）。
# 曾经本臂也指向现行版 —— 那样两个臂测同一件事，对照失效。
if [ ! -f runs/p03-old/response.md ]; then
    say "⚠️ 缺 runs/p03-old/response.md（③ 的旧版）—— 退出"
    exit 1
fi
cp runs/p03-old/response.md "$PAIR/response.md"
# 组装后校验：两篇必须都在（缺一篇 → 标签正确、内容错的静默结果）
for f in emotion_profile.md response.md; do
    if [ ! -s "$PAIR/$f" ]; then
        say "❌ $PAIR/$f 缺失或为空 —— 拒绝用不完整的提示词跑"
        exit 1
    fi
done
say "③旧版参照目录 $PAIR：$(ls -1 "$PAIR" | tr '\n' ' ')"
say "  ③旧版 response.md sha256: $(sha256sum runs/p03-old/response.md | cut -c1-16)"

say "开始 ③旧版参照跑（200 条）"
python3 tools/eval_loop.py \
    --tag p03new-slice200 \
    --source /mnt/workspace/OGC/runs/slice200/val_samples.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02-v3+p03new \
    >> runs/p03new-slice200.log 2>&1
say "③旧版参照跑结束（退出码 $?）"

echo "=== ③ 对「\\n\\n 转义」这条的强调值多少：现行版 vs 旧版 ===" | tee -a "$LOG"
python3 tools/emotion_profile/format_health.py \
    --tag integ-slice200 --tag p03new-slice200 --tag v8+p03-slice200 2>&1 | tee -a "$LOG"

python3 src/cli.py eval \
    --predictions runs/eval-p03new-slice200/submission.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --per-sample-out runs/eval-p03new-slice200/samples.jsonl \
    >> runs/.slice-report.log 2>&1

# 臂名必须与路径**对得上**：
#   integ-slice200  用的 prompts/response.md（= ③ 现行版，强调版 \n\n）
#   p03new-slice200 用的 runs/p03-old/response.md（= ③ 旧版，简述版 \n\n）
# 之前这里写反了（把现行标成"旧"）—— 标签与内容不符正是本报告反复在修的那一类问题。
python3 tools/emotion_profile/compare_arms.py \
    --arm "集成②③现行=runs/eval-integ-slice200/samples.jsonl" \
    --arm "集成②③旧=runs/eval-p03new-slice200/samples.jsonl" \
    --paired "集成②③现行" \
    --out runs/P03-OLD-VS-NEW.md \
    >> runs/.slice-report.log 2>&1

say "③现行版 vs 旧版对照 -> runs/P03-OLD-VS-NEW.md"
