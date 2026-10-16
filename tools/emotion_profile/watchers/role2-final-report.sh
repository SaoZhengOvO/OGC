#!/bin/bash
# 终局汇总：等最后一个臂跑完，把九个臂并排成一份可交付的对照（纯 CPU，不占卡）。
#
# 为什么要有它：每个臂各自出了对比表（SLICE-COMPARISON*、INTEGRATION-COMPARISON、
# FORMAT-FIX-COMPARISON、P03-OLD-VS-NEW），但**没有任何一处把九个臂放在一起**。
# 而"这一夜到底哪个改动有用、有用多少"恰恰需要那张总表。
#
# 它做三件事：
#   ① compare_arms：九个臂并排（情绪/画像三维 EMA/逐维五态分解），配对基准 = v3 隔离
#   ② format_health：九个臂的解析失败率 + 回复长度 + 残留污染（格式健康度）
#   ③ 写 runs/ALL-ARMS-COMPARISON.md —— 一次看全
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "终局汇总守护启动：等 runs/eval-p03new-slice200/score.json（链上最后一个臂）"

for _ in $(seq 1 600); do
    [ -f runs/eval-p03new-slice200/score.json ] && break
    sleep 30
done

if [ ! -f runs/eval-p03new-slice200/score.json ]; then
    say "终局汇总守护超时（最后一个臂没出分）"
    exit 1
fi

sleep 60
say "所有切片臂已跑完，开始出总表"
exec >> runs/.final-report.log 2>&1
date '+%m-%d %H:%M:%S'

# 等各臂的逐样本明细就位（每个臂的守护都会生成）
for tag in v3-slice200 v4-slice200 v5a-slice200 v6-slice200 v7-slice200 \
           integ-slice200 v8+p03-slice200 p03new-slice200; do
    for _ in $(seq 1 20); do
        [ -f "runs/eval-$tag/samples.jsonl" ] && break
        sleep 15
    done
done

# 下限那份（作为对照的第一行）
if [ ! -f runs/slice200/floor.samples.jsonl ]; then
    python3 src/cli.py eval \
        --predictions runs/slice200/preflight_floor.jsonl \
        --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
        --per-sample-out runs/slice200/floor.samples.jsonl
fi

ARMS=(
    "下限=runs/slice200/floor.samples.jsonl"
    "v3隔离=runs/eval-v3-slice200/samples.jsonl"
    "v4=runs/eval-v4-slice200/samples.jsonl"
    "v5a=runs/eval-v5a-slice200/samples.jsonl"
    "v6=runs/eval-v6-slice200/samples.jsonl"
    "v7=runs/eval-v7-slice200/samples.jsonl"
    "集成v3+③旧=runs/eval-integ-slice200/samples.jsonl"
    "集成v8+③旧=runs/eval-v8+p03-slice200/samples.jsonl"
    "集成v3+③新=runs/eval-p03new-slice200/samples.jsonl"
)

# 参数必须是**分开的两个 token**（`--arm` 和 `名字=路径`）。
# 注意不能用 "${ARMS[@]/#/--arm }" —— 那样每个元素会变成含空格的**单个**参数，
# argparse 会把它当成一个值而不是两个。
#
# 另外：**只把存在的臂加进去**。
# compare_arms 读不到文件会直接报错退出 —— 那会让"某一个臂失败"升级成
# "九臂总表全丢"。这里改成跳过缺失的臂、并把它们记进日志与总表开头。
ARM_ARGS=()
MISSING=()
for arm in "${ARMS[@]}"; do
    path="${arm#*=}"
    if [ -s "$path" ]; then
        ARM_ARGS+=(--arm "$arm")
    else
        MISSING+=("${arm%%=*}")
    fi
done

say "可对比的臂：$(( ${#ARM_ARGS[@]} / 2 )) 个；缺失：${MISSING[*]:-无}"

{
    echo "# 九臂并排总表"
    echo
    if [ "${#MISSING[@]}" -gt 0 ]; then
        echo "> ⚠️ **以下臂的逐样本明细缺失，已从对比中排除**：${MISSING[*]}"
        echo "> 缺失通常意味着该臂的守护跑失败或超时；请查 \`runs/.gpu-queue-watcher.log\`。"
        echo
    else
        echo "> 九个臂全部就位。"
        echo
    fi
} > runs/ALL-ARMS-COMPARISON.md

# ① 九臂并排（配对基准 v3 隔离）
python3 tools/emotion_profile/compare_arms.py \
    "${ARM_ARGS[@]}" \
    --paired "v3隔离" \
    --out runs/ALL-ARMS-COMPARISON.md.tmp
cat runs/ALL-ARMS-COMPARISON.md.tmp >> runs/ALL-ARMS-COMPARISON.md
rm -f runs/ALL-ARMS-COMPARISON.md.tmp

# ② 格式健康度（解析失败率 + 长度 + 残留污染）
{
    echo
    echo "---"
    echo
    echo "## 格式健康度（第 0 条判据）"
    echo
    echo '```'
    python3 tools/emotion_profile/format_health.py \
        --tag v3-slice200 --tag v4-slice200 --tag v5a-slice200 --tag v6-slice200 \
        --tag v7-slice200 --tag integ-slice200 --tag v8+p03-slice200 --tag p03new-slice200
    echo '```'
} >> runs/ALL-ARMS-COMPARISON.md

echo "[$(date '+%m-%d %H:%M:%S')] 九臂总表 -> runs/ALL-ARMS-COMPARISON.md"
