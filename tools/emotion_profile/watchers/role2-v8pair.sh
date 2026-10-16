#!/bin/bash
# v8+③ 切片：在「和③一起挂」的条件下测 v8 的格式修复。
#
# 为什么必须这样测：10% 解析失败**只在两篇提示词同时挂时出现**
# （4 个单独运行 parse_failure 全为 0）。所以隔离跑测不出 v8 的收益。
#
# 对照：integ-slice200（v3+③）vs v8+p03-slice200（v8+③）
# 关键指标：parse_failure_count —— 它等于三项全丢，必须≈0。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "v8+③ 切片守护启动：等 runs/eval-integ-slice200/score.json"

for _ in $(seq 1 900); do
    [ -f runs/eval-integ-slice200/score.json ] && break
    sleep 30
done

if [ ! -f runs/eval-integ-slice200/score.json ]; then
    say "v8+③ 切片守护超时"
    exit 1
fi

sleep 30

# 组装 v8 + ③ 的提示词目录（只读复制，不动 prompts/）
PAIR=runs/p02-v8+p03
mkdir -p "$PAIR"
cp runs/p02-v8/emotion_profile.md "$PAIR/emotion_profile.md"
cp prompts/response.md "$PAIR/response.md"

# 组装后校验：**必须两个文件都在**。
# 否则（例如源文件缺失导致 cp 失败）目录里会只剩一篇提示词，
# 而这一臂会挂着 "v8+③" 的标签跑出"只有③"的结果 —— **静默错**，从分数上看不出来。
for f in emotion_profile.md response.md; do
    if [ ! -s "$PAIR/$f" ]; then
        say "❌ $PAIR/$f 缺失或为空 —— 拒绝用不完整的提示词跑（否则会产出标签错误的垃圾结果）"
        exit 1
    fi
done
say "v8+③ 提示词目录已就位：$(ls -1 "$PAIR" | tr '\n' ' ')"
say "v8+③ 提示词目录 $PAIR：$(ls -1 "$PAIR" | tr '\n' ' ')"

say "开始 v8+③ 切片（200 条）"
python3 tools/eval_loop.py \
    --tag v8+p03-slice200 \
    --source /mnt/workspace/OGC/runs/slice200/val_samples.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02-v8+p03 \
    >> runs/v8p03-slice200.log 2>&1
say "v8+③ 切片结束（退出码 $?）"

echo "=== 解析失败率对照（这是本轮最关键的指标）===" | tee -a "$LOG"
python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path

print(f"{'臂':<34}{'n':>5}{'解析失败':>9}{'失败率':>9}{'情绪':>8}{'画像':>8}")
print("-" * 76)
for tag, label in (("eval-integ-slice200", "集成 v3+③（修复前）"),
                   ("eval-v8+p03-slice200", "集成 v8+③（修复后）")):
    d = Path("runs") / tag
    p, s = d / "performance_report.json", d / "score.json"
    if not p.exists() or not s.exists():
        print(f"{label:<34}   （缺产物）")
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    m = json.loads(s.read_text(encoding="utf-8"))["metrics"]
    n = r.get("rounds") or 0
    pf = r.get("parse_failure_count") or 0
    print(f"{label:<34}{n:>5}{pf:>9}{(pf / n if n else 0):>9.2%}"
          f"{m.get('emotion_accuracy', 0):>8.4f}{m.get('profile_accuracy_mean_3dim', 0):>8.4f}")
    if pf:
        print(f"      失败 id: {r.get('parse_failure_ids')}")
print()
print("判读：解析失败从 X% 降到 ~0% 才说明格式修复有效；")
print("      若仍是同样比例，说明冲突不在这段格式说明，要另找来源。")
PY

python3 src/cli.py eval \
    --predictions runs/eval-v8+p03-slice200/submission.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --per-sample-out runs/eval-v8+p03-slice200/samples.jsonl \
    >> runs/.slice-report.log 2>&1

python3 tools/emotion_profile/compare_arms.py \
    --arm "集成v3+③=runs/eval-integ-slice200/samples.jsonl" \
    --arm "集成v8+③=runs/eval-v8+p03-slice200/samples.jsonl" \
    --paired "集成v3+③" \
    --out runs/FORMAT-FIX-COMPARISON.md \
    >> runs/.slice-report.log 2>&1

say "格式修复对照 -> runs/FORMAT-FIX-COMPARISON.md"
