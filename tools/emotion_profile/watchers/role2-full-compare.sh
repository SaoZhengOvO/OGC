#!/bin/bash
# 全量跑完后，补出「v3 全量 vs 零成本下限」的配对显著性对比（纯 CPU，不占卡）。
#
# 为什么需要：后处理守护只出描述性统计（report.py）。而"守住验收门"这个结论要有显著性支撑
# —— 全量 2354 条的分辨力是 ±2.2 个百分点，比切片（±7.6pp）强得多，
# 所以这一份才是能写进报告的最终证据。
#
# 下限的构造：情绪一律 anxiety、画像三维一律留空、回复一句固定话。
# 输入直接用 eval_loop 已经写好的 runs/eval-v3-full/val_input.jsonl。
cd /mnt/workspace/OGC || exit 1

LOG=runs/.gpu-queue-watcher.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "全量对比守护启动：等 runs/eval-v3-full/performance_report.json"

for _ in $(seq 1 720); do
    [ -f runs/eval-v3-full/performance_report.json ] && break
    sleep 60
done

if [ ! -f runs/eval-v3-full/performance_report.json ]; then
    say "全量对比守护超时"
    exit 1
fi

# 等后处理把 samples.jsonl 出出来
for _ in $(seq 1 60); do
    [ -f runs/eval-v3-full/samples.jsonl ] && break
    sleep 30
done

say "全量已跑完，开始出「vs 下限」的配对对比"
exec >> runs/.full-compare.log 2>&1
date '+%m-%d %H:%M:%S'

FULL=runs/eval-v3-full
REF=data/processed/val_references.jsonl

# 1) 造下限预测（与 preflight_slice.py 同一口径）
python3 - "$FULL" <<'PY'
import json, pathlib, sys
full = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in (full / "val_input.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
out = full / "floor.jsonl"
with out.open("w", encoding="utf-8", newline="\n") as handle:
    for row in rows:
        handle.write(json.dumps({
            "id": row["id"],
            "response_text": "嗯，我在听。",
            "emotion_label": "anxiety",
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
            "memory_refs": [],
        }, ensure_ascii=False) + "\n")
print("下限预测", len(rows), "条 ->", out)
PY

# 2) 逐样本明细
python3 src/cli.py eval --predictions "$FULL/floor.jsonl" --references "$REF" \
    --per-sample-out "$FULL/floor.samples.jsonl"

# 3) 配对对比（基准 = 下限），这一步才是结论
python3 tools/emotion_profile/compare_arms.py \
    --arm "下限=runs/eval-v3-full/floor.samples.jsonl" \
    --arm "v3=runs/eval-v3-full/samples.jsonl" \
    --paired "下限" \
    --out runs/FULL-COMPARISON.md

# 4) 逐类诊断（含"最亏的几类"）——写不同的文件名，避免与后处理守护抢 report.json
python3 tools/emotion_profile/report.py --per-sample "$FULL/samples.jsonl" \
    --json-out "$FULL/report-vs-floor.json" --top-bad 30 > "$FULL/report.txt"

echo "[$(date '+%m-%d %H:%M:%S')] 全量对比 -> runs/FULL-COMPARISON.md"
