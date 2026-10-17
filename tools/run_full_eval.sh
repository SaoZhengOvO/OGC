#!/usr/bin/env bash
# 可断点续跑的全量评测。
#
# 为什么需要这个脚本：2026-10-17 全量 2354 条跑到 794 条时，AMD 盒子被关停，
# 而 eval_loop.py 产出 submission.jsonl 是逐条 append 的、没有断点续跑能力 ——
# 重来就是一次 138 分钟。这里改成按 offset 分块跑，每块跑完立刻落盘成独立文件，
# 断一次最多损失 CHUNK 条；重跑同一条命令时已完成的块直接跳过。
#
# 块之间的 references 是按 id 对齐的（eval_loop.py 的 write_reference_subset 用 id 匹配，
# 不是按行号切片），所以分块分别打分再合并，与一次性跑全量在口径上等价。
#
# 用法（在盒子或任何有 GPU 的机器上，仓库根目录）：
#   TAG=final-full CHUNK=300 bash tools/run_full_eval.sh
# 断了就再跑同一条命令。全部块完成后自动合并 -> --skip-inference 打分。
#
# 环境变量：
#   TAG      实验名（默认 final-full），产物落 runs/eval-<TAG>/
#   CHUNK    每块条数（默认 300）
#   TOTAL    总条数（默认 2354）
#   PROMPT_DIR 提示词目录（默认不传，用 eval_loop.py 自带的 prompts/）
#   HARDWARE_LABEL 写进 performance_report.json 的硬件标签
set -uo pipefail

TAG="${TAG:-final-full}"
CHUNK="${CHUNK:-300}"
TOTAL="${TOTAL:-2354}"
PY="${PY:-python3}"
SOURCE="${SOURCE:-data/processed/val_samples.jsonl}"
REFS="${REFS:-data/processed/val_references.jsonl}"
PROMPT_DIR="${PROMPT_DIR:-}"
HARDWARE_LABEL="${HARDWARE_LABEL:-mi300x-official-full}"

cd "$(dirname "$0")/.." || exit 1
PARTS="runs/$TAG-parts"
mkdir -p "$PARTS"

common=(--source "$SOURCE" --references "$REFS" --hardware-label "$HARDWARE_LABEL")
if [ -n "$PROMPT_DIR" ]; then common+=(--prompt-dir "$PROMPT_DIR"); fi

failed=0
for ((off = 0; off < TOTAL; off += CHUNK)); do
  want=$((TOTAL - off)); ((want > CHUNK)) && want=$CHUNK
  part="$PARTS/$(printf '%05d' "$off").jsonl"
  have=0
  if [ -s "$part" ]; then have=$(wc -l <"$part"); fi
  if ((have >= want)); then
    echo "[跳过] offset=$off 已完成（$have/$want）"
    continue
  fi
  echo "=== offset=$off limit=$want  $(date '+%H:%M:%S') ==="
  "$PY" tools/eval_loop.py --tag "$TAG-c$off" --offset "$off" --limit "$want" "${common[@]}"
  rc=$?
  if ((rc != 0)); then
    echo "[失败] offset=$off 退出码 $rc" >&2
    failed=1
    break
  fi
  cp "runs/eval-$TAG-c$off/submission.jsonl" "$part" || {
    echo "[失败] 复制不了 runs/eval-$TAG-c$off/submission.jsonl" >&2
    failed=1
    break
  }
done

if ((failed)); then
  echo "有块失败。修好后重跑同一条命令即可从上一次断点续跑。" >&2
  exit 1
fi

echo "=== 合并（$(date '+%H:%M:%S')）==="
merged="runs/$TAG/submission.jsonl"
mkdir -p "runs/$TAG"
cat "$PARTS"/*.jsonl >"$merged"
echo "合并后 $(wc -l <"$merged") 行 -> $merged"

echo "=== 打分（$(date '+%H:%M:%S')）==="
"$PY" tools/eval_loop.py --tag "$TAG" --skip-inference --predictions "$merged"
