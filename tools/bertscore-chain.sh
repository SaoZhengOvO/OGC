#!/usr/bin/env bash
# 等全量 2354 跑完，再用 GPU 补算 BERTScore。
#
# 为什么不跟全量抢 GPU：全量 2354 条是本次交付的最终分数来源，抢它的算力等于拿结论
# 去换一个代理指标。BERTScore 只依赖已有文本，晚一小时算没人损失。
#
# 为什么不用 CPU：这台机器 nproc=23、cgroup 配额也是 23 核，但实测 BERT-base
# batch16 单次前向要几十秒（10 次 1024³ matmul 十几分钟都跑不完），比正常慢两个数量级
# —— 宿主严重超卖。CPU 路线在 25 分钟超时内一条都算不完，不再尝试。
set -uo pipefail

cd /mnt/workspace/OGC || exit 1
export HF_HUB_DISABLE_XET=1
ENC=/mnt/workspace/models/bert-base-chinese
OUT=/mnt/workspace/bs-results
mkdir -p "$OUT"

echo "=== 等全量 eval_loop 退出（$(date -u +%H:%M:%S) UTC）==="
while pgrep -f "tools/eval_loop.py --tag final-full" >/dev/null 2>&1; do
  sleep 60
done
echo "=== 全量已退出（$(date -u +%H:%M:%S) UTC），GPU 空出来了 ==="

run_one() {
  local tag="$1" sub="$2" ref="$3"
  if [ ! -s "$sub" ]; then echo "跳过 $tag：没有 $sub"; return 0; fi
  if [ ! -s "$ref" ]; then echo "跳过 $tag：没有 $ref"; return 0; fi
  echo "--- $tag ---"
  python3 tools/bertscore_rescore.py \
    --submission "$sub" \
    --references "$ref" \
    --encoder "$ENC" \
    --device cuda \
    --batch-size 64 \
    --out "$OUT/$tag.json" 2>&1 | tail -6
}

# 四臂（各 300 条）
for arm in A B C D; do
  run_one "clean-$arm" "runs/eval-clean-$arm/submission.jsonl" "runs/eval-clean-$arm/val_references_subset.jsonl"
done

# 最终配置全量
run_one "final-full" "runs/eval-final-full/submission.jsonl" "data/processed/val_references.jsonl"

# 旧基线全量（用来跟 final-full 比「改了提示词之后 BERTScore 有没有掉」）
run_one "cap-v3-official-full" "runs/eval-cap-v3-official-full/submission.jsonl" "data/processed/val_references.jsonl"

echo "=== 全部完成（$(date -u +%H:%M:%S) UTC）==="
ls -la "$OUT"
