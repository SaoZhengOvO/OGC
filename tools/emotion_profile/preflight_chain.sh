#!/bin/bash
# 全链起飞前检查：一条命令回答"能不能安心让它自己跑一夜"。
#
# 为什么要有它：这一夜修的两类守护 bug（重复触发条件、路径不一致）与三次
# "残留/前缀/近似"陷阱，全都是**不报错、只在凌晨静默出错**的类型。
# 所以在把长任务交给守护链之前，先跑这一遍。
#
# 用法：bash /mnt/workspace/.team-ssh/role2-preflight.sh
#       或  bash tools/emotion_profile/preflight_chain.sh
#
# 检查七项：
#   ① 每个守护是否恰好 1 个实例（多出来的要查清是包装进程还是重复守护）
#   ② watch_audit 复核并发性质与路径一致性
#   ③ 每个守护等的触发条件（按启动时间排列）
#   ④ 上游依赖完整性（切片行数/偏差、各提示词目录文件数）
#   ⑤ 磁盘与卡上进程
#   ⑥ selftest 16 项
#   ⑦ 集成全量旗标状态
cd /mnt/workspace/OGC || exit 1

echo "############ 全链起飞前检查 $(date '+%m-%d %H:%M:%S') ############"
echo

echo "=== ① 守护是否都活着（每个应恰好 1 个实例）==="
# 列表**从仓库的 watchers/ 自动取**，不再写死。
# 写死过一次，加了 final-report 之后这里还写着 14 个、循环里也没它 ——
# 于是"检查过了"给了虚假的安心。权威集合就是 tools/emotion_profile/watchers/*.sh。
if [ -d tools/emotion_profile/watchers ]; then
    WATCHERS=$(cd tools/emotion_profile/watchers && ls -1 role2-*.sh 2>/dev/null | sed 's/^role2-//; s/\.sh$//' | tr '\n' ' ')
    echo "  （守护列表取自 tools/emotion_profile/watchers/，共 $(echo $WATCHERS | wc -w) 个）"
else
    WATCHERS="queue slice slice-report slice-v5a slice-v6 slice-v7 integration v8pair p03new integ-full final-report postprocess full-compare publish-results"
    echo "  ⚠️ 仓库 watchers/ 不在，退回写死的列表（$(echo $WATCHERS | wc -w) 个）"
fi
for s in $WATCHERS; do
    n=$(ps -eo args | grep -c "[r]ole2-$s\.sh")
    [ "$n" -eq 1 ] && printf '  %-18s ✅\n' "$s" || printf '  %-18s ⚠️ 实例数 %s\n' "$s" "$n"
done

echo
echo "=== ② 串行链 & 并发性质（watch_audit 复核）==="
python3 tools/emotion_profile/watch_audit.py --dir /mnt/workspace/.team-ssh 2>&1 | grep -E "占卡守护|✅ 没有两个|✅ 没有两个守护共用|发现"

echo
echo "=== ③ 每个守护等的触发条件（按启动时间）==="
grep "守护启动\|排队器启动" runs/.gpu-queue-watcher.log | tail -14

echo
echo "=== ④ 上游依赖完整性 ==="
echo -n "  切片文件行数: "
wc -l < runs/slice200/val_samples.jsonl
printf '  切片偏差: %s\n' "$(python3 -c "import json;print(round(max(json.load(open('runs/slice200/slice_manifest.json'))['abs_diff'].values()),6))")"
echo "  六个提示词目录:"
for d in runs/p02-only runs/p02-v4 runs/p02-v5a runs/p02-v6 runs/p02-v7 runs/p02-v8 runs/p03-new; do
    [ -d "$d" ] && printf '    %-16s %s 个文件\n' "$d" "$(ls -1 $d | wc -l)"
done

echo
echo "=== ⑤ 磁盘与卡 ==="
df -h /mnt/workspace | tail -1 | awk '{print "  NAS 可用: "$4" ("$5" 已用)"}'
echo "  卡上进程: $(ps -eo args | grep -c '[p]ython3 tools/train_sft') 个 train_sft"

echo
echo "=== ⑥ 自检（当前 21 项）==="
python3 tools/emotion_profile/selftest.py 2>&1 | tail -2

echo
echo "=== ⑦ 旗标（集成全量是否被批准）==="
[ -f runs/.go-integrated-full ] && echo "  ⚠️ 已批准，队长 touch 过" || echo "  ⏸ 未批准（不占卡）"
