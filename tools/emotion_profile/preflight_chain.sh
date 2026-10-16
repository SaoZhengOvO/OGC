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
echo -n "  切片源行数（应为 200）: "
wc -l < runs/slice200/val_samples.jsonl
echo -n "  切片参考行数（应为 200）: "
wc -l < runs/slice200/val_references.jsonl
printf '  切片偏差: %s\n' "$(python3 -c "import json;print(round(max(json.load(open('runs/slice200/slice_manifest.json'))['abs_diff'].values()),6))")"
# 全量臂（queue / integ-full）**不传 --source**，走 eval_loop 的默认 data/processed/val_*.jsonl。
# 如果默认文件行数不是 2354，那两个"全量"跑出来的就不是全量 —— 而且不会报错。
echo -n "  全量默认源行数（应为 2354）: "
wc -l < data/processed/val_samples.jsonl
echo -n "  全量默认参考行数（应为 2354）: "
wc -l < data/processed/val_references.jsonl
echo "  提示词目录（每个应只含 1 个文件）:"
for d in runs/p02-only runs/p02-v4 runs/p02-v5a runs/p02-v6 runs/p02-v7 runs/p02-v8 runs/p03-new; do
    [ -d "$d" ] && printf '    %-16s %s 个文件\n' "$d" "$(ls -1 $d | wc -l)"
done

echo
echo "=== ⑤ 磁盘与卡 ==="
df -h /mnt/workspace | tail -1 | awk '{print "  NAS 可用: "$4" ("$5" 已用)"}'
echo "  卡上进程: $(ps -eo args | grep -c '[p]ython3 tools/train_sft') 个 train_sft"

echo
echo "=== ⑥ 自检（项数以运行输出为准）==="
python3 tools/emotion_profile/selftest.py 2>&1 | tail -2

echo
echo "=== ⑦ 旗标（集成全量是否被批准）==="
[ -f runs/.go-integrated-full ] && echo "  ⚠️ 已批准，队长 touch 过" || echo "  ⏸ 未批准（不占卡）"

echo
echo "=== ⑧ 每个守护等的文件现在是否已存在（存在 = 会立刻触发）==="
echo "  只检查**正在运行的**守护 —— 一次性诊断脚本（跑完就退）也引用产物路径，"
echo "  把它们算进来会出一堆假告警，反而淹没真正要看的东西。"
premature=0
checked=0
for f in /mnt/workspace/.team-ssh/role2-*.sh; do
    name=$(basename "$f")
    # 只保留正在跑的守护
    pgrep -f "[r]ole2-${name#role2-}" >/dev/null 2>&1 || continue
    target=$(grep -oE '\[\s*-f\s+[^] ]+\s*\]' "$f" 2>/dev/null | head -1 | sed 's/\[\s*-f\s*//; s/\s*\]//')
    case "$target" in
        ''|*'$'*|*'<路径>'*) continue ;;
    esac
    checked=$((checked + 1))
    if [ -f "$target" ]; then
        printf '  ⚠️ %-32s 等的 %s **已存在** → 会立刻触发\n' "$name" "$target"
        premature=$((premature + 1))
    fi
done
[ "$premature" -eq 0 ] && echo "  ✅ 正在运行的 $checked 个守护，没有一个的等待目标已存在"

echo
echo "=== ⑨ 触发判据会不会被「包装进程」误匹配（第 174 轮踩过）==="
echo "  背景：队长起 SFT 用的是 bash -c 包一层 nohup python3 tools/train_sft.py，"
echo "        那个包装进程的命令行里也含 train_sft.py，且训练结束后仍然存活。"
echo "        于是 pgrep -f train_sft.py 永远返回 ≥1 → 等待循环**永不退出** → 链不会启动。"
echo "  判据：一个 pgrep 模式若**同时匹配到命令行以 bash -c 开头的进程**，就有这个风险。"
echo "  （注意：只扫**非注释行** —— 注释里会引述这个坑，算进来就是假阳性。）"
risky=0
for f in /mnt/workspace/.team-ssh/role2-*.sh; do
    name=$(basename "$f")
    pgrep -f "[r]ole2-${name#role2-}" >/dev/null 2>&1 || continue
    # 先剥掉整行注释，再抽判据 —— 否则会把注释里的示例当成真判据
    grep -v '^[[:space:]]*#' "$f" 2>/dev/null \
      | grep -oE "pgrep -f ['\"][^'\"]+['\"]" \
      | sed "s/pgrep -f ['\"]//; s/['\"]$//" | sort -u > /tmp/.pf9_pats.txt
    while read -r pat; do
        [ -z "$pat" ] && continue
        for p in $(pgrep -f "$pat" 2>/dev/null); do
            cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
            case "$cmd" in
                *bash\ -c*)
                    printf '  ⚠️ %-24s 判据 "%s" 匹配到包装进程 PID %s\n' "$name" "$pat" "$p"
                    printf '       %s\n' "$(echo "$cmd" | cut -c1-90)"
                    ;;
            esac
        done
    done < /tmp/.pf9_pats.txt
done > /tmp/.preflight9.txt 2>/dev/null
rm -f /tmp/.pf9_pats.txt
if [ -s /tmp/.preflight9.txt ]; then
    cat /tmp/.preflight9.txt
    risky=$(wc -l < /tmp/.preflight9.txt)
else
    echo "  ✅ 正在运行的守护里，没有判据会匹配到 bash -c 包装进程"
fi
rm -f /tmp/.preflight9.txt

