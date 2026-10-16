#!/bin/bash
# 链状态看板：每 5 分钟把"跑到哪一步了"写进 runs/CHAIN-STATUS.md。
#
# 为什么要有它：会话可能在产物落地前暂停（轮次/本地关机）。
# 有了这个看板，**任何人不用跑命令、不用登服务器** —— 只要看到那个文件，
# 就知道链走到哪一步、还差什么。
#
# 纯 CPU、不占卡、不碰任何别的产物。
cd /mnt/workspace/OGC || exit 1

OUT=runs/CHAIN-STATUS.md
for _ in $(seq 1 200); do   # 最多跑约 16 小时
    {
        echo "# 链状态看板"
        echo
        echo "> 由 \`role2-status-board.sh\` 每 5 分钟自动重写。最后更新：$(date '+%m-%d %H:%M:%S')"
        echo
        echo "## 一、总览"
        echo
        echo '| 项 | 状态 |'
        echo '|---|---|'
        if pgrep -f '^python3 tools/train_sft' >/dev/null 2>&1; then
            echo "| 队长 SFT | **仍在跑** —— $(tail -1 runs/sft-v1/train.log.jsonl 2>/dev/null | grep -oE '"step": [0-9]+' | head -1) |"
        else
            echo "| 队长 SFT | 已结束 |"
        fi
        # ⚠️ 判据必须**行首锚定** `^python3 tools/train_sft`，不能用裸的 `[t]rain_sft`：
        # 队长的 `bash -c` 包装进程（PID 7789）命令行里也含 train_sft，且训练结束后仍然存活
        # → 裸判据会让看板**永远显示"仍在跑"**，而看板正是会话中断后最主要的状态来源
        #   ⇒ 会误导恢复的人干等。（与 role2-queue.sh / role2-slice.sh 同一个坑，见 LOG §29）
        # 第 231 轮由起飞前检查第 ⑨ 项抓出。
        if pgrep -f '[e]val_loop.py' >/dev/null 2>&1; then
            echo "| 我的链 | **在跑** |"
        else
            echo "| 我的链 | 未在跑 |"
        fi
        echo "| 今晚产物数 | $(ls -d runs/eval-*slice200 runs/eval-v3-full/performance_report.json 2>/dev/null | wc -l) |"
        echo
        echo "## 二、逐个产物"
        echo
        echo '| 产物 | 状态 |'
        echo '|---|---|'
        for f in \
            "runs/eval-v3-slice200/score.json|v3 切片（链上第 1 臂）" \
            "runs/eval-v4-slice200/score.json|v4 切片" \
            "runs/eval-v5a-slice200/score.json|v5a 切片" \
            "runs/eval-v6-slice200/score.json|v6 切片（删错配词）" \
            "runs/eval-v7-slice200/score.json|v7 切片（补通路）" \
            "runs/eval-integ-slice200/score.json|集成 ②v3+③现行" \
            "runs/eval-v8+p03-slice200/score.json|集成 ②v8+③现行" \
            "runs/eval-p03new-slice200/score.json|集成 ②v3+③旧（A/B）" \
            "runs/eval-v3-full/performance_report.json|**v3 全量 2354**（目标要的数字）" \
            "runs/FULL-COMPARISON.md|全量配对显著性" \
            "runs/ALL-ARMS-COMPARISON.md|九臂总表" \
            "docs/reports/role2-full-val-results.md|**仓库内交付文件**"; do
            path="${f%%|*}"; label="${f##*|}"
            if [ -f "$path" ]; then
                printf '| %s | ✅ 已生成 |\n' "$label"
            else
                printf '| %s | ⏳ 等待中 |\n' "$label"
            fi
        done
        echo
        echo "## 三、产物在哪 / 怎么用"
        echo
        echo '```bash'
        echo 'bash /mnt/workspace/.team-ssh/role2-preflight.sh   # 9 项起飞前检查'
        echo 'python3 tools/emotion_profile/assess_verdict.py     # 按 §十九 判读表出机械裁决'
        echo 'python3 tools/emotion_profile/selftest.py           # 31 项不变量自检'
        echo '```'
        echo
        echo "> ⚠️ \`docs/reports/role2-full-val-results.md\` 由守护自动生成，"
        echo "> **但不会自动提交到 git** —— 需要人显式 \`git add\` 后提交。"
        echo "> 详见 \`tools/emotion_profile/RESUME.md\` 的「暂停指引」。"
    } > "$OUT.tmp" 2>/dev/null
    mv -f "$OUT.tmp" "$OUT" 2>/dev/null
    sleep 300
done
