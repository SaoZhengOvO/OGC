# 角色② 进度快照 / 恢复指引

> 写于 2026-10-17 01:29（box 时间）。**用途**：万一实例被关机、或本地机器关掉导致会话中断，
> 照这份文件就能知道"从哪儿继续"，不用重新推一遍。

## 一分钟看懂现在的状态

| 项 | 状态 |
|---|---|
| **队长 SFT** | 跑到 **step 1600/2282**（约 5.17 秒/步）→ 预计 **02:27 结束** |
| **我的全量** | **还没开始** —— 排队器 `role2-queue.sh` 在等 SFT 结束（不抢卡） |
| **我的切片链** | 还没开始 —— `role2-slice.sh` 在等 SFT 结束 |
| **今晚的评测产物** | **0 个**（`runs/eval-v3-slice200/`、`runs/eval-v3-full/performance_report.json` 都不存在） |
| **守护** | **14 个**在共享盘 `/mnt/workspace/.team-ssh/` 上待命（`setsid nohup`，断开 SSH 不受影响） |
| **GPU 锁** | `captain-sft.lock`（队长的）、`role2-queued.lock`（我的排队标记） |
| **集成全量旗标** | **未 touch** → 那一臂不会跑（`runs/.go-integrated-full`） |

## 卡一空会自动发生什么（不需要人盯）

```
SFT 结束
 ├─ queue  清除旧产物 → 起 v3 全量（2354 条，②隔离）      ← 目标要的那个数字
 └─ slice  过并发闸门 → 起 v3/v4 切片（各 200 条）
 然后依次：v5a → v6 → v7 → integration(②③旧) → v8pair → p03new(②③新)
 全量跑完 → postprocess（samples/report/badcase）+ full-compare（配对显著性）
 之后 → publish-results（写仓库内交付文件）+ final-report（九臂总表）
```

## 恢复后怎么接着看

```bash
# 1) 一眼看全链状态（8 项检查：守护/并发/依赖/磁盘/自检/旗标/等待目标）
bash /mnt/workspace/.team-ssh/role2-preflight.sh

# 2) 看产物落地了没有
ls runs/eval-v3-slice200/score.json runs/eval-v3-full/performance_report.json \
   runs/FULL-COMPARISON.md runs/ALL-ARMS-COMPARISON.md

# 3) 按 §十九 判读表出机械裁决（这才是"守住验收门"的出处）
python3 tools/emotion_profile/assess_verdict.py

# 4) 不变量自检（28 项）
python3 tools/emotion_profile/selftest.py

# 5) 守护的日志（谁在等什么、跑完没）
tail -30 runs/.gpu-queue-watcher.log
```

## 如果守护全没了（实例被关机后重启）

守护脚本都在共享盘上，重起即可（**注意：不要用 `pkill -f`，会匹配到自己的 SSH 命令行**）：

```bash
cd /mnt/workspace/OGC
for s in queue slice slice-report slice-v5a slice-v6 slice-v7 integration \
         v8pair p03new integ-full final-report postprocess full-compare publish-results; do
    setsid nohup bash /mnt/workspace/.team-ssh/role2-$s.sh \
        > runs/.gpu-$s.out 2>&1 < /dev/null &
done
bash /mnt/workspace/.team-ssh/role2-preflight.sh   # 复核
```

> 实例**被关机后进程全没**，但**文件都在 NAS 上**（`/mnt/workspace` 是网络盘）。
> 所以已经跑完的产物不会丢，只是没跑完的要重跑。

## 本地保活心跳（关系到实例会不会被自动关机）

`TEAM-TASKS.md`：**1 小时无操作会关机**；`COMPUTE.md`：**闲置即关**。
心跳跑在**本地机器**上：

```bash
python _local/keepalive.py --interval 480     # 每 8 分钟一次极轻 SSH 探测
```

**本地一关，心跳就停** —— 若平台的"闲置"判据是看连接活动，实例可能在 1 小时内被关机，
今晚所有进程一起没。文档措辞偏向"看连接"，但我无法安全验证（验证失败的代价就是丢数据）。

## 交付物在哪

| 内容 | 位置 | 进仓库吗 |
|---|---|---|
| 机制报告（87.7KB，含更正索引/导读/判读表/预登记） | `docs/reports/role2-emotion-profile-v1-v3.md` | ✅ |
| 台账（§1–§25） | `tools/emotion_profile/LOG.md` | ✅ |
| Runbook（守护表 14/14、工具表 19/19） | `tools/emotion_profile/RUNBOOK.md` | ✅ |
| **全量数字（自动生成）** | `docs/reports/role2-full-val-results.md` | ✅（跑完才有） |
| 逐样本明细 / badcase | `runs/eval-v3-full/` | ❌ gitignore |
| 九臂总表 | `runs/ALL-ARMS-COMPARISON.md` | ❌ gitignore |

## 待队长拍板（三个 issue 都还没回复）

| issue | 问题 | 现状 |
|---|---|---|
| #3 | 是否要跑**集成全量**（把系统级数字从 n=200 提到 2354） | 门控就绪，`touch runs/.go-integrated-full` 即开跑 |
| #2 | 是否降级 ② 的提示词投入 | 未回复 |
| #1 | smoke-vs-slice 协议 | 未回复 |
