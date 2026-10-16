# 角色② 进度快照 / 恢复指引

> 写于 2026-10-17 01:33（box 时间）。**用途**：万一实例被关机、或本地机器关掉导致会话中断，
> 照这份文件就能知道"从哪儿继续"，不用重新推一遍。

## 一分钟看懂现在的状态

| 项 | 状态 |
|---|---|
| **队长 SFT** | 跑到 **step 1660/2282**（约 5.18 秒/步）→ 预计 **02:27 结束** |
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
 然后依次：v5a → v6 → v7 → integration(②③**现行**) → v8pair(②v8+③现行) → p03new(②③**旧**)
 全量跑完 → postprocess（samples/report/badcase）+ full-compare（配对显著性）
 之后 → publish-results（写仓库内交付文件）+ final-report（九臂总表）
```

> ⚠️ **三个集成臂的输入必须是不同的文件**（第 122 轮踩过：两个臂用了同一份 prompt，
> 会跑出"没有差异"的假 A/B）。现在的实际对照：
>
> | 臂 | 提示词来源 | sha256 前 8 |
> |---|---|---|
> | `integ-slice200` | `prompts/response.md`（③现行·强调版 `\n\n`） | `639ce8c4` |
> | `v8+p03-slice200` | 同上（② 换成 v8） | `639ce8c4` |
> | `p03new-slice200` | `runs/p03-old/response.md`（③旧·简述版 `\n\n`） | `f3769150` |
>
> 自检里有一条 `A/B 两臂的输入确实不同` 专门盯这个 —— 它会在哈希相同时
> 直接报 **"这不是 A/B，是同一件事跑两遍"**。
>
> 另注：③ 在提交里说"定稿改为 v8"，但共享的 `prompts/response.md` 仍是 `639ce8c4`
> （上一版）。所以集成臂测的是 ③ 的**上一版**，不是 v8。

## 恢复后怎么接着看

```bash
# 1) 一眼看全链状态（8 项检查：守护/并发/依赖/磁盘/自检/旗标/等待目标）
bash /mnt/workspace/.team-ssh/role2-preflight.sh

# 2) 看产物落地了没有
ls runs/eval-v3-slice200/score.json runs/eval-v3-full/performance_report.json \
   runs/FULL-COMPARISON.md runs/ALL-ARMS-COMPARISON.md

# 3) 按 §十九 判读表出机械裁决（这才是"守住验收门"的出处）
python3 tools/emotion_profile/assess_verdict.py

# 4) 不变量自检（29 项）
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

## 收尾清单：数字到手后**必须**更新的位置

> 为什么单列：报告里现在有几处写着"尚未/正在跑"。
> **跑完之后不逐处改，报告就会留着过时的声明** ——
> 这正是第 82 轮修过的那类问题（判读表写"三项全丢"、§六 写"style 必须填"，
> 而它们都已被后来的数据推翻）。

**报告 `docs/reports/role2-emotion-profile-v1-v3.md` 的 4 处：**

| 位置 | 现在写着 | 改成 |
|---|---|---|
| L7（头部声明） | "全量 2354 条**尚未完成**，凡是没跑的都标「未跑」" | 完成声明 + 数字落点 |
| L32（§〇 限定框） | "全量 2354 条的数字才是能站住的结论，**正在跑**" | 实测数字 |
| L464（§十一 表格行） | "**未跑完**。原因：卡上同时有 SFT…" | 完成 + 排队结果 |
| L737（§14.3 标题） | "由此产生的两个可检验预测（写在这里，**等全量数字来打**）" | 填实测，逐条判"成立/被证伪" |

**另外两张要填实测的表**：§12（全量确认：预登记）、§19.4（A/B 两段，共 5 条预测）。

**还有两条预测散落在别处，容易漏（第 135 轮发现）：**

| 出处 | 预测 | 怎么判 |
|---|---|---|
| **§14.3 第 1 条的后半句** | v3 的画像分应当**明显超过 E1 的 0.5615**（因为 E1 在留空之外还多填错了一批） | 比 `0.5615`；只"打平下限 0.6283"不算违反，但**低于 0.5615 就是被证伪** |
| **§16.3 表格** | v3 的全量**情绪 Acc 大概率远低于 0.50**；**能否超过下限 0.3190 未知** | 实测情绪 vs 0.50 与 0.3190 —— 这条是"未知"型预测，**不设成败，只如实报告** |

> 这两条都不在 §19.4 的那张表里。**埋在别处的预测如果没登记，就等于没预登记** ——
> 跑完之后我会只去对 §19.4，而把这两条忘掉。所以在这里单列出来。

**新文件（必须显式 `git add`）**：`docs/reports/role2-full-val-results.md`
（由 `role2-publish-results.sh` 自动生成，仓库里目前**不存在**）。

**台账**：新增 §27 记录全量结果与逐条判读。

**先跑机械裁决、再手写叙述**：

```bash
python3 tools/emotion_profile/assess_verdict.py
```
