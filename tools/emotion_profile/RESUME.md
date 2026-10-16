# 角色② 进度快照 / 恢复指引

> 初稿写于 2026-10-17 01:33，**状态段更新于 06:30（终态）**。用途：万一实例被关机、
> 或本地机器关掉导致会话中断，照这份文件就能知道"从哪儿继续"，不用重新推一遍。

## 🔴 终态（2026-10-17 06:30）：实例已关机

```
最后成功心跳  06:04:03
06:12 / 06:20 / 06:28  连续三次 Unable to connect to port 30817
TCP 探测 120.220.172.117:30817 → 失败
```

**需要找队长重开实例**（`frp-rug.com:30817`）。这是当前唯一的前进阻碍。

### ✅ 但全量 2354 条的数字已交付并推送（提交 `fff9a0e4`，已确认在远端）

| 指标 | 实测 | 零成本下限 | 差 | E1 裸基线 | 差 | 判定 |
|---|---|---|---|---|---|---|
| **情绪 Acc** | **0.3849** | 0.3190 | **+0.0659** | 0.2464 | **+0.1385** | ✅ **守住** |
| 画像 Acc（三维均值） | **0.5586** | 0.6283 | −0.0697 | 0.5615 | −0.0029 | ❌ 未守住 |
| ├ personality_traits | 0.5314 | 0.6185 | −0.0871 | 0.6066 | −0.0752 | ❌ |
| ├ interests | 0.7111 | 0.7489 | −0.0378 | 0.6291 | **+0.0820** | ❌（超 E1） |
| └ style | 0.4333 | 0.5174 | −0.0841 | 0.4486 | −0.0153 | ❌ |
| 解析失败率 | **0.0000** | ≤ 2% | — | — | — | ✅ |

**结论：情绪线赢了，画像线输了。** 详见报告 §〇 与 `LOG.md` §32。

### ❌ 因实例关机而丢失（只在实例磁盘上，未及导出）

`v4 / v5a / v6 / v7` 与 3 个集成臂的切片分数、`runs/FULL-COMPARISON.md`、
`runs/ALL-ARMS-COMPARISON.md`、`docs/reports/role2-full-val-results.md`。

**重开后怎么补**：链会从"等 SFT"重新走一遍（此刻没有 SFT 在跑 → 会立刻起跑全链）。
若只想补切片臂，可把 `runs/eval-v3-full/` 的产物补回去，让 `postprocess` / `full-compare` 跳过全量。

### 链起跑的实证（第 174 轮那个修复**有效**）

```
02:27:40  SFT 结束（step 2282）
02:28:00  排队器「SFT 已结束」   ← while 循环真的退出了
02:29:15  已清理旧产物 → 已启动 v3-full，PID=45322 → lock 已写入
02:29:20  切片守护「并发闸门等待：run_inference = 2（>1，有第三方），等 60 秒再看」
03:46     全量 performance_report.json 落地 ✅
03:47     postprocess 完成；v3-slice200 起跑
06:04~06:12  实例关机
```

### 历史快照（02:10，供对照）

| 项 | 状态（02:10） |
|---|---|
| 队长 SFT | step 2080/2282 |
| 我的全量 | 未开始（等卡） |
| 守护 | 15 个 |
| 集成全量旗标 | 未 touch |

### 🚨 已修的关键 bug（第 174 轮）—— 若没修，链根本不会启动

`role2-queue.sh` / `role2-slice.sh` 原用 `pgrep -f "train_sft.py"` 等 SFT 结束。
但队长的 `bash -c` 包装进程（PID 7789）命令行里**也含** `train_sft.py`，
且训练结束后**仍然存活** → 循环永不退出 → **链不会启动**，而日志看起来一切正常。
已改成**行首锚定** `pgrep -f '^python3 tools/train_sft'`（实测：裸判据 2 个 / 锚定 1 个）。

> 起飞前检查第 **⑨** 项专门扫这类"判据被包装进程误匹配"的问题，
> 脚本侧的同名教训见 `LOG.md` §29 与 §30（后者是**我自己的工具**上重犯的）。

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
# 1) 一眼看全链状态（各项检查：守护/并发/依赖/磁盘/自检/旗标/等待目标）
bash /mnt/workspace/.team-ssh/role2-preflight.sh

# 2) 看产物落地了没有
ls runs/eval-v3-slice200/score.json runs/eval-v3-full/performance_report.json \
   runs/FULL-COMPARISON.md runs/ALL-ARMS-COMPARISON.md

# 3) 按 §十九 判读表出机械裁决（这才是"守住验收门"的出处）
python3 tools/emotion_profile/assess_verdict.py

# 4) 不变量自检（项数以运行输出为准）
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

### ⚠️ 「脱离会话的保活」试过，**没成功** —— 如实记录（第 213 轮更正）

第 210 轮我曾起了一个 `Start-Process` 的独立心跳，并在本文件写过"现在有一个脱离会话的
心跳"。**第 213 轮复查发现：那个进程只心跳了一次就消失了**（02:12:29 之后再无输出）。

三条路都试过，全部失败：

| 方式 | 结果 |
|---|---|
| `Start-Process -WindowStyle Hidden` | 心跳 1 次后**进程消失** ❌ |
| WMI `Invoke-CimMethod Win32_Process Create` | **拒绝访问**（0x80041003）❌ |
| `schtasks /create`（每 8 分钟） | **"系统找不到指定的路径"** ❌ |

**所以现在的实况是**：
* 心跳仍然只有**会话内那一个**（DSH 的后台任务），它整夜一直在正常跳（每 8 分钟一次）；
* **没有**任何独立于会话的保活机制。

### ✅ 因此：**本地机器与 DSH 会话都必须保持运行**（唯一且真实的保险）

这条没有替代方案。会话因轮次上限暂停**不一定**会让 DSH 后台任务退出
（后台任务不绑定单轮），但**我无法验证**，所以**最安全的做法是别关机器、别退 DSH**。

**恢复会话后确认心跳还在**：

```powershell
Get-Content _local\keepalive.log -Tail 3      # 应有近期（8 分钟内）的「心跳 OK」
```

若已停，重起一个：

```powershell
python _local/keepalive.py --interval 480     # 或带 --log 指定日志
```

**这是整晚唯一的单点故障** —— 链是服务器侧自治的、产物会自动落盘、看板会自动更新，
但如果没人给实例"续命"，它照样会在 1 小时后关机。

## 交付物在哪

| 内容 | 位置 | 进仓库吗 |
|---|---|---|
| 机制报告（87.7KB，含更正索引/导读/判读表/预登记） | `docs/reports/role2-emotion-profile-v1-v3.md` | ✅ |
| 台账（§1–§32） | `tools/emotion_profile/LOG.md` | ✅ |
| Runbook（守护表 14/14、工具表 19/19） | `tools/emotion_profile/RUNBOOK.md` | ✅ |
| **全量数字（自动生成）** | `docs/reports/role2-full-val-results.md` | ✅（跑完才有） |
| 逐样本明细 / badcase | `runs/eval-v3-full/` | ❌ gitignore |
| 九臂总表 | `runs/ALL-ARMS-COMPARISON.md` | ❌ gitignore |

## ⏸ 如果会话在"轮次上限"处暂停（第 155 轮写入）

本会话的自动续跑有**轮次预算**。SFT 结束前，box 时钟每轮只推进十几秒 ——
按当前节奏，**预算可能在产物落地之前用尽**，会话会暂停。**这不是失败，链仍在跑。**

### 暂停时要知道的事

1. **链是自治的**：14 个守护都是服务器侧 `setsid nohup`，**不依赖本地会话**
   （只依赖实例不被自动关机 → 所以**别关本地机器**，见上文"保活心跳"）。
2. **产物会照常落盘**在 `/mnt/workspace/OGC/runs/`，并自动生成
   `docs/reports/role2-full-val-results.md` —— **但不会自动提交到 git**，那一步要手动做。
3. **恢复会话后，先跑这三条**：

```bash
bash /mnt/workspace/.team-ssh/role2-preflight.sh        # 9 项：链还在不在、卡位对不对
python3 tools/emotion_profile/assess_verdict.py         # 机械裁决（没产物时退出码 2 + 明确提示）
```

**查"产物到哪一步"要用下面这条**（**不要用 `runs/eval-*/score.json` 通配** ——
那会把早就存在的 `eval-*-smoke/score.json` 也列出来，让人误以为有产物）：

```bash
for f in runs/eval-v3-slice200/score.json runs/eval-v4-slice200/score.json \
         runs/eval-v5a-slice200/score.json runs/eval-v6-slice200/score.json \
         runs/eval-v7-slice200/score.json runs/eval-integ-slice200/score.json \
         runs/eval-v8+p03-slice200/score.json runs/eval-p03new-slice200/score.json \
         runs/eval-v3-full/performance_report.json \
         runs/FULL-COMPARISON.md runs/ALL-ARMS-COMPARISON.md \
         docs/reports/role2-full-val-results.md; do
    [ -f "$f" ] && echo "✅ $f" || echo "⏳ $f"
done
```

> 或者**直接看 `runs/CHAIN-STATUS.md`** —— 那份看板每 5 分钟自动重写，
> 12 个产物逐项标 ✅/⏳，**不用跑命令**。

> ⚠️ 上面这几条是**第 233 轮演练恢复流程时改的**：
> 演练发现原命令的 `eval-*` 通配会把 5 个旧的 smoke 产物列出来（看起来像"有产物"），
> 而且顺手发现同一处的"8 项"已是 9 项。**演练恢复流程本身就有价值。**

4. **然后照"收尾清单"逐处更新报告与台账**，
   并把 `docs/reports/role2-full-val-results.md` **显式 `git add` 后提交**。

### 暂停期间不会发生的事（避免误判）

* **不会有人替你提交产物** —— git 提交必须手动做；
* **不会有人替你解读数字** —— `assess_verdict.py` 给机械判据，但**叙述要我写**；
* 队列不会乱 —— 守护之间严格串行 + 并发闸门。

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
