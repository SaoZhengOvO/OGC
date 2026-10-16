# 角色② 迭代 Runbook

> 一条实验从「想到一个改动」到「能下结论」的完整流程。
> 全部命令都在 `/mnt/workspace/OGC` 下执行（AMD MI300X 实例）。
> 口语化的逐轮记录在 `LOG.md`，正式报告在 `docs/reports/role2-emotion-profile-v1-v3.md`。

---

## 0. 三条必须先知道的口径

**① 不要为了让 smoke 好看去调提示词。**
n=20 上 95% 可分辨的最小差值是 **±23.6 个百分点**，而所有验收门都在这个阈值之下 ——
「在 smoke 上过门」很大程度上取决于抽到哪 20 条。
（推导见 `docs/reports/role2-emotion-profile-v1-v3.md` §13.4）

**② 决策用 200 条等距切片，结论用全量。**

| 口径 | 样本 | 95% 可分辨最小差值 | 单轮耗时 | 用途 |
|---|---|---|---|---|
| smoke | 20（val 前 20 条） | ±23.6 pp | 3.5 min | 只验「链路通不通」 |
| 切片 | 200（跨全 val 等距） | ±7.6 pp | 8 min | **要不要改 / 要不要推全量** |
| 全量 | 2354 | ±2.2 pp | 1.6 h | **最终结论** |

**③ 真正的门槛是「零成本下限」，不是裸模型。**
情绪：一律答 `anxiety` = **0.3200**；画像：三维一律留空 = **0.6283**。
裸模型（E1）是 0.2464 / 0.5615，**两个都低于下限**。

---

## 1. 改提示词

提示词的唯一可编辑源头是 `tools/emotion_profile/prompt_rules.txt`。
`#` 开头的行是注释，由 `make_prompt.py` 剥离，不会进产物（产物是原样拼进系统提示的，见
`deploy/participant/wiring.py:compose_system_prompt`）。

```bash
# 改完后重新生成产物（会覆盖 prompts/emotion_profile.md）
python3 tools/emotion_profile/make_prompt.py

# 校验产物与模板一致（提交前必跑）
python3 tools/emotion_profile/make_prompt.py --check

# 做候选版本：不要动已定稿的产物，渲染到别处
python3 tools/emotion_profile/make_prompt.py \
    --template tools/emotion_profile/prompt_rules.v5a.txt \
    --out runs/p02-v5a/emotion_profile.md
```

**铁律**
- 一次只改一处。改完用 `Compare-Object`（或 `diff`）确认只有预期的那几行不同。
- 词表不许手抄：它由 `tools/gen_label_block.py` 从 `configs/label_space.json` 生成。
- **不要写「整句 → 标签」的正面示例**（模型会照搬，v1/v2 都栽在这）；低填充率的维度给**反面示例**。

---

## 2. 控制变量：隔离其它线的提示词

`wiring.py` 会把 `PARTICIPANT_PROMPT_DIR` 下的 `emotion_profile.md` 与 `response.md` **都**拼进去。
角色③ 交付了 `response.md` 之后，不隔离就分不清是谁的功劳。

```bash
# 建一个只含自己那篇的目录
mkdir -p runs/p02-only && cp prompts/emotion_profile.md runs/p02-only/
```

产物哈希可以当控制变量的证据：

```bash
sha256sum runs/p02-only/emotion_profile.md runs/p02-v4/emotion_profile.md
```

---

## 3. 跑一轮

```bash
# --- smoke（只验链路）---
python3 tools/eval_loop.py --tag my-smoke --limit 20 \
    --inference-script wired --prompt-dir /mnt/workspace/OGC/runs/p02-only

# --- 200 条等距切片（决策用；总体无偏）---
python3 tools/emotion_profile/make_eval_slice.py --size 200          # 只需跑一次
python3 tools/emotion_profile/preflight_slice.py --slice runs/slice200  # 先预检，再占卡
python3 tools/eval_loop.py --tag my-slice200 \
    --source /mnt/workspace/OGC/runs/slice200/val_samples.jsonl \
    --references /mnt/workspace/OGC/runs/slice200/val_references.jsonl \
    --inference-script wired --prompt-dir /mnt/workspace/OGC/runs/p02-v4

# --- 分层诊断切片（只为看 per-class，总体 Acc 有偏、不能外推）---
python3 tools/emotion_profile/make_eval_slice.py --strategy stratified --per-class 15
python3 tools/eval_loop.py --tag my-strat15 \
    --source /mnt/workspace/OGC/runs/stratified15/val_samples.jsonl \
    --references /mnt/workspace/OGC/runs/stratified15/val_references.jsonl \
    --inference-script wired --prompt-dir /mnt/workspace/OGC/runs/p02-v4

# --- 全量（结论用）---
python3 tools/eval_loop.py --tag my-full \
    --inference-script wired --prompt-dir /mnt/workspace/OGC/runs/p02-v4
```

**两种切片别混用**：

| 切片 | 取样 | 总体 Acc | 用途 |
|---|---|---|---|
| `--size 200`（等距） | 跨全 val 每 11.77 条取一条 | **可代表全量**（逐维偏差 0.0024） | 决策：要不要改 / 要不要推全量 |
| `--strategy stratified --per-class 15`（230 条） | 每类情绪固定 15 条（小类全取） | **有偏，不能外推**（style 偏差 0.0826） | 诊断：小类到底能不能救 |

为什么要有分层那种：全量里 `surprise` 9 条、`care` 11 条，
200 条等距切片里各只有约 1 条，per-class 指标没有意义；
分层切片把 16 类都凑到 ≥9 条。

裸基线对照（官方原件，无接线层）：

```bash
python3 tools/eval_loop.py --tag baseline --inference-script official
```

---

## 4. 出证据

`eval_loop.py` 只写 `score.json`，下面这些要单独跑（都读它的 `submission.jsonl`）：

```bash
OUT=runs/eval-my-slice200
REF=/mnt/workspace/OGC/runs/slice200/val_references.jsonl   # 全量时换成 data/processed/val_references.jsonl

# 逐样本明细
python3 src/cli.py eval --predictions $OUT/submission.jsonl \
    --references $REF --per-sample-out $OUT/samples.jsonl

# 逐类 P/R/F1 + 画像五态分解 + 最亏的几类
python3 tools/emotion_profile/report.py --per-sample $OUT/samples.jsonl \
    --json-out $OUT/report.json

# 给人读的 badcase 清单（含对话原文）
python3 tools/emotion_profile/badcase_pack.py --per-sample $OUT/samples.jsonl \
    --out $OUT/badcase.md --per-class 8
```

---

## 5. 下结论（**必须过这一关**）

```bash
python3 tools/emotion_profile/compare_arms.py \
    --arm 下限=runs/slice200/floor.samples.jsonl \
    --arm v3=runs/eval-v3-slice200/samples.jsonl \
    --arm v4=runs/eval-v4-slice200/samples.jsonl \
    --paired v3 \
    --per-sample-diff \
    --out runs/SLICE-COMPARISON.md
```

`--paired` 给的是**同一批样本上的配对差值 + 95% 置信区间**：

```
差值 = (b - c) / n          b = 基准错/本臂对，c = 基准对/本臂错
SE   = sqrt(b + c - (b - c)² / n) / n
```

**区间跨过 0 就不许说「这一版更好」。** 切片上小于约 8 个百分点的差异一律不下结论。

---

## 6. 卡的排队规矩

见 `runs/GPU-RULES.md`。要点：**同时最多 2 个 GPU 任务**；
占卡前读 `runs/.gpu-lock.d/`，占卡写自己的 lock，跑完删自己的；
长任务一律 `setsid nohup`。

排队不要靠盯着等，写 watcher：

```bash
while pgrep -f "<前一个任务的特征串>" >/dev/null; do sleep 60; done
setsid nohup python3 tools/eval_loop.py --tag ... > runs/x.log 2>&1 < /dev/null &
```

本线已挂在共享盘上的守护（`/mnt/workspace/.team-ssh/`，不依赖任何人的笔记本）：

| 脚本 | 触发条件 | 干什么 |
|---|---|---|
| `role2-queue.sh` | `train_sft.py` 结束 | 起 v3 全量 |
| `role2-slice.sh` | v3 全量在跑 | 跑 v3-slice200、v4-slice200 |
| `role2-slice-report.sh` | v4-slice200 出分 | 出 `runs/SLICE-COMPARISON.md` |
| `role2-slice-v5a.sh` | 同上 | 跑 v5a-slice200，出 `runs/SLICE-COMPARISON-v5a.md` |
| `role2-postprocess.sh` | v3 全量跑完 | 出 `samples/report/badcase` |
| `role2-full-compare.sh` | v3 全量跑完 | 造零成本下限 → **配对显著性对比** → `runs/FULL-COMPARISON.md` |

> 最后一个才是"守住验收门"的最终证据：全量 2354 条的分辨力是 **±2.2 个百分点**，
> 比切片（±7.6pp）强得多。**只有过了 `--paired` 那一关的差值才写进报告。**

---

## 7. 提交前检查

```bash
python3 -m unittest discover -s tests        # 必须全绿（当前 114 条）
python3 tools/emotion_profile/make_prompt.py --check
git status --short                            # 只应有本线文件；不要 git add -A
```

只动 `prompts/emotion_profile.md` 与 `tools/emotion_profile/`；
`docs/reports/` 归 ⑤B（本线只在里面新建自己那篇报告，不改别人的文件）。

---

## 8. 当前实验链

| 臂 | 提示词产物 | sha256 前 8 位 | 相对上一臂的改动 | 结果 |
|---|---|---|---|---|
| official | （官方内置，无接线层） | — | 裸基线 | 全量 0.2464 / 0.5615（E1） |
| v1 | 已归档 `prompt_rules.v1.txt` | — | 首版规则（含正面示例） | n=20：0.4500 / 0.3167（**负收益**） |
| v2 | — | — | 删正面示例 | n=20：0.4500 / 0.2833 |
| v3 | `prompts/emotion_profile.md` | `6bcd6515` | 默认留空 + 负面示例 + 可判定措辞信号 | n=20：0.6000 / 0.5000（**统计上不显著**） |
| v4 | `prompt_rules.v4.txt` | `ce75f8cb` | 只把 style 从「默认留空」里摘出来 | 排队中 |
| v5a | `prompt_rules.v5a.txt` | `66fb9df2` | 只加情绪规则优先级（≤5 行） | 排队中 |
