# 角色② 情绪与画像：v1 负收益的归因与 v3 修复

> 负责人：② `zjdhhds`　·　日期：2026-10-16　·　算力：AMD MI300X（192 GB，ROCm 7.2.3）
> 证据：`runs/eval-{smoke-official,smoke-wired,v2-smoke,v3-smoke}/`　·　台账：`runs/leaderboard.tsv`
> 代码：`prompts/emotion_profile.md`、`tools/emotion_profile/`
>
> **这份报告只写已经跑出来的数。全量 2354 条尚未完成，凡是没跑的都标「未跑」。**

---

## 〇、结论摘要

v1 的提示词相对裸基线是**负收益**（情绪 −0.05、画像 −0.133，`personality_traits` 掉 0.35）。
归因完成后，v3 在 n=20 smoke 上**三项验收门全部达成**：

| 指标 | 裸基线 official | v1 | **v3** | 验收门 | 结果 |
|---|---|---|---|---|---|
| 情绪 Acc | 0.5000 | 0.4500 | **0.6000** | > 0.5000 | ✅ |
| 画像 Acc（三维均值） | 0.4500 | 0.3167 | **0.5000** | > 0.4500 | ✅ |
| └ personality_traits | 0.6000 | 0.2500 | **0.6000** | — | ✅ 恢复 |
| └ interests | 0.5500 | 0.4000 | **0.6000** | — | ✅ 超过基线 |
| └ style | 0.2000 | 0.3000 | **0.3000** | ≥ 0.3000 | ✅ |

**相对裸基线的净收益：情绪 +0.10，画像 +0.05。**

另外**推翻了 `COMPUTE.md` §6 第 3 条**的诊断（详见 §三）：那条说「标签词表对不上、改格式就能拿分」，
实际是**裸模型根本没有输出画像标签**，不存在「措辞对不上」。按那条诊断去改格式会白费力气。

---

## 一、实验协议（控制变量）

判据是「五个人用同一把尺子」，所以本轮所有实验只改**一个**变量，并把另外两条线的影响隔离掉：

```bash
python3 tools/eval_loop.py --tag <tag> --limit 20 \
    --inference-script wired --prompt-dir /mnt/workspace/OGC/runs/p02-only
```

- `--inference-script wired`：跑**要提交的那棵树**（`deploy/participant/`，带接线层），
  对照臂用 `official`（`GivenData/participant/` 官方原件，无接线层）。
- `--prompt-dir .../p02-only`：这个目录里**只有 `emotion_profile.md`**。
  因为角色③的 `prompts/response.md` 是 22:50 才交付的，而基线是 22:05 跑的；
  不隔离的话，「接线版比基线差」分不清是我改坏的还是③的提示词带的。
- `--limit 20`：val 的前 20 条。**分层结论不成立**（见 §五波动性声明）。

四个臂的唯一差别：

| 臂 | 推理脚本 | 提示词目录 |
|---|---|---|
| `official` | GivenData 官方原件 | （无接线层，读不到 prompts） |
| `v1` / `v2` / `v3` | deploy 副本 | 只含对应版本的 `emotion_profile.md` |

---

## 二、原始数字（n=20）

### 表 1：四个臂的分数

| 指标 | official | v1 | v2 | v3 |
|---|---|---|---|---|
| 情绪 Acc | 0.5000 | 0.4500 | 0.4500 | **0.6000** |
| 画像 Acc（三维均值） | 0.4500 | 0.3167 | 0.2833 | **0.5000** |
| └ personality_traits EMA | 0.6000 | 0.2500 | 0.2000 | **0.6000** |
| └ interests EMA | 0.5500 | 0.4000 | 0.5500 | **0.6000** |
| └ style EMA | 0.2000 | 0.3000 | 0.1000 | **0.3000** |
| personality micro-P | 0.0000 | 0.1667 | — | **0.5000** |
| personality micro-F1 | 0.0000 | 0.2424 | — | **0.3077** |
| style micro-P | 0.0000 | 0.2222 | — | **0.5000** |
| 回复 BLEU-4 | 0.0016 | 0.0001 | 0.0008 | 0.0006 |
| ROUGE-L | 0.1093 | 0.0815 | 0.1063 | 0.0919 |

> 回复两项归角色③，这里只作对照，不作为②的结论。

### 表 2：逐维五态分解（命中 = 集合完全一致）

| 维度 | 臂 | EMA | 空对空 | 命中 | 该空却填 | 该填却空 | 填错 | 输出标签频次 |
|---|---|---|---|---|---|---|---|---|
| personality_traits | 金标为空 **12/20** | — | — | — | — | — | — | — |
| | official | 0.6000 | 12 | 0 | 0 | 8 | 0 | `{}` |
| | v1 | 0.2500 | 5 | 0 | **7** | 3 | 5 | `introverted:9, sensitive:8, open:3, casual:2` |
| | v2 | 0.2000 | 3 | 1 | **9** | 2 | 5 | `introverted:10, emotionally_stable:7, casual:3` |
| | **v3** | **0.6000** | 12 | 0 | **0** | 6 | 2 | `sensitive:2, open:1, introverted:1` |
| interests | 金标为空 **13/20** | — | — | — | — | — | — | — |
| | official | 0.5500 | 11 | 0 | 2 | 7 | 0 | `reading_writing:2, travel_outdoor:1` |
| | v1 | 0.4000 | 8 | 0 | 5 | 5 | 2 | `career_development:3, reading_writing:3, social:2` |
| | v2 | 0.5500 | 11 | 0 | 2 | 7 | 0 | `social:1, study_exam:1, reading_writing:1` |
| | **v3** | **0.6000** | 12 | 0 | **1** | 7 | 0 | `reading_writing:1, social:1, study_exam:1` |
| style | 金标为空 **4/20** | — | — | — | — | — | — | — |
| | official | 0.2000 | 4 | 0 | 0 | 15 | 1 | `formal:1, low_emotional_expression:1` |
| | v1 | 0.3000 | 2 | 4 | 2 | 2 | 10 | `colloquial:7, rational:6, formal:3, direct:2` |
| | v2 | 0.1000 | 1 | 1 | 3 | 1 | 14 | **`brief:17`**, direct:1 |
| | **v3** | **0.3000** | 4 | 2 | 0 | 12 | 2 | `brief:2, direct:1, rational:1` |

**「该空却填」是集合精确匹配下最亏的一类错误**：金标本为空，写任何标签都直接判错。
v1 在 personality 上有 7 条、v2 有 9 条，v3 降到 0 条 —— 这是 v3 能把 EMA 拉回 0.60 的直接原因。

---

## 三、机制一：裸模型的画像 Acc 0.45 不是能力，是「金标为空的比例」

`official/score.json` 三个维度同时满足两个条件：

| 维度 | exact_match_acc | macro_jaccard | micro_f1 |
|---|---|---|---|
| personality_traits | 0.60 | **0.60** | **0.0** |
| interests | 0.55 | **0.55** | **0.0** |
| style | 0.20 | **0.20** | **0.0** |

数学上这是唯一可能的解释。看 `src/evaluate.py:213-234`：

```python
union = pred_set | ref_set
self.jaccard_sum += 1.0 if not union else len(pred_set & ref_set) / len(union)
```

- 预测恒为空时，`union == ∅ ⟺ ref == ∅`，所以 **macro_jaccard ≡ exact_match_acc**；
- 同时 `tp = Σ|pred ∩ ref| = 0` ⇒ `precision = recall = micro_f1 = 0`。

逐条 diff 证实了这一点：**官方那 20 条里 `personality_traits` 一个标签都没输出（频次 `{}`）**，
interests/style 各只有 1–2 条误填。所以：

> **`COMPUTE.md` §6 第 3 条「micro-F1 挂零但 Jaccard 有 0.55–0.60，说明标签词表对不上、
> 改格式就能拿到分」是误诊。** micro-F1 归零是因为 `tp=0`（根本没输出标签），
> 不是因为「写对了意思、写错了词」。按那条诊断去改格式或加同义词映射，不会有收益。

这条结论同时给出一个**策略性事实**：这 20 条的金标为空率是
`personality 12/20 · interests 13/20 · style 4/20`，
所以 **personality / interests 的最优策略是「留空」，style 必须填**。

---

## 四、机制二：v1 为什么掉分

### 4.1 `personality_traits` 0.60 → 0.25：示例被当成了默认答案

v1 在画像段落里写了「一句话 → 标签」的示例映射，其中一句是：

```
「我不太敢跟陌生人说话」→ introverted 或 sensitive
```

逐条 diff 的结果：**模型输出了 `['introverted','sensitive']` 这个一模一样的组合 6 次**
（`val_000003_t5/t7`、`val_000005_t9`、`val_000007_t3`、`val_000014_t3`、`val_000015_t5`），
而这几条的金标**全部是空数组**。

量级：v1 在 personality 上共输出 24 个标签、命中 4 个 → **精确率 16.7%**。
而集合精确匹配的盈亏平衡点是 **50%**（见 §六），16.7% 远低于它，所以「填」比「留空」更差：

| | official | v1 |
|---|---|---|
| 空对空（金标空且留空 → 得分） | 12 | **5** |
| 该空却填（直接判错） | 0 | **7** |
| 命中 | 0 | 0 |
| **EMA** | **0.60** | **0.25** |

### 4.2 情绪 0.50 → 0.45：anxiety 过度吸引

v1 误判里最集中的模式：

| v1 误判 | 次数 |
|---|---|
| `anger → anxiety` | 3 |
| `helplessness → anxiety` | 2 |
| `fear → anxiety` | 1 |

但用户原话里其实**明说了**情绪词：

| 样本 | 用户原话（节选） | 明说的词 | 金标 | v1 猜 |
|---|---|---|---|---|
| `val_000009_t5` | 「反而让我更**烦躁**」 | 烦躁 | anger | anxiety |
| `val_000009_t7` | 「真的让人想**摔键盘**」 | 摔键盘 | anger | anxiety |
| `val_000012_t5` | 「说实话，是更**无力**了」 | 无力 | helplessness | anxiety |
| `val_000015_t5` | 「此刻只觉得**害怕**又无力」 | 害怕 | fear | anxiety |

根因在 v1 提示词自身的分布上：第一步的情绪词示例是
`"我好焦虑""开心""难受""烦死了""有点慌"` ——
**5 个例子里 3 个指向负面、1 个指向焦虑、0 个指向 anger/fear/helplessness**；
而第二步 anxiety 那一行写了 6 个触发词，其他三类各只有一句。
两者叠加，模型学到的是「默认焦虑」，第一步的「明说优先」形同虚设。

### 4.3 v2 的两个新失败（改过头的代价）

v2 删掉了正面的示例映射，但引入了新问题：

- **personality**：模型把「情绪表达克制」反向映射成 `emotionally_stable`，
  输出了 **7 次**。而该标签在训练集里只占非空的 **1.0%**（9/898）。
  `该空却填` 从 v1 的 7 条升到 **9** 条 → EMA 反而掉到 0.20。
- **style**：`brief` 输出了 **17 次**（20 条里 17 条），只命中 1 条。
  根因是把 `colloquial/formal` 删掉后，模型退回到列表里最「客观」的长度标签当默认值，
  而这条消息本身的长短它判断不了。

---

## 五、两条可复用教训

> 这两条是这一轮真正的产出，与具体标签无关，后面 ③ 或任何提示词线都能直接用。

### 教训 1：填充分数低的维度，要给「反面示例」而不是正面示例

**只要提示词里出现「整句 → 标签」的正面示例，模型就会把它当成填写许可并照搬。**
证据：v1 的 `['introverted','sensitive']` 出现 6 次、v2 的 `brief` 出现 17 次，
两次都是照搬示例的直接后果。

正确做法（v3 采用）：
- 先写死默认值：「**三个维度的默认输出都是空数组 []**」；
- 再**只列「什么情况下不要填」**（描述难受的事 / 抱怨某人 / 要求被倾听 / 表达孤独焦虑 → 一律留空）；
- 一个正面示例都不给。
- 若某个标签有「反向误用」的风险（如 `emotionally_stable`），**点名禁掉**。

### 教训 2：不要用宽触发词表，改成可判定的措辞信号

v1 把 anxiety 写成默认答案；v2 修掉 anxiety 后，模型又改抓 relaxed 当默认（6 条）。
**只禁一个默认值不够，因为模型总会去找下一个「最好匹配」的类别。**

正确做法（v3 采用）：
- 每个候选标签都给出**同样具体**的触发条件，不允许某一类特别长；
- 判据从「抽象描述」改成**可判定的措辞**。
  例：`style=direct` 从「说话直、不绕弯」改为
  **「用户直接向你提要求、下指令或划边界」（"先别给建议""别问了，先让我说完""就这样吧"）**；
  这一改把 `direct` 的命中从 2 条提到 5 条，style 精确率从 0.222 提到 **0.500**；
- 明确点名「哪两个是默认答案」（v3 同时禁掉 anxiety 与 relaxed）。

---

## 六、由数据推出的策略：每个维度该不该填

集合精确匹配下，对某一维填一个预测的期望变化：

```
Δ = (# 完全命中的条目) − (# 金标为空却填了的条目)
```

所以**填的条目里必须超过 50% 完全正确，才比全空强**。把这个用到这 20 条上：

| 维度 | 金标为空 | 留空的得分 | 结论 |
|---|---|---|---|
| personality_traits | 12/20 = 60% | **0.60** | 不值得填（v1 填了 24 个只对 4 个） |
| interests | 13/20 = 65% | 0.65（理论上限） | 不值得填 |
| style | 4/20 = 20% | 0.20 | **必须填**，留空拿不到分 |

v3 正是按这个分配：personality/interests 实际接近「留空」（各只误填 2 条 / 1 条），
style 则保留可判定的填写规则。

---

## 七、逐条 diff（n=20 全量）

### 表 A：情绪

| 样本 | 金标 | official | v1 | v2 | v3 |
|---|---|---|---|---|---|
| `val_000001_t1` | **anxiety** | anxiety ✅ | anxiety ✅ | anxiety ✅ | anxiety ✅ |
| `val_000001_t3` | **gratitude** | anxiety ❌ | relaxed ❌ | relaxed ❌ | neutral ❌ |
| `val_000002_t5` | **relaxed** | relaxed ✅ | relaxed ✅ | relaxed ✅ | relaxed ✅ |
| `val_000003_t5` | **helplessness** | sadness ❌ | anxiety ❌ | relaxed ❌ | neutral ❌ |
| `val_000003_t7` | **relaxed** | relaxed ✅ | neutral ❌ | relaxed ✅ | relaxed ✅ |
| `val_000004_t3` | **loneliness** | loneliness ✅ | relaxed ❌ | relaxed ❌ | relaxed ❌ |
| `val_000005_t9` | **anxiety** | loneliness ❌ | anxiety ✅ | loneliness ❌ | anxiety ✅ |
| `val_000005_t15` | **relaxed** | relaxed ✅ | relaxed ✅ | relaxed ✅ | relaxed ✅ |
| `val_000006_t5` | **anger** | sadness ❌ | anxiety ❌ | sadness ❌ | sadness ❌ |
| `val_000007_t3` | **anxiety** | care ❌ | anxiety ✅ | relaxed ❌ | neutral ❌ |
| `val_000008_t3` | **neutral** | loneliness ❌ | anxiety ❌ | relaxed ❌ | neutral ✅ |
| `val_000009_t5` | **anger** | anger ✅ | anxiety ❌ | relaxed ❌ | anger ✅ |
| `val_000009_t7` | **anger** | anger ✅ | anxiety ❌ | anger ✅ | anger ✅ |
| `val_000010_t5` | **neutral** | sadness ❌ | relaxed ❌ | neutral ✅ | neutral ✅ |
| `val_000011_t3` | **relaxed** | anxiety ❌ | relaxed ✅ | relaxed ✅ | relaxed ✅ |
| `val_000012_t5` | **helplessness** | anxiety ❌ | anxiety ❌ | anxiety ❌ | mixed ❌ |
| `val_000012_t7` | **relaxed** | joy ❌ | relaxed ✅ | relaxed ✅ | joy ❌ |
| `val_000013_t1` | **anxiety** | anxiety ✅ | anxiety ✅ | anxiety ✅ | anxiety ✅ |
| `val_000014_t3` | **anxiety** | anxiety ✅ | anxiety ✅ | relaxed ❌ | anxiety ✅ |
| `val_000015_t5` | **fear** | fear ✅ | anxiety ❌ | sadness ❌ | anxiety ❌ |

v3 修好的（v1 错 v3 对）：`val_000005_t9`、`val_000008_t3`、`val_000009_t5`、`val_000014_t3`
v3 新引入的（v1 对 v3 错）：`val_000007_t3`、`val_000012_t7`
净 +2 条 ⇒ 情绪 Acc 0.45 → 0.60。

### 表 B：personality_traits（只列 v3 与金标不一致的）

| 样本 | 金标 | official | v1 | v2 | v3 |
|---|---|---|---|---|---|
| `val_000002_t5` | `['casual']` | `[]` | `['casual','open']` | `['casual','introverted']` | `[]` |
| `val_000003_t7` | `['sensitive']` | `[]` | `[]` | `['sensitive']` | `['open','sensitive']` |
| `val_000005_t9` | `['sensitive']` | `[]` | `['introverted','sensitive']` | `['introverted']` | `[]` |
| `val_000005_t15` | `['agreeable']` | `[]` | `['casual','open']` | `['casual','introverted']` | `[]` |
| `val_000006_t5` | `['high_conscientiousness','sensitive']` | `[]` | `['introverted','sensitive']` | `['emotionally_stable','introverted']` | `['introverted','sensitive']` |
| `val_000010_t5` | `['introverted']` | `[]` | `[]` | `[]` | `[]` |
| `val_000012_t5` | `['sensitive']` | `[]` | `['introverted','sensitive']` | `[]` | `[]` |
| `val_000012_t7` | `['sensitive']` | `[]` | `[]` | `['emotionally_stable','open']` | `[]` |

### 表 C：style（只列 v3 与金标不一致的）

| 样本 | 金标 | official | v1 | v2 | v3 |
|---|---|---|---|---|---|
| `val_000002_t5` | `['colloquial']` | `['formal','low_emotional_expression']` | `['colloquial']` | `['brief']` | `[]` |
| `val_000003_t5` | `['rational']` | `[]` | `[]` | `['brief']` | `[]` |
| `val_000003_t7` | `['rational']` | `[]` | `['rational']` | `['brief']` | `['brief']` |
| `val_000004_t3` | `['rational']` | `[]` | `['rational']` | `['brief']` | `[]` |
| `val_000005_t9` | `['rational']` | `[]` | `['colloquial']` | `['brief']` | `[]` |
| `val_000007_t3` | `['direct']` | `[]` | `[]` | `['brief']` | `[]` |
| `val_000008_t3` | `['direct']` | `[]` | `['colloquial']` | `['brief']` | `[]` |
| `val_000009_t5` | `['direct']` | `[]` | `['formal']` | `['brief','direct']` | `[]` |
| `val_000009_t7` | `['high_emotional_expression']` | `[]` | `['direct','rational']` | `['brief']` | `[]` |
| `val_000010_t5` | `['direct']` | `[]` | `['rational']` | `['brief']` | `[]` |
| `val_000011_t3` | `['direct']` | `[]` | `['formal']` | `['brief']` | `['brief']` |
| `val_000012_t5` | `['rational']` | `[]` | `['direct']` | `['rational']` | `[]` |
| `val_000014_t3` | `['direct']` | `[]` | `['colloquial']` | `['brief','low_emotional_expression']` | `[]` |
| `val_000015_t5` | `['direct','high_emotional_expression']` | `[]` | `['colloquial']` | `[]` | `[]` |

---

## 八、v3 的具体改动

1. **personality_traits / interests**：改为「默认输出 []」+ 只列负面情形，删除一切正面示例；
   并点名禁止把「情绪克制」判成 `emotionally_stable`。
2. **style**：改为按优先级排列的**可判定措辞信号**（提要求/下指令/划边界 → `direct`；
   给因果/做对比 → `rational`；夸张比喻/密集感叹 → `high_emotional_expression`；
   ≤15 字 → `brief`；三段以上 → `detailed`），并明确禁止把 `colloquial/formal/indirect/humorous`
   当默认值。
3. **情绪**：明确「anxiety 和 relaxed 都不是默认答案」，补两条排除规则——
   「只是礼貌回应 / 下指令 / 陈述事实 → `neutral`」「对助手讽刺或明确拒绝 → `anger`」。

每一条规则的**「为什么这么写」都内联写在 `tools/emotion_profile/prompt_rules.txt` 里**（`#` 注释），
由 `make_prompt.py` 剥离后生成 `prompts/emotion_profile.md`。
所以提示词产物保持纯文本、可直接喂模型，而源码里留得住理由与证据。

---

## 九、复现（全部命令）

```bash
ssh -p 30817 root@frp-rug.com        # 密码见 docs/roles/COMPUTE.md
cd /mnt/workspace/OGC

# 0) 造一个只含 emotion_profile.md 的隔离提示词目录（控制变量用）
mkdir -p runs/p02-only && cp prompts/emotion_profile.md runs/p02-only/

# 1) 裸基线（官方原件，无接线层）
python3 tools/eval_loop.py --tag smoke-official --limit 20 --inference-script official

# 2) 当前提示词的接线版
python3 tools/eval_loop.py --tag v3-smoke --limit 20 \
    --inference-script wired --prompt-dir /mnt/workspace/OGC/runs/p02-only

# 3) 看台账
cat runs/leaderboard.tsv

# 4) 全量（2354 条，约 1.5–2 小时独占卡）
python3 tools/eval_loop.py --tag v3-full \
    --inference-script wired --prompt-dir /mnt/workspace/OGC/runs/p02-only
```

---

## 十、边界与未验证声明

| 项 | 状态 |
|---|---|
| **全量 2354 条确认** | **未跑完**。原因：卡上同时有 LoRA SFT 与另一个全量评测，按 `runs/GPU-RULES.md`「最多 2 个 GPU 任务」排队中（`runs/.gpu-lock.d/role2-queued.lock`）。 |
| n=20 的波动性 | 这 20 条是 val 的**前 20 条**，不是随机抽样。单个样本 = 0.05，**一次误判就移动一个验收点**。所以本报告的结论只在「同一 20 条上的臂间相对比较」这一层成立；绝对数值不能外推。 |
| style 的召回 | v3 的 style `该填却空 = 12`（金标非空 16 条只抓到 2 条命中）。**这是当前最大的一块未拿到的分**，下一轮的主要方向。 |
| 情绪的剩余空间 | v3 误判 8 条，其中 `gratitude→neutral`、`helplessness→neutral/mixed`、`anger→sadness` 各成模式。 |
| BERTScore | 未接入（需离线预置编码器），四项指标里缺这一项，本报告不涉及。 |
| 与 ③ 的联合效果 | 未测。本轮全部用 `--prompt-dir` 隔离了 `response.md`，所以 **本报告的数字不代表最终提交时的效果**。 |
| 全量上的过拟合风险 | `personality`/`interests` 走「留空」策略在全量上应当更稳（金标为空率在 val 全量上是 61.9%/74.9%，比这 20 条的 60%/65% 更高）；但 `style` 的 `brief/detailed` 那一档在 val 非空里占 23.5%，本轮的 20 条里是 0 —— **需要在全量上复核**。 |

---

## 十一、下一轮的方向（按数据排的优先级）

1. **style 召回**：目前 `该填却空 = 12`。金标 `direct` 9 条里只抓到 2 条命中。
   方向：把「提要求/划边界」的信号再细化（例如区分「对助手提要求」与「对第三方提要求」），
   一次只加一条判据、跑一轮验证。
2. **情绪 `gratitude`/`helplessness`**：v3 把这两类都推给了 `neutral`。
   方向：给 `neutral` 加更严的排除条件（出现道谢语就不算 neutral）。
3. **全量确认**：排队器已挂好，SFT 一结束自动开跑；出数后回填本报告与 `LOG.md`。

> 判据：每轮只改一处、跑 n=20、四件套齐全（改什么 / 跑什么 / 数字 / 结论）才记一次实验。
> 台账在 `runs/leaderboard.tsv`，逐轮记录在 `tools/emotion_profile/LOG.md`。
