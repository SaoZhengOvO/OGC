# 提交操作手册（2026-11-05 版）

> 目标：今天把镜像交上去。这份文档只写**动作和判据**，不重复讲原理——原理在
> `docs/reports/technical-plan.md`，材料清单在 `docs/reports/review-package.md`。
>
> 一句话流程：**等构建 → 下 tar → 验 tar → 上传镜像 → 起实例跑 3 条样例 → 通过就提交测评。**

---

## 〇、现在的状态（本文件写定时）

| 项 | 状态 |
|---|---|
| 代码 | `0554a52`（main，已推送） |
| 构建 | GitHub Actions run `42043416912`，约 32 分钟，产物名 `ogc-participant-image` |
| 镜像内容 | 基座 Qwen2.5-3B-Instruct + 本地提示词 v3/v8 + 画像 `force` + RAG（默认关，见 §四） |
| 评审材料 | 9 个文件齐全（`pack_review.py` 本地实跑「材料齐全」） |
| 平台额度 | 5 次成功测评机会，**已用 0 次**；卡时券 10 卡时 |
| 计分 | **以最后一次成功提交的成绩计分** |

⚠️ 上一次成功构建是 `7ee9df0`，产物保留 7 天后已在 11-05 之前被清空，**旧 tar 已经没有了**；
而且那一版构建时评审材料里队名还是 `<队名>` 占位符。所以今天必须用新构建的这一版。

---

## 一、拿 tar（约 10 分钟，看网速）

构建变绿之后：

```powershell
cd D:\Documents\deepseekharness-workspace\OGC\OGC
gh run download 42043416912 -n ogc-participant-image -D deploy\dist\artifact
```

下载完目录里应该有四个文件：

| 文件 | 期望 |
|---|---|
| `ogc-participant-OGC.tar.gz` | 约 **8.2 GB** |
| `ogc-participant-OGC.tar.gz.sha256` | 一行摘要 |
| `image-inspect.json` | CI 里那次内容检查的报告 |
| （无） | —— |

**校验摘要**（这一步别跳，下载断了 tar 就是废的）：

```powershell
cd deploy\dist\artifact
$actual = (Get-FileHash ogc-participant-OGC.tar.gz -Algorithm SHA256).Hash.ToLower()
$expect = (Get-Content ogc-participant-OGC.tar.gz.sha256 -Raw).Split()[0]
if ($actual -eq $expect) { "OK  $actual" } else { "MISMATCH  实际=$actual  期望=$expect" }
```

---

## 二、验 tar（5 分钟，只读，不占额度）

```powershell
cd D:\Documents\deepseekharness-workspace\OGC\OGC
.venv\Scripts\python.exe tools\inspect_image_tar.py deploy\dist\artifact\ogc-participant-OGC.tar.gz `
  --expect /root/participant/start.sh `
  --expect /root/participant/run_inference.py `
  --expect /root/participant/configs/inference_config.json `
  --expect /root/participant/model/config.json `
  --expect /root/participant/model/model-00001-of-00002.safetensors `
  --expect /root/participant/model/model-00002-of-00002.safetensors `
  --expect "/root/AI技术赛道-数字人综合情感对话模型-OGC/01-technical-plan.md" `
  --expect "/root/AI技术赛道-数字人综合情感对话模型-OGC/02-experiments.md" `
  --expect "/root/AI技术赛道-数字人综合情感对话模型-OGC/03-reproduce.md" `
  --expect "/root/AI技术赛道-数字人综合情感对话模型-OGC/06-leaderboard.tsv" `
  --json-out deploy\dist\artifact\inspect-local.json
```

**通过判据**：末行 `FAIL 0`，且 `06-leaderboard.tsv` 在列（上一版镜像就是缺这个）。
`--expect-dir` 那部分会拿本地代码逐字节比对，构建后本地若又改过代码会有差异，属正常——
以「构建那次 CI 变绿」为准，不必因此重下。

---

## 三、上传镜像（平台操作，不占测评次数）

1. 咪咕仝学平台 → AI 算力实例 → 镜像管理 → 上传参赛镜像 → 选 `ogc-participant-OGC.tar.gz`。
2. 等状态变成「可用」。8 GB 上传按上行 20 Mbps 估约 **55 分钟**，按 10 Mbps 估约 **110 分钟**。
   **这是今天最长的不可压缩环节**，先启动再去做别的。
3. 记一条平台运行记录：`docs/platform-log/2026-11-05-image-upload/README.md`
   （日期、镜像哈希、上传起止时间、平台返回的镜像 ID）。这属于专家评审 30 分里的「平台使用」10 分。

---

## 四、起实例跑 3 条样例（预演，判 K1）

**目的**：证明「镜像在他们的环境里能起来并产出合法文件」。这一步不提交成绩，因此不消耗 5 次机会。

1. 创建实例：算力规格 **1 卡**（RTX 4090），镜像选刚上传的那个。
2. 进容器后确认三件事：

```bash
ls -l /root/participant/start.sh                     # 有 x 权限
python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
ls /root/participant/model/*.safetensors | wc -l     # 2
```

3. 跑官方样例（3 条）：

```bash
mkdir -p /root/result
time bash /root/participant/start.sh /data/test_inference_data.jsonl /root/result
```

4. 收产物：

```bash
wc -l /root/result/submission.jsonl          # 必须是 3
head -c 600 /root/result/submission.jsonl
python3 -c "import json;d=json.load(open('/root/result/performance_report.json'));print(json.dumps(d,ensure_ascii=False,indent=2)[:1200])"
```

### 通过判据

| 检查 | 通过 | 不过怎么办 |
|---|---|---|
| `start.sh` 退出码 0 | 是 | 看 `[start.sh]` 首屏是否 conda 分支异常；贴报错再判 |
| `submission.jsonl` 行数 = 3 | 是 | 看 `performance_report.json` 的 `parse_failure_count` |
| 每行五个字段齐全 | 是 | 用 `python3 /root/participant/../../src/cli.py` 不可用时就地手查 key |
| `emotion_label` 非空 | **至少 2/3** | 见 §六 风险 A |
| 单条延迟 | < 8000 ms | 平台卡更强，正常应更快；明显更慢说明在跑 CPU |

**首次加载时间**要记下来（模型 5.76 GB 从镜像层读进显存，预计 20–60 s），这是平台侧真实耗时的关键数据。

---

## 五、提交测评（占 1 次机会）

预演通过后：

1. 平台 → 作品提交 → 选镜像 → 提交测评。
2. 等结果，记录：总分、三项自动分、耗时、任何报错原文。
3. 写 `docs/platform-log/2026-11-05-eval-01/README.md`，把平台返回的原始 JSON/截图存 `assets/`。

**判据**：只要出现分数，这次就算成功提交，且会**覆盖**之前成绩。所以提交前最后一次确认：
本地跑的是不是同一个配置（§四 预演已验证），以及**这一版是不是你能接受的最好版本**。

---

## 六、已知风险与对策

### 风险 A：`emotion_label` 为空（25 分项，最要命）

本地 `tools/chat.py` 实跑发现：单轮问候场景下模型没吐出 `emotion_label`，塌陷重试一次仍为空。
全量 2354 上情绪命中率 0.404，空值会被判错。

**对策**：预演时看 3 条里空了几条。
- 0–1 条空 → 照常提交。
- 2–3 条空 → 先别提交，回来加一条「解析失败就按多数类 `anxiety` 兜底」的改动（改动很小，
  但要走一次重建，约 35 分钟）。

### 风险 B：镜像体积

平台口径是「公有镜像 + 系统盘已用空间 ≤ 50 GB」。我们 tar.gz 8.2 GB，解压约 10 GB，
1 卡规格系统盘 50 GB —— 理论上够，但**没有任何实测**。上传成功即说明平台接受了这个体积。

### 风险 C：`hardware_label`

`start.sh` 现在写的是 `$(hostname)`，跑到哪台机器记哪台，不会出现「写 RTX4090 实际不是」的假话。
专家评审对照平台记录时这是加分项，不用改。

### 风险 D：RAG 默认关

`PARTICIPANT_RETRIEVED_IN_SYSTEM` 的缓解**没有验证过**，RAG A/B 是负结果（n=200，
BLEU 无显著差异、情绪掉 5.5pp）。所以镜像里检索默认不启用——这是有意的决定，不是漏配。
若被问到，答复口径：见 `docs/reports/experiments.md` 的 RAG 小节。

---

## 七、今天的时间预算

| 环节 | 预计 | 谁做 |
|---|---|---|
| 等构建变绿 | 32 min | 自动 |
| 下载 + 校验 + 验 tar | 15–30 min | 你 |
| 上传镜像到平台 | 55–110 min | 你，**尽早开始** |
| 起实例预演 | 15 min | 你 |
| 提交测评 + 等结果 | 10–40 min | 平台 |

**关键路径是上传**。构建一变绿就先下载、先开始上传，验 tar 可以并行做。
