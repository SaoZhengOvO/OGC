# 专家评审材料包：清单与打包

> 作者：角色⑤B（专家评审文档；同时持有角色①部署与评测）。对应官方评分项「专家评审 — 方案完整性（10 分）」与「平台运行记录（10 分）」。
> **队名：`OGC`**（已确认）。目录名与打包参数都用这个值。
> 官方要求（`GivenData/README.md`）：专家评审资料打包放进镜像 `/root` 下，
> 目录命名 `AI技术赛道-数字人综合情感对话模型-OGC`。

## 一、材料清单

`/root/AI技术赛道-数字人综合情感对话模型-OGC/` 下应包含：

| 文件 | 来源 | 作用 | 状态 |
|---|---|---|---|
| `MANIFEST.md` | 打包脚本生成 | 本目录的索引 + 每个文件的 sha256 | 自动 |
| `01-technical-plan.md` | `docs/reports/technical-plan.md` | 技术方案（对应方案完整性 10 分） | 已就绪 |
| `02-experiments.md` | `docs/reports/experiments.md` | 实验设计与结果（基线、消融、负结果） | 已就绪（数字待补） |
| `03-reproduce.md` | `docs/reports/reproduce.md` | 从裸机到提交的完整复现步骤（对应代码规范性 10 分） | 已就绪 |
| `04-review-package.md` | `docs/reports/review-package.md` | 本文件 | 已就绪 |
| `05-data-quality/` | `data/reports/{train,val}_quality.json`、`dataset_summary.json` | 数据质量核查留档（口径与异常样本 ID） | 已就绪 |
| `06-leaderboard.tsv` | `runs/leaderboard.tsv`，没有则用入库快照 `docs/reports/leaderboard.tsv` | 每一次评测的指标台账（**不手写，由脚本追加**） | 已就绪（54 行） |
| `07-platform-log/` | `docs/platform-log/*.md` + `assets/` | **平台运行记录**：每次上机的时间、机型、命令、结果、截图 | **必须从第一天开始记** |
| `08-code-review.md` | `docs/reports/code-review.md` | 代码规范性自检清单与问题分派 | 已就绪 |

### 两个必须强调的点

1. **「平台运行记录」是硬性要求，事后补不出来**（`README.md` 与官方评分细则均列为 10 分项）。
   每次在咪咕仝学平台上的操作都要落一份 `docs/platform-log/YYYY-MM-DD-<事由>.md`，
   附上机型（RTX 4090 / 几卡）、额度消耗、执行的命令、产物截图。
2. **台账不许手写**：`06-leaderboard.tsv` 由 `tools/eval_loop.py` 每次评测追加一行，
   列固定为 `time / tag / backend / model / n_samples / n_comparable / emotion_acc /
   profile_acc / bleu4 / rouge_l / memory_f1`。手抄的数字一旦与产物对不上，专家评审会直接扣分。
   **但「不手写」不等于「只在一台机器上」**：`runs/` 被 `.gitignore` 排除，而大规模评测跑在
   AMD 盒子上，本地 `runs/leaderboard.tsv` 只有 9 行、盒子有 45 行——2026-10-17 就是这样，
   报告里写着 n=2354 的 0.404，评审材料里的台账却停在更早的一行。现在用
   `tools/merge_leaderboard.py` 合并多机台账，把结果作为快照 `docs/reports/leaderboard.tsv`
   入库，打包脚本优先用本机最新的、缺失时退回快照。

## 二、打包脚本

放在 `docs/reports/pack_review.py`（纯标准库）。它把上表里存在的文件复制到目标目录并生成
`MANIFEST.md`（含每个文件的 sha256 与字节数），**缺失的项会明确打印 `[缺]` 而不是静默跳过**。

```bash
python docs/reports/pack_review.py --team OGC --out deploy/dist/review-package
```

生成后，把整个目录放进镜像的 `/root/` 下即可（`deploy/build.sh` 组装 context 时可直接拷贝）。

### 为什么写 sha256

专家评审关心「可复现性」。每个文件带一个 sha256，评审可以确认这份材料与仓库里跑出这些数字的
那一版代码是对应的；我们自己回看时也能确认没有拿旧报告冒充新结果。

## 三、镜像里的落点

```
/root/
├── participant/                        # 官方入口与推理脚本（角色①）
│   ├── start.sh
│   ├── run_inference.py
│   ├── wiring.py
│   ├── configs/inference_config.json
│   ├── templates/
│   ├── prompts/                        # 角色②③ 的提示词
│   └── model/                          # Qwen2.5-3B-Instruct，约 5.8 GB
├── rag/                                # 角色④⑤A 的检索与融合
│   ├── corpus/  index/  retrieve.py  memory.py  fusion.py
└── AI技术赛道-数字人综合情感对话模型-OGC/   # 本材料包
    ├── MANIFEST.md
    ├── 01-technical-plan.md ...
    └── 07-platform-log/
```

注意：`prompts/` 放在 `/root/participant/prompts`，`rag/` 放在 `/root/rag`。
接线层与融合层通过环境变量 `PARTICIPANT_CONTENT_ROOT` / `PARTICIPANT_PROMPT_DIR` 定位，
默认值与上表一致；本机开发时（不设环境变量）会自动回退到仓库内的相对路径。

## 四、提交前的最后一道门

按 `docs/platform-log/preflight-checklist.md` 逐条打勾，其中与本材料包相关的三条：

- [ ] `MANIFEST.md` 里每个文件的 sha256 与实际文件一致（打包脚本会自检）
- [ ] `06-leaderboard.tsv` 里的基线数字与 `runs/eval-<tag>/` 下的产物对得上
- [ ] `07-platform-log/` 至少有本队第一次上机的记录（含机型、额度、截图）
