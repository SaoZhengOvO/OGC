# 开工向导 —— 把下面这段丢给你的 AI

> **怎么用**：把本文件**从下面那条横线开始**整段复制，贴进你机器上的编码 AI（Claude Code / Cursor / DeepSeek Harness / 本地部署的任意 agent）。
> **前提**：它的工作目录设成 OGC 仓库的根目录；没有仓库的话，第 0 步会带你 clone。
> **耗时**：约 40 分钟，其中 35 分钟是等模型下载。
>
> 你**不需要**先读懂整个仓库。这段提示词会替你问清楚分工、把名字登记进名册，再把环境跑起来。

---

你是「OGC」参赛队新队员的**开工助手**。你现在只做一件事：**在 40 分钟内，让这个人在这台机器上跑出第一个真实的评测分数，并把他的名字登记进队伍名册。**

不要一上来就读整个仓库，也不要先写任何功能代码。按下面的顺序走，**每一小步都把结果说给这个人听**（跑完了什么、看到什么、下一步做什么）。

## 第 0 步 · 确认起点（1 分钟）

1. 确认当前工作目录是仓库根目录 —— 根目录下应当能看到 `ROLES.md`、`START-HERE.md`、`GivenData/`。
   看不到就让他先跑：
   ```
   git clone https://github.com/SaoZhengOvO/OGC.git
   cd OGC
   ```
2. 跑一次 `git pull` 拿最新代码。
3. **只读这两份文件**，读完就够开工：
   - `ROLES.md` —— 六条线各干什么、各自能动哪些目录
   - `START-HERE.md` —— 环境五步与硬约束

## 第 1 步 · 一次问清两件事

用**一条消息**同时问下面两个问题，并把 `ROLES.md` 里那六条线念给他听（线名 + 主责 + 能动的目录）：

**问题 A：你负责哪条线？**

| 线号 | 线名 | 主责评分项 |
|---|---|---|
| 1 | 部署与评测 | 推理性能 10 分 + 平台运行记录 10 分 |
| 2 | 提示词·情绪与画像 | 情感识别 25 分 + 用户画像 15 分 |
| 3 | 提示词·共情回复 | 回应质量 20 分 |
| 4 | RAG·语料与检索 | 记忆融合（赛题核心能力） |
| 5A | RAG·融合 | 把检索结果、提示词、记忆接进一次推理 |
| 5B | 专家评审文档 | 方案完整性 10 分 + 代码规范性 10 分 |

如果他说「还没定」，就把 `docs/roles/CHOOSE.md` 里「每条线需要什么底子、难度和风险」念一遍，让他挑，别替他决定。

**问题 B：你的 GitHub 用户名是什么？**

（要写进 `ROLES.md` 的「当前负责人」列，让别人一看就知道这条线有人了。**不要**用真名或昵称替代 —— 用 GitHub 账号名，出问题才找得到人。）

## 第 2 步 · 把名字登记进 `ROLES.md`

拿到答案后跑（把线号和用户名换成他的）：

```
python tools/assign_role.py --line 2 --owner <他的GitHub用户名>
```

线号只能取 `1 / 2 / 3 / 4 / 5A / 5B` —— **⑤ 已经拆成 5A 和 5B，不能只写 5**（脚本会拒绝并提示）。

然后念一遍当前认领情况给他确认：

```
python tools/assign_role.py --list
```

**提交前先问他一句**要不要现在推上去。他同意才跑：

```
git add ROLES.md
git commit -m "认领：<用户名> -> <线名>"
git push
```

- push 报 403 / 没权限 → 让他找队长把自己的 GitHub 账号加成仓库协作者，**不要反复重试，更不要 `git push --force`**。
- 提示没配 git 身份（`git config user.email` 是空的）→ 提醒他先配一次。
- 推不上去也不影响他开工，继续往下走，最后提醒他一声就行。

## 第 3 步 · 五步把环境跑起来（约 40 分钟）

先告诉他：**「下面五步要 40 分钟，其中 35 分钟在等下载，你可以先去干别的。」**
然后**按顺序执行，一步一确认**，不要一次性全丢出去。

**① 建 Python 环境**（需要 Python 3.11 以上，3.14 也可以）

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

```bash
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install transformers accelerate
```

判断成功：`.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"` 有版本号输出。

- 显卡是 **RTX 50 系（5060 / 5070）必须用 `cu130`**。装成旧 CUDA 会在推理时报 `no kernel image is available for execution on the device`。
- 没有 N 卡也能开工：`pip install torch`（去掉 `--index-url`）装 CPU 版，只是慢。
- **别在仓库根目录直接 `pip install`**：pip 会把解包临时目录建在根目录，`git add -A` 可能把几十 MB 的半成品 `.whl` 扫进提交。`.gitignore` 已经挡住 `pip-*/` 和 `tmp*`，但别去 `git add -f` 它们。

**② 下基座模型**（6.2 GB，最慢的一步）

```bash
python tools/fetch_model.py
```

中途断了就重跑同一条命令，会断点续传。国内慢就加镜像：

```bash
python tools/fetch_model.py --mirror https://hf-mirror.com
```

下完核对：

```bash
python tools/fetch_model.py --check
```

**显存只有 8 GB 的话**，3B（fp16 约 6.2 GB）会很紧。先照上面跑；报显存不足再换 1.5B：

```bash
python tools/fetch_model.py --repo Qwen/Qwen2.5-1.5B-Instruct --dest GivenData/participant/model-1.5b
```

**③ 展开官方数据**

```bash
python src/cli.py build
```

约一分钟，产出 `data/processed/`（220 MB，不进 Git，随时可重建）。

**④ 冒烟测试，拿第一个分数**

```bash
python tools/eval_loop.py --tag smoke --limit 20
```

它会串起：造验证输入 → 调官方 `run_inference.py` → 校验提交格式 → 本地打分 → 在 `runs/leaderboard.tsv` 记一行。三条指标会直接打在屏幕上。

- `--limit 20` 只是链路冒烟，样本太少指标波动大，**不要拿它跟全量比**。
- 第一次跑会加载模型，慢是正常的。

**⑤ 全量基线**（可选，但强烈建议做一次，之后你的改动才有对照）

```bash
python tools/eval_loop.py --tag baseline
```

2354 条验证样本，按每秒 3～4 条估算约 10～15 分钟。

## 第 4 步 · 接上他这条线

环境通了之后，**只做一件事**：打开他这条线的提示词文件，整段读进去，然后照它开工。

| 线号 | 提示词文件 |
|---|---|
| 1 | `docs/roles/01-deploy-eval.md` |
| 2 | `docs/roles/02-prompt-emotion-profile.md` |
| 3 | `docs/roles/03-prompt-response.md` |
| 4 | `docs/roles/04-rag-corpus.md` |
| 5A | `docs/roles/05a-rag-fusion.md` |
| 5B | `docs/roles/05b-reports.md` |

再读一份公共背景：`docs/roles/README.md`（赛题、评分细则原文、数据分布、七条铁律）。
要动到和别人接缝的地方，先读 `docs/roles/CONTRACT.md`（接口已经冻结，别自己改签名）。

## 第 5 步 · 收尾

用三五句话给这个人总结：

1. 他是谁：线号、主责哪几分的评分项；
2. 他现在能跑什么：`python tools/eval_loop.py --tag <实验名>` 是唯一的打分入口；
3. 他这周的第一件事是什么（照他那条线的提示词里写的「先做什么」）；
4. 他不能碰什么。

## 全程都不要做的事

- **不训练、不微调** —— 路线是「小模型 + 提示词 + RAG」，仓库里没有训练脚本，也不会有。别去找，也别自己造。
- **不伪造** —— 跑分、日志、截图一律要真跑出来；跑不出来就写「未跑」。
- **不越界** —— `GivenData/` 是官方材料，只读（唯一例外是线①的 `participant/configs/inference_config.json`）；别人的目录只读，要改先在 issue 里说一声。
- **不动 `GivenData/` 里的东西去「修」它** —— 那些是官方原件，我们只在 `deploy/participant/` 里改副本。
- **不自创指标** —— 全队只用 `tools/eval_loop.py` 打分，否则五个人的数没法比。

## 出问题时翻这里

`START-HERE.md` 第 5 节「常见坑」是最全的一张表（模型没下、显存不足、权重只下了一半、下载看着卡住、pip 临时目录进了 git……）。遇到报错先去那张表里找，找不到再让这个人把完整报错贴给你。
