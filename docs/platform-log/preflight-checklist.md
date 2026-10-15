# 提测前检查清单（角色① 维护）

> **只有队长能决定提测。** 这份清单的作用是：队长说「提」之前，逐条打勾；任何一条没过，
> 就不提。
>
> 为什么这么严：每队**最多 5 次测评成功机会**，**以最后一次成功提交的成绩计分**；
> 镜像运行失败虽然不占次数，但**卡时券只有 10 卡时、账户余额 0 AIQ**——额度耗在
> 环境问题上，后面就没得试了。

打勾方式：把 `[ ]` 改成 `[x]`，并在括号里写上证据（命令、文件路径、平台日志链接）。
**没跑过的不要打勾**，写「未验证」比写假勾安全。

---

## 一、镜像本身

- [ ] **入口命令正确**：组委会只执行 `bash /root/participant/start.sh TEST_FILE RESULT_DIR`
      （`GivenData/participant/README.md`）。镜像内 `ls -l /root/participant/start.sh` 有可执行权限。
- [ ] **`conda_env` 陷阱已处理**：官方 `start.sh` 原件第 5 行是 `source activate conda_env`，
      注释写着「名称固定，不要修改」——配上 `set -euo pipefail`，**没有这个 conda 环境就会直接退出**。
      我们的副本已改成「有 conda 就激活、没有就用 PATH 上的 python3」，**但必须在真实基础镜像里
      实测过一次**，不能只看代码。
      证据：____（在基础镜像里跑 `bash /root/participant/start.sh` 的首屏输出）
- [ ] **`--tensor-parallel-size` 不是 2**：官方默认 `VLLM_ARGS=(--tensor-parallel-size 2)` 是双卡配置；
      平台只提供 RTX 4090，我们按 1 卡申请。副本已改为 `VLLM_TENSOR_PARALLEL_SIZE`（默认 1）。
      （当前 `backend=transformers`，走不到 vLLM 分支，但改掉免得将来踩。）
- [ ] **镜像体积 ≤ 50 GB**：平台口径是「公有镜像 + 系统盘已用空间总和」（细则页原文）。
      1 卡规格的系统盘只有 **50 GB**，正好等于这个上限。
      证据：`docker images` 里本镜像的 SIZE = ____
- [ ] **权重打进镜像**：`ls /root/participant/model/*.safetensors` 有两个分片，
      共约 6.2 GB（`model-00001-of-00002.safetensors` 3,968,658,944 字节、
      `model-00002-of-00002.safetensors` 2,203,268,048 字节）。
      ⚠️ 权重**必须**在 `/root/participant/model/`，因为官方脚本按 `model_path="./model"`
      相对 `/root/participant` 解析。
- [ ] **提示词与 rag 已进镜像**：`ls /root/participant/prompts/`、`ls /root/rag/`。
      缺了也能跑（`wiring.py` 会退回官方提示词），但要**明确知道自己交的是哪一版**。

## 二、离线（最容易零分的一条）

- [ ] **断网真跑通过**：`docker run --rm --network none <镜像> bash /opt/offline_check.sh` 全绿。
- [ ] **没有外部网络依赖**：`offline_check.sh` 第 4 节的静态扫描无 `[FAIL]`。
      （唯一允许的 URL 是 `127.0.0.1` —— vLLM 本地服务。）
- [ ] **HF 离线开关在**：镜像里 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 已设；
      `run_inference.py` 加载模型时用的是 `local_files_only=True`。
- [ ] **没有运行时下载**：镜像内不存在 `pip install` / `huggingface_hub` / `snapshot_download` 调用。

## 三、产出格式

- [ ] **两个产物都生成**：`RESULT_DIR/submission.jsonl`、`RESULT_DIR/performance_report.json`。
- [ ] **行数一致**：`wc -l submission.jsonl` 与输入 JSONL 行数相同，**顺序一致**。
- [ ] **schema 过关**：每行恰好含 `id` / `response_text` / `emotion_label` / `user_profile` / `memory_refs`
      五个字段；`emotion_label` 在 16 类枚举内；`user_profile` 恰好三维、
      取值在 ASCII 词表内；`memory_refs` 匹配 `^mem_[0-9]{6}$`。
      工具：`python src/cli.py validate --submission <文件> --input <输入文件>`
- [ ] **ID 不自作主张**：沿用输入里的 id 原样输出。
      ⚠️ 已知矛盾：官方 schema 的正则是 `^test_[0-9]{6}_t[0-9]+$`，但官方样例 id 是
      `test1_000001_t5`（多一个 `1`）。我们采「**沿用输入 ID**」策略，可用
      `python src/cli.py validate --submission X --input Y` 核对；不要手动重编号。
- [ ] **解析失败数为 0**（或已知且可解释）：看 `performance_report.json` 的
      `parse_failure_count` 与 `parse_failure_ids`。

## 四、性能口径

- [ ] **`performance_rounds` 与实测条数匹配**：只有够 100 条计时样本时
      `performance_report.json` 的 `complete` 才是 `true`。正式提交的测试集远超 100 条，
      但**本地用 3 条样例时 `complete=false` 是正常的**，别误判成失败。
- [ ] **计时范围没被动过**：`timing_scope` 必须是
      「tokenize+generate+decode+parse+validate；不含模型加载」。
      改这个口径等于自己骗自己。
- [ ] **`hardware_label` 是真实值**：不能留 `fill-in-before-official-run`。
      副本里已填 `migu-aigpu-rtx4090-1card`，**上机实测后按实际情况更正**。

## 五、平台与额度

- [ ] **本地全流程先跑通**，再上平台（余额 0 AIQ、卡时券 10 卡时，浪费不起）。
- [ ] **实例规格与花费已记录**：每次上机写 `docs/platform-log/YYYY-MM-DD-<事由>.md`，
      含实例规格、镜像、命令、耗时、AIQ/卡时消耗、原始日志或截图（存 `docs/platform-log/assets/`）。
- [ ] **平台运行记录连续可查**：这是专家评审 30 分里的**硬性 10 分**（训练日志、调优过程、
      平台使用规范），**事后补不出来**，必须从第一次上机就开始记。
- [ ] **参赛镜像已上传**：实例列表页提示「在提交作品测评之前，需要先上传您的参赛镜像」。

## 六、提交前最后一遍（队长执行）

- [ ] 队伍信息不再需要改（**初赛作品上传截止后不支持修改队伍信息**）。
- [ ] 认可「**以最后一次成功提交计分**」：所以最后一次提交必须是最好的那一版，
      不要拿未验证的版本去占次数。
- [ ] 两次提交之间留有足够时间。

---

## 当前仍未闭合、会影响提测的未知项

| 未知项 | 影响 | 当前处理 |
|---|---|---|
| ID 正则与官方样例矛盾 | 可能整批 id 被判不合规 | 沿用输入 ID；提测前用官方 3 条样例核对 |
| `memory_refs` 是否计分 | 决定要不要花力气做记忆抽取 | 官方基线提示词让它固定输出 `[]`；本地默认不计入画像主指标 |
| 推理性能满分的延迟阈值 | 不知道多少毫秒算满分 | 只能尽量压低单条延迟 |
| 三项回复指标的权重 | 不知道 BLEU/ROUGE/BERTScore 谁更重要 | 三项都报，不做取舍假设 |
| 数字人赛道是否有官方自检脚本约定 | 若真有 `run_all.sh`/`check_solution.py`，我们没提供 | 官方材料里查无此项（全属乒乓球赛道）；按 `start.sh` 单入口准备 |
| 平台公有镜像里到底有没有 `conda_env` | 决定 `start.sh` 走哪条分支 | 副本已做双路守卫，但**必须在真实基础镜像里实测一次** |

---

## 本地已知限制（诚实记录，不要当成已通过）

- **本机没有 bash**（`Get-Command bash` 为空，Git Bash 也不在常见路径）：
  `deploy/build.sh`、`deploy/offline_check.sh`、`deploy/participant/start.sh`
  **从未在本机执行过**，只做过逐行人工检查。这三条必须在 Linux/WSL 或平台上首次实测。
- **本机没有 docker 构建记录**：镜像尚未真正构建过一次，50 GB 体积是估算而非实测。
- 以上两条在首次上机前应保持「未验证」，不得在专家材料里写成已完成。
