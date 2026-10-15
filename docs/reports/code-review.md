# 代码规范性评审（问题清单，带行号）

> 角色⑤B · 2026-10-15 · 对应 `docs/reports/README.md` 第 2 条（B 项，无前置依赖）
> **基线**：`main` 上 `450956f`（角色②的四个新脚本已并入）时的代码树，27 个文件 / 6,562 行。
> 行号与计数会随代码变动漂移，**重跑 `audit_code.py` 以当次输出为准**。

本报告回答一个问题：**如果专家只翻我们的代码，他会挑出什么？**

结论不是「我们的代码有问题」，而是**我们的标准是在项目中途才立起来的，而且没有回头补齐**：
最近写的 `rag/fusion.py`、`deploy/participant/wiring.py` 一个不漏地写了文档字符串，角色②当天提交的
四个脚本也有 88% / 75%，而最早落地的 `src/`、`tools/` 只有两成到五成。这个断层比任何单个缺陷都更显眼
——因为它一眼就能看出来。

## 一、这份报告是怎么来的（可复现）

不靠肉眼看。跑一个脚本，它扫 27 个 Python 文件 / 6,562 行，逐项输出证据：

```bash
python docs/reports/audit_code.py
# 完整结果（含逐文件明细）：docs/reports/code-review.audit.json
```

脚本只读不写，检查项：模块 docstring、公开函数有无文档字符串与类型标注、类 docstring、
写死的 Windows 绝对路径、中文标识符、裸 `except`、TODO/FIXME、超过 120 字符的行、
跨文件重复的连续代码块（规范化后 6 行窗口）。

**判定标准取自仓库自己**：`src/*.py` 的既有风格（模块必有 docstring、公开函数必有文档字符串与类型标注）。
没有引入外部 linter，因为要新增依赖，而赛题禁止。

> 一处诚实交代：巡检工具**不检查它自己源码里的逐行文本**（`docs/reports/audit_code.py`）。
> 因为它必然包含要搜的那些字面量——`TODO|FIXME` 的正则、`D:/...` 的示例注释——逐行查会对自身产生
> 假阳性。该文件仍参与 AST 检查（docstring、类型标注、中文标识符）。规则写在脚本的
> `TEXT_CHECK_EXEMPT` 里，可复核。

## 二、文档字符串覆盖率：断层在哪里

| 公开函数 | 缺文档 | 覆盖率 | 文件 |
|---:|---:|---:|---|
| 10 | 0 | **100%** | `rag/fusion.py` |
| 7 | 0 | **100%** | `deploy/participant/wiring.py` |
| 4 | 0 | **100%** | `src/label_space.py` |
| 8 | 1 | 88% | `tools/emotion_profile/report.py` |
| 7 | 1 | 86% | `tools/assign_role.py` |
| 4 | 1 | 75% | `tools/emotion_profile/badcase_pack.py` |
| 4 | 1 | 75% | `tools/gen_label_block.py` |
| 17 | 8 | 53% | `src/data_processor.py` |
| 2 | 1 | 50% | `tools/emotion_profile/make_prompt.py` |
| 2 | 1 | 50% | `tools/emotion_profile/make_val_input.py` |
| 6 | 3 | 50% | `docs/reports/pack_review.py` |
| 15 | 9 | 40% | `src/evaluate.py` |
| 6 | 4 | 33% | `src/cli.py` |
| 6 | 4 | 33% | `src/validate_submission.py` |
| 14 | 10 | 29% | `docs/reports/audit_code.py` |
| 4 | 3 | 25% | `tools/bootstrap.py` |
| 11 | 9 | 18% | `tools/eval_loop.py` |
| 8 | 7 | 12% | `tools/fetch_model.py` |
| 1 | 1 | 0% | `deploy/make_val_input.py` |
| 19 | 18 | 5% | `deploy/participant/run_inference.py`（官方原件） |

（测试文件按惯例豁免，未列入统计；`docs/reports/` 下两个脚本是本次写报告用的，
**用同一把尺子量了自己**，不给自己开绿灯。最后一行的官方原件见 §五说明。）

**结论**：不是「有人不写文档」，而是**新写的脚本在补、早期的没回头补**。
`src/data_processor.py`、`src/evaluate.py`、`src/cli.py` 是最早落地的三个文件，
也正是三个人每天都要用的三个文件——写的时候还没有这条规范，之后没有人回过头来补。

一个**当天新增的反例**：角色②刚提交的四个脚本（`450956f`）覆盖率 88% / 75% / 50% / 50%，
四个模块文档字符串齐全，但四个 `main()` 全部没有文档字符串。这说明规范的传播靠的是
「写代码时脑子里有没有这根弦」，而 `main()` 恰恰是最容易被跳过的地方——这条已经列入 F6。

未写文档的具体位置（按文件）：

```
src/data_processor.py:81 add_response_length   :84 summary   :87 pct   :138 read_jsonl
src/data_processor.py:142 write_jsonl   :153 write_json   :177 turn_id_of   :258 infer_prefix
src/evaluate.py:88 get_tokenizer   :116 add   :126 score   :174 add   :188 score
src/evaluate.py:213 add   :224 score   :247 to_json   :431 cosine
src/cli.py:168 cmd_eval   :233 cmd_validate   :250 build_parser   :301 main
src/validate_submission.py:61 render   :75 ok   :78 render   :312 main
tools/eval_loop.py:73 read_jsonl   :81 write_jsonl   :147 run_step   :154 score
tools/eval_loop.py:178 extract_row   :194 format_cell   :202 append_leaderboard
tools/eval_loop.py:211 build_parser   :235 main
tools/fetch_model.py:62 human   :75 request   :82 list_files   :97 resolve_url
tools/fetch_model.py:119 wanted   :125 download_one   :198 main
tools/bootstrap.py:43 venv_python   :49 run   :61 main
tools/assign_role.py:145 main
tools/gen_label_block.py:88 main
deploy/make_val_input.py:36 main
```

## 三、问题清单

### F1 · 中 · 三个共享目录没有归属人

**证据**：`docs/roles/CONTRACT.md:7-18` 的目录所有权表只列了 `prompts/`、`rag/`、`deploy/`、`docs/`；
`ROLES.md` 的「独占写文件范围」列同理。**`src/`、`tools/`、`tests/` 不在任何一条线的名下。**

**影响**：这不是文档疏漏，是分工的真实状态——评测裁判（`src/evaluate.py` 决定全队每天看到的所有分数）、
数据适配器、以及所有工具脚本处于「谁都可以改、也谁都不负责」的区间。上面那张覆盖率表里垫底的文件，
**恰好就是无主的那几个**。代码规范性 10 分里最容易拿的部分，没有人在盯。

**建议**：在契约里把 `src/`、`tools/`、`tests/` 显式定为公共设施，并指定一个维护人。归谁都可以，但要有名字。

### F2 · 低 · 同一段 JSONL 读取逻辑写了两遍，且抛出两种异常

**证据**：

- `src/data_processor.py:125-135` `iter_jsonl()` — 坏行抛 `DataContractError`
- `src/evaluate.py:475-486` `load_predictions()` — 同一逻辑，坏行抛 `ValueError`

**影响**：「第 N 行 JSON 坏了」这一种故障，在两个模块里是两个不同的异常类。调用方要么同时 catch 两个，
要么在某条路径上漏掉。目前没有测试覆盖「坏行」这个分支，所以它是静默的。

**建议**：`evaluate.py` 改为复用 `data_processor.iter_jsonl`，或统一到同一个异常。两处都是十几行样板代码。

### F3 · 低 · 两个 `load_prompt_block` 行为不同（**判定：不是 bug，但需注明**）

**证据**：

- `rag/fusion.py:453-465` `_load_prompt_block()` — 遍历**多个**候选目录 × 多个扩展名
- `deploy/participant/wiring.py:92-103` `load_prompt_block()` — 只读**一个**固定目录

两处连续 6 行代码完全相同（`path.is_file()` → `read_text` → 空则继续），因此被重复检测抓到。

**判定**：这是**有意的分歧**，不是缺陷。容器里提示词只有一个合法位置（`/root/participant/prompts/`），
开发树里则要能容忍候选目录。把它们强行合并反而是错的。

**但**：两个同名（差一个下划线）的函数，行为不同，而且都叫 `load_prompt_block`。将来有人改了一个，
不会传导到另一个，也不会报错——只会表现为「开发环境对、镜像里不对」这类最难查的偏差。

**建议**：在 `rag/fusion.py:453` 的 docstring 里加一句「容器内的对应实现是
`deploy/participant/wiring.py` 的 `load_prompt_block`，两者刻意不同：那边只认单一目录」。

### F4 · 低 · 生产代码 8 处公开函数缺类型标注

```
tools/eval_loop.py:73  read_jsonl      （缺返回类型）
tools/eval_loop.py:81  write_jsonl(rows)
tools/eval_loop.py:147 run_step(cmd, env)
tools/eval_loop.py:194 format_cell(value)
tools/eval_loop.py:235 main(argv)
tools/fetch_model.py:75  request       （缺返回类型）
tools/fetch_model.py:198 main(argv)
tools/bootstrap.py:61  main(argv)
```

注意 `main(argv)` 这个写法在三个 CLI 里一致地缺标注——是同一个习惯，改起来也是同一个动作。

### F5 · 低 · 4 个生产类缺 docstring

```
src/evaluate.py:169  RougeLAccumulator
src/evaluate.py:242  EvaluationReport
src/validate_submission.py:55  ValidationIssue
src/validate_submission.py:66  ValidationResult
```

`RougeLAccumulator` 尤其值得补：它决定分数怎么累计，但没人能从类名看出它按什么口径聚合。

### F6 · 中 · 同一段 JSONL 读取逻辑现在有四份，抛出三种不同的东西

**证据**（F2 的扩大版，角色②当天提交的脚本让它翻了一倍）：

| 位置 | 坏行时抛出 |
|---|---|
| `src/data_processor.py:125-135` `iter_jsonl()` | `DataContractError` |
| `src/evaluate.py:475-486` `load_predictions()` | `ValueError` |
| `tools/emotion_profile/badcase_pack.py:38-48` `read_jsonl()` | `SystemExit` |
| `tools/emotion_profile/make_val_input.py:30-40` `read_jsonl()` | `SystemExit`（与上一条逐字相同） |
| `tools/emotion_profile/report.py:33-47` `read_jsonl()` | `SystemExit`（同一段，改成一次性返回 list） |

后三条连错误消息都是同一句：`f"{path}:{number}: JSON 解析失败: {exc.msg}"`。

**影响**：同一种故障（「第 N 行 JSON 坏了」）现在有三种异常类。其中 `SystemExit` 最麻烦——
它继承自 `BaseException` 而**不是** `Exception`，调用方写 `except Exception` 接不住，进程直接退出。
在一个工具脚本里这或许是有意的（错误即终止），但它和另外两份实现放在同一个仓库里，
下一个人复制哪一份取决于他打开了哪个文件。

**建议**：`tools/` 下建一个共用的 JSONL 读取模块（或直接复用 `src/data_processor.iter_jsonl`），
`evaluate.py` 与 `tools/emotion_profile/*` 改为 import。这正是 F1 的代价——
`tools/` 没有归属人，就没有人负责说「这份别再抄了」。

**顺带**：这四个新脚本的四个 `main()` 都没有文档字符串（`badcase_pack.py:76`、`make_prompt.py:60`、
`make_val_input.py:43`、`report.py:228`），是同一类遗漏，一个动作可以一起补掉。

## 四、检查过、结果是干净的

写进报告是为了让专家知道**这些项我们真查了**，而不是没查：

- **模块 docstring：27/27，零例外。**
- **中文标识符：0 处。** 中文只出现在注释与文档字符串里，符合仓库既定风格。
- **写死的 Windows 绝对路径：0 处。** 唯一命中是 `tools/fetch_model.py:14` 的用法示例
  （`--dest D:/models/qwen7b`），那是文档，不是代码。
- **写死的容器绝对路径：0 处。** `/root/...` 全部来自官方原件的既定布局或配置项，没有散落在逻辑里。
- **TODO / FIXME / XXX / HACK：0 处。** 没有无主的债务标记。
- **裸 `except:`：0 处。** 唯一的 `except ...: pass` 在 `deploy/participant/run_inference.py:275`，
  位于官方原件内部。

## 五、我们对官方原件做过什么改动（专家一定会问）

`deploy/participant/run_inference.py` 与 `start.sh` 是官方原件的**副本**，我们改过。改动量：

| 文件 | 相对官方原件 |
|---|---|
| `deploy/participant/run_inference.py` | +22 / −2 行 |
| `deploy/participant/start.sh` | +17 / −3 行 |

每一处都写了「为什么改」，且**全部可退回官方基线**：

- `run_inference.py:25-31` 一个 `try: import wiring`。删掉 `wiring.py` 即完全按官方基线跑。
- `run_inference.py:411-427` 官方提示词变量改名 `system_prompt` → `official_prompt`，
  再把 `wiring.compose_system_prompt()` 的结果接上；接线层返回空则原样使用官方提示词。
- `start.sh:5-15` 官方是 `source activate conda_env`，注释写着「名称固定，不要修改」。
  但脚本开头是 `set -euo pipefail`——**这一行失败会直接终止整个脚本，等于零分**。
  改成「有 conda 就进 conda_env，没有就用 PATH 上的 python3」。
- `start.sh:28-31` 官方默认 `--tensor-parallel-size 2`（双卡）。平台算力市场只有 RTX 4090，
  我们按 1 卡申请，双卡并行在单卡上起不来。改为默认 1，可用环境变量覆盖。

**这是有意的、最小的、可回退的改动。** 官方原件的 8 处超长行与 5% 的文档覆盖率**不计入我们的规范性评价**——
改它风险高于收益。

## 六、本报告查不到的地方（如实标注）

静态检查只能证明「有没有写」，不能证明「写得对不对」。以下**未覆盖**：

- 逻辑正确性、边界条件、并发与性能
- 测试是否真能抓住回归（98 个测试全绿，但测试本身的强度未经评估）
- 命名是否达意、注释与代码是否一致
- `docs/` 下的非 Python 文件（markdown 的数字一致性由 `docs/reports/pack_review.py` 另管）

## 七、分派

| 问题 | 位置 | 归属 |
|---|---|---|
| F1 共享目录无主 | `docs/roles/CONTRACT.md:7-18` | **队长裁定** |
| F2 + F6 JSONL 读取四份、三种异常 | 见 §三 F6 的表 | 待定（取决于 F1；F6 已包含 F2） |
| F3 两处 `load_prompt_block` | `rag/fusion.py:453`、`deploy/participant/wiring.py:92` | **⑤A**（加一句注明即可） |
| F4 8 处缺类型标注 | 见 §三 F4 | 待定（取决于 F1） |
| F5 4 个类缺 docstring | 见 §三 F5 | 待定（取决于 F1） |
| F6 附：4 个 `main()` 缺 docstring | `tools/emotion_profile/*.py` | **②**（顺手，一处一行） |

按协作规约，我不改别人的文件。以上清单一经确认，由各线自行修；改完重跑
`python docs/reports/audit_code.py` 即可验证。

> **F1 是这份清单里唯一需要队长拍板的**，其余五条都在等它：`src/`、`tools/`、`tests/` 一旦有了
> 归属人，F2/F4/F5/F6 就都有明确的执行者，否则它们会一直挂在「谁都可以改、也谁都不负责」的区间里。
