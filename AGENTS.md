# AGENTS.md

## 先读这个（如果你是被队友的提示词拉进来的编码 AI）

本仓库是「2026 动感地带 AI+ 高校创智计划」AI 技术赛道赛题二数字人综合情感陪伴对话模型的参赛工程，由五个人并行开发，按六条线分工（有人兼两条，见 `ROLES.md`）。

- 分工总表与用法：`ROLES.md`
- 公共背景（赛题 / 官方评分细则 / 硬约束 / 命令 / 铁律）：`docs/roles/README.md`
- 各线之间的接口契约：`docs/roles/CONTRACT.md`
- 权威计划：`PLAN.md`

**只动你自己那条线的目录**，不要改别人的文件；`GivenData/` 是官方材料，永远只读。

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues in `SaoZhengOvO/OGC`, driven with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles map to labels of the same names: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

single-context — one `CONTEXT.md` at the repo root plus architecture decisions in `docs/adr/`. See `docs/agents/domain.md`.
