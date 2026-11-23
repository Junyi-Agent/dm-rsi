# dm-rsi

DM-RSI-Framework 的 meta RSI：统一入口，调度各方向的 sub-RSI（Data、Training、Decision Policy），监控结果、诊断问题，用 issue 和各方向双向沟通。设计和决策见 `docs/adr/0001-meta-rsi.md`、`docs/design.md`。

## 布局

- `subrsi/decision-policy`、`subrsi/data`：sub-RSI 的 submodule，**只读**。不在里面改文件、不提交、不 `pip install -e`。升级 = 移动 submodule 指针。
- `src/dm_rsi/`：`config.py`（`dm-rsi.yaml`）、`approvals.py`（审批队列）、`issues.py` + `github.py`（issue 双向流）、`pipeline.py`（流程：步骤、状态、advance）、`steps/`（各步骤：`synthetic`、`dgen`、`dpolicy`、`training`（Elliptic 训练/打分）、`final`、`judge`）、`backends/`（`local`、`taskmgr`、`ssh`、`pbs`）、`agent.py`（判断步骤的 claude 会话）、`profile.py`（xtra-rsi profile `dm`，meta loop）、`viz.py`、`cli.py`。
- `harness/<stage>/`：meta loop 五个阶段的规则。只改 `AGENTS.local.md`，然后 `uv run rsi sync`。
- `.claude/jy-skills`：Junyi-Agent/skills（submodule），`.claude/skills/jy-*` 是软链接。
- 运行产物、数据、分数、审批队列都在 `$DMRSI_HOME`（默认 `/ssd1/junyi/jev/dmrsi`），不进 git。

## 命令

```bash
uv sync
uv run pytest -q                 # 含 slow：用 sub-RSI 真实 CLI 跑合成流程（约 2 分钟）
uv run pytest -q -m "not slow"
uv run ruff format && uv run ruff check && uv run basedpyright src
```

## 规则

- **审批只由人做。** `dm-rsi approve` / `reject` 由人在自己的终端运行；agent 不批准任何请求，包括合成数据上的。dm-rsi 启动的 agent 带 `DMRSI_AGENT=1`，`approve` 会拒绝。
- 每个审批都要带 `explain`（`StepContext.gate` 的必填参数）：是什么、为什么要人确认、上下文（例如 val 结果）、批准 / 拒绝之后会怎样、建议。人只看 `dm-rsi pending` 就要能做决定。
- 需要审批：每次启动训练、`final-test`（及 `--again`）、对外发 issue、付费 LLM 生成数据、提交到 ssh / pbs 后端、Data RSI 的人工抽检完成确认。
- 不读 test：只有 `final_test` 步骤（审批后）读 test。
- 给 sub-RSI 的问题用 `dm-rsi issue draft` 起草（进审批队列），不要直接 `gh issue create`。
- sub-RSI 的产物通过文件契约交接（`steps/contracts.py`）；契约不符时报错并转成 issue，不在 dm-rsi 里悄悄修补别人的数据。
- 架构或流程决策写 ADR（`docs/adr/`）；agent 最多把状态改到「采纳」。
