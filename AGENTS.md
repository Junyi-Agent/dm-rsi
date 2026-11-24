# dm-rsi

DM-RSI-Framework 的 meta RSI：统一入口，调度各方向的 sub-RSI（Data、Training、Decision Policy），监控结果、诊断问题，用 issue 和各方向双向沟通。设计和决策见 `docs/adr/0001-meta-rsi.md`、`docs/design.md`。

## 布局

- `subrsi/decision-policy`、`subrsi/data`：sub-RSI 的 submodule，**只读**。不在里面改文件、不提交、不 `pip install -e`。升级 = 移动 submodule 指针。
- `src/dm_rsi/`：`config.py`（`dm-rsi.yaml`）、`approvals.py`（审批队列）、`issues.py` + `github.py`（issue 双向流）、`workspace.py`（工作区：行动阶段执行过的流程段、产物、审批）、`flows.py`（流程动作表、参数白名单、默认决策规则）、`steps/`（各流程的实现：`synthetic`、`dgen`、`dpolicy`、`trainrsi`（Training RSI：选训练方案）、`training`（Elliptic 训练/打分，按选出的方案）、`final`、`judge`）、`backends/`（`local`、`taskmgr`、`ssh`、`pbs`）、`versions.py`（版本登记、升级规则、停止条件）、`agent.py`（判断步骤的 claude 会话）、`profile.py`（xtra-rsi profile `dm`，meta loop：观察、推理、决策由 dm-rsi 做，行动执行流程）、`viz.py`、`cli.py`。
- `harness/<stage>/`：meta loop 五个阶段的规则。只改 `AGENTS.local.md`，然后 `uv run rsi sync`。
- `.claude/jy-skills`：Junyi-Agent/skills（submodule），`.claude/skills/jy-*` 是软链接。
- 运行产物、数据、分数、审批队列都在 `$DMRSI_HOME`（默认 `/ssd1/junyi/jev/dmrsi`），不进 git。

## 命令

```bash
uv sync
uv run pytest -q                 # 含 slow：用 sub-RSI 真实 CLI 跑合成流程（约 2 分钟）
uv run pytest -q -m "not slow"
uv run ruff format src tests && uv run ruff check src tests && uv run basedpyright   # CI（.github/workflows/ci.yml）跑同样的检查和 -m "not slow"
```

## 规则

- 流程只在 meta loop 的行动阶段执行（`rsi run --profile dm`），没有绕过 loop 的入口。新流程加到 `flows.py` 的动作表并给出允许的参数。

- **审批只由人做。** `dm-rsi approve` / `reject` 由人在自己的终端运行；agent 不批准任何请求，包括合成数据上的。dm-rsi 启动的 agent 带 `DMRSI_AGENT=1`，`approve` 会拒绝。
- 每个审批都要带 `explain`（`StepContext.gate` 的必填参数）：是什么、为什么要人确认、上下文（例如 val 结果）、批准 / 拒绝之后会怎样、建议。人只看 `dm-rsi pending` 就要能做决定。
- 需要审批：每次启动训练、`final-test`（及 `--again`）、对外发 issue（只有人手动起草的）、付费 LLM 生成数据、提交到 ssh / pbs 后端、Data RSI 的人工抽检完成确认。
- 不读 test：只有 `final_test` 步骤（审批后）读 test。
- **dm-rsi 和它启动的 agent 不给各方向提 issue、不 review、不改他们的代码**（用户 2026-11-24 决定）：只负责 dm-rsi 自己。流程失败时 loop 停下等人。`dm-rsi issue draft` 只留给人手动用。
- sub-RSI 的产物通过文件契约交接：**只在 `steps/handoff.py` 里读 sub-RSI 写的文件**（格式见 `docs/handoff.md`，`tests/test_handoff.py` 会检查），分数文件的校验和重算在 `steps/contracts.py`；契约不符时报错、流程失败、loop 停下等人，不在 dm-rsi 里悄悄修补别人的数据。
- 架构或流程决策写 ADR（`docs/adr/`）；agent 最多把状态改到「采纳」。
