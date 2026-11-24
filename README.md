# dm-rsi

DM-RSI-Framework 的 meta RSI（"High-level controller"）：整条流程的统一入口。调度各方向的 sub-RSI 跑完从数据到 final-test 的流程，监控结果、诊断问题，给各方向起草 issue，也接收各方向提给 dm-rsi 的 issue。

```
build-data → data-rsi → train → score → data-compare → dp-rsi → final-test
  (CPU)     (dgen,CPU)  (GPU)   (GPU)     (dgen)      (dpolicy)  (人批准)
```

| 方向 | sub-RSI | 在 dm-rsi 里 |
|---|---|---|
| Data（Bingqiao） | `subrsi/data`（`BingqiaoL/DM-RSI-Framework@data-generation-0.3`，`dgen`） | `dgen.rsi`、`dgen.compare` |
| Training（Chendie，`CD22104`） | `subrsi/training` ← `CD22104/decision-model-training@main`（`trsi`） | 流程 `train_rsi`（Elliptic 模板）：trsi 在卡时预算内选基座、配方和超参；之后的 `train_base`、`retrain` 都按选出的方案训练 |
| Decision Policy（Yile） | `subrsi/decision-policy`（`Jireh-Chen/DM-RSI-Framework@main`，`dpolicy`） | `dpolicy.rsi` |

## 怎么工作

dm-rsi 是一个 xtra-rsi meta loop（profile `dm`）。**观察、推理、决策由 dm-rsi 自己做；行动执行决策选中的那一段具体流程**：

```
observe   : 工作区里已有的产物、每段流程执行的状态、待审批、submodule、issue
reasoning : 缺什么、什么过时了、哪里失败了、归哪个方向
decision  : 下一段流程 + 参数（prepare_data / train_rsi / train_base / score_base / data_rsi / retrain /
            data_compare / dp_rsi / final_test），或 resume / draft_issue / wait
action    : 执行这段流程（调 dgen / dpolicy / onejev-harness、提交 GPU 作业、登记审批）
feedback  : 结果和 lessons；只剩人要做的事时停下
```

## 开始

```bash
git clone --recurse-submodules https://github.com/Junyi-Agent/dm-rsi && cd dm-rsi
uv sync
uv run dm-rsi status

# 合成数据（全程 CPU，约 1 分钟）：从零跑到 final-test 的审批停下
DMRSI_WORKSPACE=syn uv run rsi run --profile dm --iterations 20 --run-dir $DMRSI_HOME/loops/syn-1
uv run dm-rsi pending              # 每个审批都有说明：是什么、val 结果、批准 / 拒绝之后会怎样
uv run dm-rsi approve a0001
DMRSI_WORKSPACE=syn uv run rsi run --profile dm --iterations 5 --run-dir $DMRSI_HOME/loops/syn-2

uv run rsi viz $DMRSI_HOME/loops   # 面板：展开「行动」看每段流程调用了什么
uv run dm-rsi work show syn        # 工作区里每段流程的执行和产物
```

真实流程（Elliptic++ × elephant-1）：`DMRSI_TEMPLATE=elliptic DMRSI_WORKSPACE=ell-1 rsi run --profile dm ...`。训练和打分走 `dm-rsi.yaml` 里的 GPU 后端（默认 `xtrah200`，gpu-taskmgr），每次训练都要审批。本机环境还没装，准备方法（以及交给 `claude -p` 去准备的提示词）见 [`docs/real-run.md`](docs/real-run.md)。

## issue

- dm-rsi → 各方向：`dm-rsi issue draft --to data --key <诊断键> --title ... --body ...` 进审批队列，`dm-rsi approve <id>` 后发到这个方向收 issue 的仓库（`dm-rsi.yaml` 的 `issue_repo`，不填就是 `Jireh-Chen/DM-RSI-Framework`；Training 发到 `CD22104/decision-model-training`），标题带 `[data]` / `[decision-policy]` / `[training]`。
- 各方向 → dm-rsi：在本仓库提 issue（问题 / 需求 / 契约变更模板）。`dm-rsi issue list` 和 meta loop 的 observe 会读到。

## 文档

- [`docs/adr/0001-meta-rsi.md`](docs/adr/0001-meta-rsi.md)：定位、流程、后端、issue、审批、分阶段实施
- [`docs/design.md`](docs/design.md)：设计细节
- [`docs/real-run.md`](docs/real-run.md)：真实流程的环境准备
- [`AGENTS.md`](AGENTS.md)：在本仓库工作的 agent 须知
