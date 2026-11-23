# dm-rsi

DM-RSI-Framework 的 meta RSI（"High-level controller"）：整条流程的统一入口。调度各方向的 sub-RSI 跑完从数据到 final-test 的流程，监控结果、诊断问题，给各方向起草 issue，也接收各方向提给 dm-rsi 的 issue。

```
build-data → data-rsi → train → score → data-compare → dp-rsi → final-test
  (CPU)     (dgen,CPU)  (GPU)   (GPU)     (dgen)      (dpolicy)  (人批准)
```

| 方向 | sub-RSI | 在 dm-rsi 里 |
|---|---|---|
| Data（Bingqiao） | `subrsi/data`（`BingqiaoL/DM-RSI-Framework@data-generation-0.3`，`dgen`） | `dgen.rsi`、`dgen.compare` |
| Training（CD） | 还没有 | dm-rsi 直接调 onejev-harness：`elliptic.train`、`elliptic.score` |
| Decision Policy（Yile） | `subrsi/decision-policy`（`Jireh-Chen/DM-RSI-Framework@main`，`dpolicy`） | `dpolicy.rsi` |

## 开始

```bash
git clone --recurse-submodules https://github.com/Junyi-Agent/dm-rsi && cd dm-rsi
uv sync
uv run dm-rsi status

# 合成数据上跑一遍（全程 CPU，约 1 分钟），停在 final-test 的审批
uv run dm-rsi run new --template synthetic --id syn-1
uv run dm-rsi run advance syn-1
uv run dm-rsi pending
uv run dm-rsi approve a0001        # 人确认
uv run dm-rsi run advance syn-1

# 或者交给 meta loop：自己新建流程、推进、把失败转成 issue 草稿，停在审批点
uv run rsi run --profile dm --iterations 20 --run-dir $DMRSI_HOME/loops/l1
uv run rsi viz $DMRSI_HOME/loops
```

真实流程（Elliptic++ × elephant-1）：`dm-rsi run new --template elliptic`。本机环境还没装，准备方法（以及交给 `claude -p` 去准备的提示词）见 [`docs/real-run.md`](docs/real-run.md)。训练和打分走 `dm-rsi.yaml` 里的 GPU 后端（默认 `xtrah200`，gpu-taskmgr），每次训练都要审批。

## issue

- dm-rsi → 各方向：`dm-rsi issue draft --to data --key <诊断键> --title ... --body ...` 进审批队列，`dm-rsi approve <id>` 后发到 `Jireh-Chen/DM-RSI-Framework`，标题带 `[data]` / `[decision-policy]` / `[training]`。
- 各方向 → dm-rsi：在本仓库提 issue（问题 / 需求 / 契约变更模板）。`dm-rsi issue list` 和 meta loop 的 observe 会读到。

## 文档

- [`docs/adr/0001-meta-rsi.md`](docs/adr/0001-meta-rsi.md)：定位、流程、后端、issue、审批、分阶段实施
- [`docs/design.md`](docs/design.md)：设计细节
- [`docs/real-run.md`](docs/real-run.md)：真实流程的环境准备
- [`AGENTS.md`](AGENTS.md)：在本仓库工作的 agent 须知
