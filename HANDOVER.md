# 交接（2026-11-24）

下一个 session 从这里接着做。设计和决策的来龙去脉在 `docs/adr/0001-meta-rsi.md`（含 2026-11-23、2026-11-24 两次修订），这里只写现状和待办。

## 现状

- 仓库 `Junyi-Agent/dm-rsi`，main 最新提交 `c174cea`（方案 A）。本地路径 `/ssd1/junyi/jev/harness/dm-rsi`，`$DMRSI_HOME = /ssd1/junyi/jev/dmrsi`（目前只有 `approvals/`、`envs/`，没有工作区、没有待审批）。
- 三个 sub-RSI（submodule，只读）：

  | 人 | 方向 | GitHub | submodule | issue 发到 |
  |---|---|---|---|---|
  | Yile | Decision Policy（`dpolicy`） | `Jireh-Chen` | `subrsi/decision-policy` ← `Jireh-Chen/DM-RSI-Framework@main` | `Jireh-Chen/DM-RSI-Framework` |
  | Bingqiao | Data（`dgen`） | `BingqiaoL` | `subrsi/data` ← `BingqiaoL/DM-RSI-Framework@data-generation-0.3` | 同上（fork 关了 issues），前缀 `[data]` |
  | Chendie | Training（`trsi`） | `CD22104` | `subrsi/training` ← `CD22104/decision-model-training@main` | `CD22104/decision-model-training` |

- 合成模板（`DMRSI_TEMPLATE=synthetic`）端到端跑通过：7 段流程，停在 final-test 审批。
- Elliptic 模板（真实流程）**只在假 ojh / 内存 GPU 后端上测过**，从没在真实数据和真卡上跑过。流程：
  `prepare_data → train_rsi → train_base → score_base → data_rsi → retrain → data_compare → dp_rsi → final_test`
- 测试：`uv run pytest -q`（38 个，约 80 秒；`-m "not slow"` 不到 1 秒）。`uv run ruff check src tests`、`uv run basedpyright src` 都是 0 错误。

## 规则（别破坏）

- 审批只能由人批（`uv run dm-rsi pending` 看说明，`dm-rsi approve/reject`）。agent 绝不批准任何审批；`DMRSI_AGENT=1` 时 CLI 也会拒绝。
- 每次启动训练都要审批（含 Training RSI 的每个种子）。审批说明必须自解释（做什么、为什么、批准/拒绝的后果、建议）。
- 不改 `subrsi/` 下被 git 跟踪的文件；sub-RSI 的命令在 submodule 外面跑。
- 提交、push、发 issue 都要用户明确同意。

## 待办

### 1. 准备真实运行环境（用户之前说先不装，留文档；要装时再做）

按 `docs/real-run.md` 的「要准备什么」表逐项准备，文档最后有一段可以直接交给 `claude -p` 的提示词。共 7 项：嵌套 submodule（onejev-harness、crypto-fraud-detection）、harness 的 `setup.sh`、elephant-1 基座、Elliptic++ 数据、数据 venv（`$DMRSI_HOME/jev-venv`）、gpu-taskmgr 的 `task` CLI、trsi 能在本机 `ojh run`（`ojh doctor`、`ojh fetch-base elephant-1`）。

注意 xtrah200 的显卡看门狗（20 分钟内 GPU duty < 10% 会被强制登出）：准备阶段不要在 GPU 上跑短进程。

### 2. 第一次真实运行

```bash
DMRSI_TEMPLATE=elliptic DMRSI_WORKSPACE=ell-1 uv run rsi run --profile dm --iterations 20 --run-dir $DMRSI_HOME/loops/ell-1
uv run dm-rsi pending
```

- 建完数据后会停在 Training RSI 起点种子 0 的训练审批。批之前 `nvidia-smi` 看卡：`train_gpu_ids` 默认 `[0, 1]`，2026-11-24 查看时 0 号和 3 号卡被占用，可能要改参数（决策可以给 `train_rsi(train_gpu_ids=...)`）。
- 起点 3 个种子在 H200 上各约 3–4 小时；Training RSI 预算默认 100 卡时，最多 4 个候选。

### 3. 第一次真实运行时要盯的未验证点

- trsi 调 `ojh run` 时没有传 `--machine`：harness 默认机器配置是否存在、`out_root` 在哪，需要确认；需要时给 `train_rsi` 设参数 `ojh_machine`（`steps/trainrsi.py` 的 `write_configs`）。
- `trsi check --ojh` 第一次在真实环境里跑能否通过。
- `train_base` / `retrain` 现在把 Training RSI 方案的 overrides（`{key: {value, reason}}`）写进 harness 的 exp.yaml，经 `ojh prepare` / `ojh resolve`：harness 是否接受这种写法、要不要额外的 override 审批。
- 去掉 `fields.text` 之后，`03_score_checkpoint.sh` 和 `04_policy_features.py` 是否还依赖 `state_json` 文本字段。
- `dgen check` 在真实行上能否通过（行格式差异见 ADR「实施中发现的契约差异」）。
- LoRA 合并（`02b_merge_lora.py`）和打分作业在 taskmgr 上的实际表现。

### 4. 已知的设计缺口（可以先做，不依赖环境）

- **agent 写的候选超出 trsi 的 space**：`trsi launch` 拒绝 → 整段 `train_rsi` 失败 → 会起草发给 Training 的 issue，但这是 dm-rsi 的 agent 写错了，不是 Chendie 的问题。应该把被拒的候选记下来跳过（或让 agent 重写），不让整段失败。
- **trsi 的训练失败**：现在任何一个试验 failed 整段就失败，只能 `dm-rsi work reset` 后重来。trsi 支持用同一条命令重跑失败的试验，可以加一个"重试失败试验"的审批。
- **方案变了，基线不会重训**：`flows.suggest` 只看 `train_scheme` 是否存在。如果之后重跑 `train_rsi` 选出新方案，已有的 `base_model` / `models` 不会被判为过时。
- **trsi 不经 taskmgr 调卡**：同一台机器上 taskmgr 的作业和 trsi 的 `ojh run` 可能抢卡（harness 会检查卡是否空闲，但两边互相看不到预约）。
- **`tests/` 下有既有的 pyright 报错**（146 个，主要是测试里的假对象类型；`src/` 是 0）。可以清理，或者把 basedpyright 的 `include` 限定到 `src`。

### 5. 可以和各方向沟通的（都要先进审批队列，用户确认后才发）

- Chendie：trsi 的交接现在靠 `report.md` 的 Markdown 表；dm-rsi 直接读 `incumbent.yaml` 和 `trials/*.json`，够用，但一个机器可读的交接文件（例如 `handoff.yaml`）会更稳。不急。
- Bingqiao / Yile：ADR 里记的契约差异。dgen 的合成 val 只有 1 个 step，dpolicy 默认按 step 划分会失败；Data 用 `addr_json_noul`（`train_sub40k` + pool），Decision Policy 和 Training 用 `addr_txhist_noul`（完整训练集）。
- 起草：`uv run dm-rsi issue draft --to <方向> --key <诊断键> --title ... --body ...`，进审批队列，人 `dm-rsi approve <id>` 才发。

## 常用

```bash
uv run dm-rsi status            # 各方向、submodule、工作区
uv run dm-rsi pending           # 待审批（带说明）
uv run dm-rsi work list|show <id>|reset <id> <step>
uv run rsi viz ...              # xtra-rsi 面板（dm-rsi 提供了 viz 插件）
uv run rsi sync                 # 改了 harness/*/AGENTS.local.md 后重新生成 AGENTS.md
```
