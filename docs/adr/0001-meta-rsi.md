# 0001 dm-rsi 作为 meta RSI：统一入口，调度 sub-RSI，分五个阶段实施

状态：采纳（2026-11-23 用户授权按五个阶段实施）

## 背景

DM-RSI-Framework 按方向拆成多个 sub-RSI：Data（Bingqiao，`dgen`）、Training（Chendie，上游 README 里写作 CD，`trsi`）、Output / Decision Policy（Yile，`dpolicy`）。各方向自成一体，只通过文件契约交接，但没有人负责把它们串成一条流程、看端到端的结果、把一个方向发现的问题转给另一个方向。README 里 "High-level controller" 一行的位置是「待定」。

约束：

- sub-RSI 的代码归各自负责人，dm-rsi 只能读，不能改。
- 我们对上游仓库只有 READ 权限；Data 的 fork 关了 issues。
- 训练要 GPU（elephant-1 LoRA 约 2.4 GPU 小时一组），机器不止一台：本机 xtrah200（有 GPU 看门狗）、学校 A100 服务器、Borealis（PBS）。
- 两个 sub-RSI 已经规定了必须由人批准的动作：`final-test`、`--again`、提交 GPU 作业、付费 LLM 生成。

## 决策

### 定位

- dm-rsi 是整条流程的**入口**：从建数据到 final-test，由 dm-rsi 调度各 sub-RSI 实际运行，不只是监控。
- dm-rsi 是 xtra-rsi 的下游 profile `dm`，用它的五阶段 loop、节点预算、viz。

### sub-RSI 以 submodule 接入，只读

- `subrsi/decision-policy` = `Jireh-Chen/DM-RSI-Framework@main`，`subrsi/data` = `BingqiaoL/DM-RSI-Framework@data-generation-0.3`。
- dm-rsi 不在 submodule 里改文件、不提交。升级 = 移动 submodule 指针（dm-rsi 仓库内的提交，不需要审批）。
- 嵌套的 `external/` submodule 不默认初始化，用到 onejev-harness 时再初始化。

### 流程由步骤组成，步骤之间只走文件契约

- 步骤：`build-data → data-rsi → train → merge → score → data-compare → dp-rsi → final-test`。
- 契约沿用 sub-RSI 已有的：`<name>_{pool,val,test}.parquet`（`id, y, p`，可选 `step, post43`）、`dgen build` 的 `train.jsonl` + `build_report.json`、`dpolicy` 的 `incumbent.yaml`。dm-rsi 在步骤结束时校验契约。
- 步骤分两种：机械步骤直接调 sub-RSI 的 CLI；判断步骤（写诊断、配方、提案）在对应 submodule 目录里起 agent 会话，提示词是该 sub-RSI 的 SKILL.md 加本步骤的目标和限制。
- （已修订，见下文「修订」：改为工作区 `$DMRSI_HOME/work/<id>/`）流程状态写 `$DMRSI_HOME/runs/<run>/state.yaml`。所有数据、权重、分数、run 产物放 `$DMRSI_HOME`，不进 git。

### 执行后端可插拔

- 统一接口 `submit / status / fetch / cancel`。
- 第一版实现 `local`（CPU，subprocess）和 `taskmgr`（GPU，[gpu-taskmgr](https://github.com/Junyi-Agent/gpu-taskmgr) 的 `task` CLI）；`ssh`、`pbs` 留接口。
- 后端配置里有 `max_gpus`（共享服务器每人 ≤ 2 卡）。GPU 上只放长时间满载的训练和打分，避免触发 xtrah200 的看门狗。

### issue 双向流

- dm-rsi → sub-RSI：发到 `Jireh-Chen/DM-RSI-Framework`，标题前缀 `[decision-policy]` / `[data]` / `[training]`。草稿就是审批队列里的一个 `issue` 类请求，人运行 `dm-rsi approve <id>` 才发出（实施时从单独的 outbox 改成复用审批队列，只有一个入口）。
- sub-RSI → dm-rsi：在 `Junyi-Agent/dm-rsi` 提 issue，dm-rsi 每轮读取。
- 正文末尾带 `<!-- dm-rsi:fp=<指纹> -->`，用于去重和把 issue 对应到 dm-rsi 的诊断。同一指纹已有 open issue 时不再起草。

### 统一的审批队列

- `dm-rsi pending` / `approve <id>` / `reject <id> --reason`。未批准的动作不执行、不花钱。
- 需要审批：每次启动训练、`final-test` 及 `--again`、对外发 issue、付费 LLM 生成、提交到 ssh/PBS 后端（训练的审批里写明了后端，算同一次确认）。
- Data RSI 的人工抽检也走这个队列（`spotcheck` 类）：人填完 `spotcheck.csv` 后 `approve`，表示「人工步骤完成」。
- dm-rsi 启动的 agent 带 `DMRSI_AGENT=1`，`approve` / `reject` 在这种环境里一律拒绝。
- 不需要审批：CPU 步骤、打分、submodule 指针升级。

### 分五个阶段实施

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | 仓库骨架、jy-skills、issue outbox/send/inbox、审批队列、`dm-rsi status` | 单元测试覆盖草稿、去重、审批；`gh` 调用可替换为假实现 |
| P1 | `local` 后端；`dgen`、`dpolicy` 适配器；`data-rsi`、`dp-rsi` 的机械部分 | 用两个 sub-RSI 自带的 `demo_synthetic` 配置，不用 GPU 跑通，产物与手动运行一致 |
| P2 | `taskmgr` 后端；`build-data`、`train`、`merge`、`score` 适配器 | 假 `task` CLI 下的测试通过；真实训练等审批后在 xtrah200 上跑 |
| P3 | xtra-rsi profile `dm`、判断步骤的 agent 会话、viz 插件 | `rsi run --profile dm --node root` 在合成数据上从零跑完，只在审批点停下 |
| P4 | `ssh`、`pbs` 后端；Training RSI 出来后接入 | — |

详细设计见 [`docs/design.md`](../design.md)。

### 实施记录（2026-11-23）

| 阶段 | 状态 | 验证 |
|---|---|---|
| P0 | 已实现 | `tests/test_approvals_issues.py`：草稿、去重、审批、agent 不能审批、发送失败记录 |
| P1 | 已实现 | `tests/test_synthetic_e2e.py`：合成数据上 `make-data → … → final-test`，submodule 不被写入 |
| P2 | 已实现，**未在真实数据上跑过** | `tests/test_training.py`（假 ojh、内存 GPU 后端）；真实运行要 Elliptic++ 数据、harness 环境和训练审批 |
| P3 | 已实现 | `tests/test_meta.py`：`run_loop(profile dm)` 从零跑完合成流程并停在 final-test 审批；失败步骤只起草一次 issue；agent 判断会话（假 claude）写配方并记账 |
| P4 | ssh / pbs 后端已实现；Training RSI 待其出现后接入 | `tests/test_remote_backends.py`（本机执行的假 ssh、假 qsub/qstat） |

实施中发现的契约差异（要和各方向确认）：

- dgen 的合成数据 val 只有 1 个 step，dpolicy 默认 `by: step` 协议需要多个 step；合成模板改用 `by: group`。
- Decision Policy 的流程用 `addr_txhist_noul` 行、在全部 train 上训练；Data RSI Stage 1 用 `addr_json_noul` 行、在 `train_sub40k` 上训练，pool = 训练期里其余的行。`elliptic` 模板按 Data 的做法从 train 里切 `train_sub` + `pool`，schema 默认 `elliptic_addr.yaml`，是否适用于 txhist 行要 `dgen check` 在真实数据上确认。

### 修订（2026-11-23）：流程放进「行动」阶段

用户确认的分工：**观察、推理、决策是 dm-rsi 自己的；行动执行具体的流程**（数据、训练、打分……）。原设计里流程是一个和 meta loop 平级的独立对象（按模板生成固定步骤列表，`dm-rsi run advance` 可以绕过 loop 直接推进），面板上也看不到。改为：

- **工作区**（`$DMRSI_HOME/work/<id>/`）代替「流程 run」：只记录行动阶段执行过的流程段和产物，不再有固定的步骤列表。流程段之间不声明依赖，要什么产物就在工作区里找最新完成的那一份。
- **决策选下一段流程**（`dm_rsi.flows` 的动作表：`prepare_data / train_base / score_base / data_rsi / retrain / data_compare / dp_rsi / final_test`，带参数），另有 `resume`（继续没结束的执行）、`draft_issue`、`wait`。默认规则按产物判断缺什么、什么过时；LLM 决策可以不按默认顺序（重跑、换配方、多跑几轮），但流程名和参数只能取动作表里的。同一时间只执行一段流程。
- **行动执行这段流程**，执行过程（调用的 sub-RSI 命令、日志、作业）写进本轮的 `action.yaml` 的 `events`，viz 插件把它作为行动阶段的内部过程显示。
- 去掉 `dm-rsi run new/advance`：流程只经由 meta loop 执行。`dm-rsi work list/show/reset` 只用来查看和重置失败的执行。
- 同一模型重复执行 `dp_rsi` 是在同一个 dpolicy run 上多跑几轮；对比选出的模型变了才开新的 dpolicy run。

## 后果

- 好处：一条命令跑完整条流程；跨方向的问题有固定去向（issue）；所有对外和花钱的动作都经过同一个审批队列。
- 代价：dm-rsi 依赖 sub-RSI 的 CLI 和文件契约，它们改了 dm-rsi 要跟着改。用 submodule 固定版本，升级时跑适配器测试。
- 两个 submodule 目前是同一个仓库的两个分支，内容一半重复。Data 分支合进上游后合并成一个。
- 已知缺口：
  - 监控各负责人**自己**跑的 run 需要他们把产物放到 dm-rsi 能读到的地方，第一版只监控 dm-rsi 调度的 run。
  - 真实数据（Elliptic++）和分数文件本机还没有，P1、P3 先用合成数据验收。
  - onejev-harness 用嵌套 submodule 那份还是本机 `harness/OneJev`，P2 实施时定。

## 验证

（只由人类填写。状态改成「已验证」时填：验证人、日期、测了什么、结果。）

## 修订（2026-11-24）：接入 Training RSI

Chendie 按 Yile 的写法发布了 Training RSI：`CD22104/decision-model-training`，包 `rsi_skills/training`，命令 `python -m trsi`。它在卡时预算内选择基座、全参或 LoRA、超参，训练一律经 `ojh run`。

- 加为第三个 submodule `subrsi/training`（`main`，`f50c3b2`），`dm-rsi.yaml` 的 `training` 方向补全 `github_login: CD22104`、`module: trsi`。
- 它是独立仓库、开着 issues，所以方向配置新增 `issue_repo`：给 Training 的 issue 发到 `CD22104/decision-model-training`，其余方向仍发到 `github.upstream`。去重和跟踪（`outbound`）按各方向的仓库查。
- 流程里暂时还没有调 `trsi`，`elliptic.train` / `elliptic.retrain` 仍由 dm-rsi 直接调 onejev-harness。换成调 `trsi` 的位置待定。

接入时发现的差异：

| 差异 | Training RSI | dm-rsi 现在的 elliptic 模板 |
|---|---|---|
| 训练配方 | 起点 `elephant_lora_1g.yaml`：1 卡，lr 1e-4，每步 64 行（团队实际用过的设置） | `pipelines/elliptic_elephant/recipes/elephant_lora_ddp_2g.yaml`：2 卡 DDP，lr 1e-5，每步 32 行，文件里写明"超参没调" |
| 训练集 | 完整 `addr_txhist_noul/train.jsonl`（和 Decision Policy 一致） | Data 方向的切法：`train_sub40k` + `pool` |
| 选 checkpoint 的评测集 | 完整 val（24,225 个地址） | harness 默认 |
| 启动训练 | `trsi launch` 直接在本机 `ojh run --gpus <ids>`（tmux，立即返回）；用户一次性批准 `space` 后范围内的候选不再逐个问 | 经可插拔后端（taskmgr 等）；每次启动训练都要人在 `dm-rsi pending` 里批准 |
| 交接 | `report.md` 第一节（Markdown 表）列当前方案每个种子的 run 目录和 checkpoint；机器可读的是 `incumbent.yaml` 和 `trials/*.json` | 流程产物（`base_model`、`models`） |
