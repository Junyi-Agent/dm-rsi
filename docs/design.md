# dm-rsi 设计

> 2026-11-23 修订：流程不再是独立对象，而是 meta loop「行动」阶段执行的内容，由决策逐段选择。本文 §2「流程」、§4「Meta loop」、§7「目录」里关于流程 run、模板固定步骤的描述以 [ADR 0001 的修订](adr/0001-meta-rsi.md) 为准。

dm-rsi 是 DM-RSI-Framework 里 "High-level controller" 方向的实现：决策模型 RSI 的**统一入口**。它负责调度各方向的 sub-RSI 把整条流程跑通，监控结果、诊断问题，在 dm-rsi 和各 sub-RSI 之间用 issue 双向沟通。

它本身是 [xtra-rsi](https://github.com/Junyi-Agent/xtra-rsi) 的一个下游 profile（`dm`），用 xtra-rsi 的五阶段 loop、节点预算和 viz 面板。

## 1. 管什么

| 方向 | 负责人 | 位置 | dm-rsi 怎么调用 |
|---|---|---|---|
| Data | Bingqiao | `subrsi/data`（`BingqiaoL/DM-RSI-Framework@data-generation-0.3`）的 `rsi_skills/data/`，`dgen` | CLI + agent 步骤 |
| Training | CD | 还是占位 | dm-rsi 直接调 onejev-harness（`ojh`），等 Training RSI 出来再换成调它 |
| Output / Decision Policy | Yile | `subrsi/decision-policy`（`Jireh-Chen/DM-RSI-Framework@main`）的 `rsi_skills/decision_policy/`，`dpolicy` | CLI + agent 步骤 |

- `subrsi/*` 是只读 submodule，dm-rsi 不在里面改代码、不提交。要改就提 issue，升级就移动 submodule 指针。
- 两个 submodule 目前是同一个仓库（Data 分支 = Jireh `main` + 2 个 commit）。Data 分支合进上游后，合并成一个 submodule。
- 所有数据、权重、checkpoint、分数、run 输出放 `$DMRSI_HOME`（`/ssd1` 大盘），不进 git。

## 2. 流程（pipeline）

```
build-data ─► data-rsi ─► train ─► merge+check ─► score ─► data-compare ─► dp-rsi ─► final-test
 (CPU)        (CPU+agent)  (GPU)     (CPU)         (GPU)     (CPU)        (CPU+agent)   (人批准)
```

| 步骤 | 内容 | 产物（文件契约） | 后端 | 需要人确认 |
|---|---|---|---|---|
| `build-data` | `pipelines/elliptic_elephant/01_build_data.sh` | `$DMRSI_DATA/.../*.jsonl` | local | 否 |
| `data-rsi` | `dgen check → diagnose`，agent 写配方，`generate`，抽检，`build` | `train.jsonl` + `build_report.json` | local + agent | 抽检由人填；LLM 生成（`external`）要报费用 |
| `train` | `ojh run`（elephant-1 LoRA，2 卡） | `<RUN>/` checkpoint、`RESULT.md` | taskmgr | **每次启动都确认** |
| `merge+check` | `02b_merge_lora.py` + `05_check_merged.py` | 合并后的权重 | local | 否 |
| `score` | `03_score_checkpoint.sh`，加打诊断池 | `$DMRSI_SCORES/<name>_{pool,val,test}.parquet` | taskmgr | 否 |
| `data-compare` | `dgen compare`（val） | compare 报告 | local | 否 |
| `dp-rsi` | `dpolicy check → init`，agent 诊断/提案，`evolve` ×N，3 seed `consensus` | `incumbent.yaml`、`report.md` | local + agent | 否 |
| `final-test` | `dgen final-test` / `dpolicy final-test` | `final_test.json` | local | **是**（每个 run 一次） |

- 步骤之间只通过文件契约交接（`<name>_{pool,val,test}.parquet`：`id, y, p`，可选 `step, post43`），这也是两个 sub-RSI 已有的约定。
- 每个步骤是幂等的：产物已存在且契约校验通过就跳过。sub-RSI 的命令本身拒绝覆盖输出，dm-rsi 用新的 `run_name` / `--tag` 重跑。
- 流程状态写在 `$DMRSI_HOME/runs/<pipeline_run>/state.yaml`：每步的状态、输入输出路径、后端 job id、耗时、花费。

### 两种步骤

- **机械步骤**：直接调 CLI（`dgen generate`、`dpolicy evolve`、`ojh run`）。
- **判断步骤**：sub-RSI 的 SKILL.md 里要 agent 做判断的部分（写诊断、写配方/提案、解读报告）。dm-rsi 在对应 submodule 目录里起一个 Claude Code 会话，提示词 = 该 sub-RSI 的 SKILL.md + 本步骤的目标和限制（预算、轮数、停止条件），并把 usage 记到 dm-rsi 的节点上。agent 只能用 sub-RSI 规定的命令，`final-test` 和 `--again` 被拦截，转成审批请求。

## 3. 执行后端（可插拔）

```python
class Backend(Protocol):
    def submit(self, step: StepSpec) -> JobRef: ...
    def status(self, job: JobRef) -> JobStatus: ...   # queued / running / done / failed / unknown
    def fetch(self, job: JobRef, dest: Path) -> None: ...
    def cancel(self, job: JobRef) -> None: ...
```

| 后端 | 用途 | 第一版 |
|---|---|---|
| `local` | CPU 步骤，本机 subprocess | 实现 |
| `taskmgr` | GPU 步骤：用 [gpu-taskmgr](https://github.com/Junyi-Agent/gpu-taskmgr) 的 `task pack` / `task submit` / `task status` / `task fetch`；`gpus` 用 `{"count": 2, "min_free_gb": 40}` 让服务器挑空闲卡 | 实现（先接 xtrah200，它已在跑 `taskmgr-remote`） |
| `ssh` | 学校 8×A100 服务器 | 留接口 |
| `pbs` | Borealis，用 `pipelines/borealis_elephant/*.pbs.template` | 留接口 |

- xtrah200 的 GPU 看门狗（20 分钟内总 duty < 10% 会被强制登出）：GPU 步骤只放长时间满载的训练和打分，不在 GPU 上跑短进程；dm-rsi 自己的 loop 不占 GPU。
- 共享服务器上每人最多 2 张卡的约定由后端配置里的 `max_gpus` 强制。

## 4. Meta loop（xtra-rsi profile `dm`）

每一轮：

| 阶段 | dm-rsi 里做什么 |
|---|---|
| observe | 流程状态（`state.yaml`）、各 sub-RSI 的产物（`report.md`、`diagnosis.md`、`archive.jsonl`、`VERSIONS.md`）、submodule 上游的新 commit、两边 issue 的状态、待审批队列 |
| reasoning | 诊断：哪一步卡住/失败、指标是否退步、跨方向问题（例如 DP 报"模型本身自信错误" → 属于 Data/Training）、契约是否被破坏 |
| decision | 从固定动作表里选**一件**事：运行/重跑某步、升级某个 submodule 指针、起草 issue、申请审批、结束本次流程 |
| action | 执行；对外动作只生成草稿或审批请求 |
| feedback | 本轮效果：流程推进了几步、指标变化（val 上的 `f1_e2e`、AUPRC 等，不读 test）、issue 有没有被处理；写 `lessons` |

- 用 xtra-rsi 的节点记账：`rsi node init --target dm-pipeline --budget <美元>`，所有 agent 步骤的 LLM 花费都记在节点上，超预算停止。GPU 小时另记在 `state.yaml`。
- meta loop 的目标指标：端到端流程在 val 上的最终指标，加上流程健康度（失败步骤数、等待审批时长）。test 只在 `final-test` 里出现。

## 5. Issue 双向流

| 方向 | 发到哪里 | 怎么发 |
|---|---|---|
| dm-rsi → Decision Policy | `Jireh-Chen/DM-RSI-Framework`，标题前缀 `[decision-policy]` | 草稿 → 人确认 → `gh issue create` |
| dm-rsi → Data | 同上（Bingqiao 的 fork 关了 issues），前缀 `[data]`，正文 @ 负责人 | 同上 |
| dm-rsi → Training | 同上，前缀 `[training]` | 同上 |
| sub-RSI → dm-rsi | `Junyi-Agent/dm-rsi` 的 issues | 他们直接在 GitHub 上提；dm-rsi 的 observe 每轮读取 |

- 草稿写在 `issues/outbox/<id>.yaml`（标题、正文、目标仓库、证据路径、指纹）。`dm-rsi issue list` 查看，`dm-rsi issue send <id>` 发出，`dm-rsi issue drop <id>` 丢弃。
- 正文末尾带 `<!-- dm-rsi:fp=<指纹> -->`：observe 用它把 issue 和 dm-rsi 的诊断对应起来，同一个指纹已有 open 的 issue 时不再起草（去重）。
- 你对 Jireh 仓库只有 READ 权限：能提 issue、能评论，不能打 label、不能指派。分类靠标题前缀。
- `Junyi-Agent/dm-rsi` 提供 issue 模板（问题 / 需求 / 契约变更），方便他们提。

## 6. 审批

统一的审批队列 `$DMRSI_HOME/approvals/`，`dm-rsi pending` 列出，`dm-rsi approve <id>` / `dm-rsi reject <id> --reason ...` 处理。未批准的动作不执行、不花钱。

需要审批的：启动训练、`final-test` 和 `--again`、对外发 issue、付费 LLM 生成、提交到 ssh/PBS 后端。不需要审批的：CPU 步骤、打分、本地 submodule 指针升级（只在 dm-rsi 自己仓库里提交）。

## 7. 目录

```
dm-rsi/
  subrsi/decision-policy/   submodule，只读
  subrsi/data/              submodule，只读
  src/dm_rsi/
    cli.py                  dm-rsi 命令：run / status / pending / approve / reject / issue
    pipeline.py             步骤定义、状态机、契约校验
    steps/                  每个步骤的适配器：build_data / dgen / ojh / score / dpolicy
    backends/               local / taskmgr（ssh、pbs 后续）
    agent.py                判断步骤：在 submodule 里起 Claude Code 会话、记 usage
    issues.py               outbox / send / 读取两边 issue
    approvals.py            审批队列
    profile.py              xtra-rsi profile `dm`（五阶段 = CommandStage → LLM agent）
    viz.py                  xtra-rsi viz 插件：流程泳道、各步骤指标
  harness/<stage>/AGENTS.local.md   meta 层规则（rsi sync 合进 AGENTS.md）
  pipeline.yaml             本次流程的配置：数据集、基座、后端、预算
  docs/                     本文档、ADR
```

## 8. 分阶段实施

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | 仓库骨架、AGENTS.md、`Junyi-Agent/dm-rsi` + issue 模板、issue outbox/send、审批队列、`dm-rsi status` | 能起草并（批准后）发出一个 issue；能读到 dm-rsi 仓库的 issue |
| P1 | `local` 后端 + `dpolicy` / `dgen` 适配器，在**已有的分数文件**上跑 `data-rsi`（到 build）和 `dp-rsi` | 不用 GPU 跑通 DP RSI 一整轮，产物与手动运行一致 |
| P2 | `taskmgr` 后端 + `build-data` / `train` / `merge` / `score`，在 xtrah200 上跑一遍 baseline | 得到 `<name>_{pool,val,test}.parquet`，和 Jireh 已有的 baseline 分数对得上 |
| P3 | meta loop：profile `dm` 五阶段、agent 判断步骤、viz 插件 | `rsi run --profile dm --node root` 从零跑完整条流程，只在审批点停下 |
| P4 | `ssh` / `pbs` 后端；Training RSI 出来后接入 | — |

## 9. 待定

- P1 需要现成的分数文件（`<name>_{pool,val,test}.parquet`）和 Elliptic++ 数据。能否从 Jireh / Bingqiao 那里直接拿到，还是由 P2 在本机重新生成？
- onejev-harness 用哪一份：`subrsi/*/external/onejev-harness`（`BoruiXu/Decision-Model-Study@onejev-harness`，嵌套 submodule），还是本机的 `harness/OneJev`？
- 两个 sub-RSI 的 run 产物现在在各人的服务器上。dm-rsi 要监控**他们**自己跑的结果，需要他们把 run 的 `report.md` / `archive.jsonl` 放到 dm-rsi 能读到的地方（共享目录或推到某个分支）；只监控 dm-rsi 自己调度的 run 则不需要。
