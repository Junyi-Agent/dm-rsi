# 真实流程（Elliptic++ × elephant-1）的环境准备

dm-rsi 的 `elliptic` 模板已经写好（`src/dm_rsi/steps/training.py`），但本机的运行环境**还没装**。这里列出要准备的东西和检查方法。准备工作建议交给一个 `claude -p` 会话去做（见最后一节）。

## 要准备什么

| # | 什么 | 在哪 | 怎么检查 | 谁用 |
|---|---|---|---|---|
| 1 | 嵌套 submodule：onejev-harness、crypto-fraud-detection | `subrsi/decision-policy/external/` | `git -C subrsi/decision-policy submodule status` 两行都没有 `-` 前缀 | 全部 |
| 2 | onejev-harness 的环境（`setup.sh`） | `subrsi/decision-policy/external/onejev-harness/.venv`、`vendor/OneJev`（在 submodule 里，但被它自己的 `.gitignore` 忽略） | `.venv/bin/ojh doctor` 全部 ok | `elliptic.train`、`elliptic.score` |
| 3 | elephant-1 基座 | `/ssd1/junyi/jev/models/elephant-1`（本机已有） | 目录里有 `model.safetensors` | `elliptic.train` |
| 4 | Elliptic++ 原始数据和 OneJev 行 | `$DMRSI_HOME/elliptic/data/onejev/addr_txhist_noul/*.jsonl` | `01_build_data.sh` 结束时打印各文件行数；按 `DATA_PIPELINE.md` 第 5 节核对 split id | `elliptic.build_data` |
| 5 | 数据环境（Python 3.11 + `requirements_jev.txt`） | `$DMRSI_HOME/jev-venv` | `$DMRSI_HOME/jev-venv/bin/python -c "import pandas"` | `01_build_data.sh`、`04_policy_features.py` |
| 6 | GPU 后端：gpu-taskmgr 的 `task` CLI，服务器 `xtrah200` 已登记 | `~/.local/bin/task`、`~/.taskmgr/machines.json` | `task host ping xtrah200` 有应答 | `elliptic.train`、`elliptic.score` |

`elliptic.build_data` 会在 1 缺失时停下并提示；2、4、5 缺失时它会尝试自己跑 `setup.sh` 和 `01_build_data.sh`。6 没有时，提交训练会报 `task` 找不到。

环境变量不用手动 source：dm-rsi 按 `dm-rsi.yaml` 的 `home` 拼出 `env.example.sh` 里的全部变量（`steps/training.py` 的 `script_env`）。

## 注意

- xtrah200 有 GPU 看门狗：一个用户所有 GPU 进程 20 分钟内的总 duty 低于 10% 会被强制登出。环境准备阶段不要在 GPU 上跑短进程（`ojh doctor` 只查环境，不占卡）。
- 训练前 dm-rsi 会登记 `train` 审批，每次都要人在 `dm-rsi pending` 里确认。
- 两个方向的行格式不一致（ADR 0001「实施中发现的契约差异」）：`dgen check` 在真实数据上第一次运行时要看它是否通过。

## 交给 agent 准备

在 dm-rsi 根目录运行（会安装软件包、下载数据，约几十 GB；按需加 `--max-budget-usd`）：

```bash
claude -p --permission-mode acceptEdits "$(cat <<'EOF'
读 docs/real-run.md，按「要准备什么」表逐项准备 dm-rsi 真实流程的运行环境。
- 每一项先检查，已经满足的跳过。
- 不要修改 subrsi/ 下任何被 git 跟踪的文件；setup.sh 产生的 .venv、vendor/ 被它自己的 .gitignore 忽略，可以。
- 不要启动训练、不要在 GPU 上跑任何东西、不要批准任何 dm-rsi 审批。
- 数据下载失败（例如 Google Drive 限流）时停下，报告原因。
- 最后输出每一项的检查结果（命令和输出），以及还缺什么。
EOF
)"
```

准备好之后：

```bash
uv run dm-rsi run new --template elliptic --id ell-1
uv run dm-rsi run advance ell-1     # 建数据 → 停在第一次训练的审批
uv run dm-rsi pending
```
