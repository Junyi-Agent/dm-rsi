## dm-rsi：meta RSI 的 action

行动阶段由代码执行，不由 LLM 执行：按 `plan` 在工作区（`$DMRSI_HOME/work/<workspace>/`）里新建或推进一次流程执行，调用 sub-RSI 的命令、提交 GPU 作业、登记审批。输出 `result`（`step`、`flow`、`state`、`note`）和 `events`（这次执行调用的命令和日志路径，`rsi viz` 展开行动阶段时显示）。

- 流程名、参数不在动作表里时拒绝执行（`result.error`）。
- 训练、final-test、对外 issue、远程提交、人工抽检都会登记审批并让这次执行停在 blocked，下一轮 `resume`。
- 不在 sub-RSI 的 submodule 里写任何文件。
