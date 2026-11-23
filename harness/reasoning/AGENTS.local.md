## dm-rsi：meta RSI 的 reasoning

本 loop 改进的对象是 DM-RSI 的端到端流程（`$DMRSI_HOME/runs/<run>/`），不是某个模型。upstream 是 observe 的输出：`runs`（每个流程每一步的 state / note）、`pending`（待人审批）、`submodules`（`behind`、`dirty`）、`inbox` / `outbound`（issue）、`lessons`。

输出 `findings` 列表，每条有 `kind`、`evidence`、`suggest`，并原样透传 `lessons`、`usage_offsets`、`template`、`pending`：

| kind | 什么时候 | suggest |
|---|---|---|
| `runnable` | 某流程第一个未完成的步骤是 pending / running（带 `run`、`step`） | `advance` |
| `step_failed` | 某步骤 failed（带 `run`、`step`、`owner`：data / decision-policy / training，dm-rsi 自己的 bug 时为 null） | `draft_issue`（owner 非空时），否则 `wait` |
| `waiting_human` | 某步骤 blocked（等审批或人工抽检） | `wait` |
| `submodule_dirty` | 有东西写进了只读的 sub-RSI 目录（dm-rsi 的 bug） | `wait` |
| `submodule_behind` | 上游有新 commit | `wait`（升级指针由人决定） |
| `inbox_issue` | sub-RSI 给 dm-rsi 提了 issue | `wait` |
| `no_active_run` | 没有进行中的流程 | `new_run` |

- 判断失败归属时看 note 里的证据：sub-RSI 命令报错、契约不符（缺列、step 不够）归对应方向；dm-rsi 拼错参数、路径不存在归 dm-rsi 自己（owner: null）。
- 不读 test，不建议任何会读 test 的动作。
