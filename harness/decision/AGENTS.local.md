## dm-rsi：meta RSI 的 decision

从 reasoning 的 `findings` 里选**一件**事，输出 `plan`：`action`、`finding`（选中的那条，原样）、`why`；`rejected` 列出没选的。透传 `lessons`、`usage_offsets`、`template`、`pending`。

`action` 只能是下面之一，action 阶段会拒绝动作表之外的值：

| action | 做什么 |
|---|---|
| `advance` | 推进 `finding.run` 这个流程，直到完成、等审批 / 作业或失败 |
| `new_run` | 按 `template` 新建流程 |
| `draft_issue` | 把 `finding`（step_failed）起草成发给 `finding.owner` 的 issue；进审批队列，人确认后才发 |
| `wait` | 只剩人要做的事，本轮不动 |

- 优先级：`advance` > `draft_issue` > `new_run` > `wait`。
- `lessons` 里已有 `issue:<run>/<step>` 的失败不再起草 issue。
- 不选会读 test、会花钱或对外的动作以外的任何事：这些都由流程步骤自己走审批。
