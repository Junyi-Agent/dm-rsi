## dm-rsi：meta RSI 的 decision

从 reasoning 的 `findings` 里选**一件**事，输出 `plan`；透传 `lessons`、`usage_offset`、`workspace`、`template`、`pending`。

`plan.action` 只能是下面之一（action 阶段会拒绝动作表之外的值、流程名和参数）：

| action | 其他字段 | 行动阶段做什么 |
|---|---|---|
| `flow` | `flow`（流程名）、`args`（只能用该流程允许的参数）、`why` | 在工作区新建一次流程执行并推进 |
| `resume` | `step`、`why` | 继续推进还没结束的那次执行（等作业、等审批之后） |
| `draft_issue` | `finding`（flow_failed 那条，原样）、`why` | 起草发给 `finding.owner` 的 issue，进审批队列，人确认后才发 |
| `wait` | `why` | 本轮不动，loop 停下等人 |

- 优先级：`resume` > `wait`（有人要做的事）> `draft_issue` > `flow` > `wait`。同一时间只执行一段流程：有没结束的执行时不开始新的。
- `lessons` 里已有 `issue:<step>` 的失败不再起草 issue。
- 流程名和允许的参数：`prepare_data(train_sub)`、`train_rsi(judge, train_gpu_ids, gpu_hours, max_candidates)`（只有 elliptic 模板）、`train_base(train_gpus)`、`score_base()`、`data_rsi(judge, new_recipes, max_arms)`、`retrain(recipes, train_gpus)`、`data_compare(tag)`、`dp_rsi(judge, rounds, replicates, iterations)`、`final_test()`。
