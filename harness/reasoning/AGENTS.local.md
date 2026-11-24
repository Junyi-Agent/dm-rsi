## dm-rsi：meta RSI 的 reasoning

分工：观察、推理、决策是 dm-rsi 自己的；**行动执行决策选中的那一段具体流程**（数据、训练、打分、对比、Decision Policy、final-test）。本 loop 改进的对象是 DM-RSI 的端到端结果，不是某个模型。

upstream 是 observe 的输出：`workspace`、`template`、`artifacts`（工作区里已有的产物：`rows_dir`、`base_model`、`models`、`arms`、`compared`、`model`、`dp_model`、`dp_headline`……）、`executions`（每次流程执行的 `step`、`flow`、`state`、`note`、`approval`）、`pending`、`submodules`、`inbox` / `outbound`、`lessons`。

输出 `findings` 列表，每条有 `kind`、`evidence`、`suggest`；透传 `lessons`、`usage_offset`、`workspace`、`template`、`pending`：

| kind | 什么时候 | suggest |
|---|---|---|
| `open_execution` | 某次执行 pending / running，或 blocked 但审批已有结果（带 `step`） | `resume` |
| `waiting_human` | 某次执行 blocked，审批还在 pending | `wait` |
| `flow_failed` | 某个流程最近一次执行失败（带 `owner`：data / decision-policy / training；dm-rsi 自己的问题时为 null） | `draft_issue`（owner 非空），否则 `wait` |
| `next_flow` | 按产物看缺哪一段（带 `flow`、`args`）：没数据 → `prepare_data`；没有训练方案（elliptic 模板）→ `train_rsi`；没基线 → `train_base`；基线没打分 → `score_base`；没有新训练数据 → `data_rsi`；有配方没重训 → `retrain`；有模型没对比 → `data_compare`；对比选出的模型和 DP 用的不同 → `dp_rsi`；都齐了 → `final_test` | `flow` |
| `blocked_by_failure` | 下一段应该做的流程上次失败了 | `wait` |
| `finished` | 都完成了 | `wait` |
| `submodule_dirty` / `submodule_behind` / `inbox_issue` | 见名字 | `wait` |

- 不必死守上面的顺序：证据充分时可以建议重跑某段（例如对比里没有配方通过 → `data_rsi` 带 `new_recipes: true`；DP 没有候选通过 → `dp_rsi` 再跑一轮）。`args` 只能用 `dm_rsi.flows.ARGS` 里该流程允许的键。
- 判断失败归属看 note 里的证据：sub-RSI 命令报错、契约不符归对应方向；dm-rsi 拼错参数、路径不存在归 dm-rsi 自己（owner: null）。
- 不读 test，不建议任何会读 test 的动作（`final_test` 由流程自己走审批）。
