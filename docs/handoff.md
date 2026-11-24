# sub-RSI → dm-rsi 的交接文件（`handoff: 1`）

dm-rsi 读各方向结果的唯一入口是 `src/dm_rsi/steps/handoff.py`。各方向在自己的运行目录里写一个 `handoff.yaml`，dm-rsi 就只认这个文件的字段，不再读内部记录；内部记录（`archive.jsonl`、`incumbent.yaml`、`trials/`……）以后怎么改都不影响 dm-rsi。

还没有 `handoff.yaml` 时，dm-rsi 从现在的内部记录推出同样的内容（兼容读取），两种来源过同一套字段检查。缺字段或类型不对时，这段流程失败，错误写明文件、字段和实际值，loop 停下等人处理。

公共字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `handoff` | 整数 | 格式版本，现在是 `1`。字段有不兼容的改动时加一 |
| `kind` | 字符串 | `training` / `decision-policy` / `data` |

写的时机：这一段结束时写（或覆盖），先写临时文件再改名，避免 dm-rsi 读到半个文件。

## Training（trsi）：`<out_dir>/<run_name>/handoff.yaml`

`trsi report` 时写。dm-rsi 用它决定之后的基线和重训用哪个方案。

```yaml
handoff: 1
kind: training
scheme:                       # 当前方案（incumbent）
  candidate_id: 3f2a9c        # 字符串
  base: elephant-1            # 字符串
  recipe: /abs/path/to/elephant_lora_1g.yaml   # 绝对路径
  overrides:                  # 键 → {value, reason}，没有就写 {}
    onejev.lr: {value: 2.0e-4, reason: "曲线到最后还在变好"}
  is_start: false             # 布尔：是不是起点（团队配方）
  note: "学习率加倍"           # 字符串，可省略
checkpoints:                  # 种子 → 这个方案该种子的训练结果（只列完成的）
  "0": {run_dir: /abs/run, checkpoint: checkpoints/step-800, metrics: {auprc: 0.91}}   # metrics 可以是 null
```

编排训练时 dm-rsi 现在还要读 trsi 的过程记录（`archive.jsonl` 的 `use`、`launch`、`incumbent`、`propose`、`probe`、`verdict` 事件和 `trials/<id>.json` 的 `status`、`seed`），这些字段已经在 `TrsiRecords` 里做了检查。更稳的做法是 trsi 提供一个机器可读的状态命令（例如 `trsi status <run.yaml> --json`：在跑的试验和占的卡、还要启动的试验、失败的试验和尝试次数、预算），dm-rsi 就不用自己复现 `next` 的逻辑。

## Decision Policy（dpolicy）：`<out_dir>/<run_name>/handoff.yaml`

每次 `evolve` 结束时写。dm-rsi 用它登记版本（`versions.py`）和写 final-test 的审批说明。

```yaml
handoff: 1
kind: decision-policy
incumbent:
  name: abstain30_logreg      # 字符串
  hash: 9b1c0e                # 字符串
  objective: 0.612            # 数：val(select) 上的目标值（配置里的 objective.metric）
  diff_vs_previous: 0.018     # 数，可省略：最后一次接受时相对前一个 incumbent 的差
baseline:
  objective: 0.560            # 基线策略的目标值
evaluated: 10                 # 评估过的候选数（不含基线）
accepted: 2                   # 被接受的候选数；0 表示 incumbent 仍是基线策略
```

## Data（dgen）

dgen 现在交给 dm-rsi 的有三处：`generate` 之后的配方列表、`build` 之后的训练文件、`compare` 的结果。建议在 `<out_dir>/handoff.yaml` 里汇总：

```yaml
handoff: 1
kind: data
arms:                          # generate 产出的配方，按生成顺序
  - {hash: 20fda9982f37, name: smote_baseline, kept: 812}
builds:                        # build 产出的训练文件
  smote_baseline: {train_path: /abs/train/smote_baseline/train.jsonl, rows: 40812}
compares:                      # 每次 compare 一项，键是 --tag
  dmrsi-005-data_compare:
    primary: auprc
    n: 1200                    # 参与对比的行数
    n_pos: 81
    table:                     # 模型 → 各指标（auprc、auroc、brier 会被 dm-rsi 从分数文件重算核对）
      base: {auprc: 0.489, auroc: 0.906, brier: 0.047, f1: 0.52, threshold: 0.31}
    diffs:                     # 每个模型、每个指标相对基线
      - {run: 20fda9982f37, metric: auprc, diff: 0.059, lo: 0.014, hi: 0.102, passes_accept: true}
```

dgen 的正式交接文件 dm-rsi 还没有接（`handoff.py` 里 Data 部分只有兼容读取），等格式和 Bingqiao 定下来再接。
