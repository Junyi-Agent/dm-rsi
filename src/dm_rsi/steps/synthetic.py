"""合成数据的步骤（ADR 0001 P1 / P3 的验收用）：不用 GPU，用 sub-RSI 自带的合成数据和打分器。

- ``synthetic.make_data``：``dgen.synthetic make``，一张有 step 列的表（train / pool / val / test）。
- ``synthetic.score``：``dgen.synthetic score``，在训练行上拟合一个小模型给 pool / val / test 打分，
  代替真实流程里的「训练 + 打分」。打分器写 csv，这里转成契约要求的 parquet。
- ``synthetic.retrain``：对 Data RSI 产出的每个训练文件各「重训」一次（同上）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from dm_rsi import subenv
from dm_rsi.steps.contracts import check_scores
from dm_rsi.workspace import Outcome, StepState, step_kind

if TYPE_CHECKING:
    from xtra_rsi.payload import Payload

    from dm_rsi.workspace import StepContext

SPLITS = ["pool", "val", "test"]


@step_kind("synthetic.make_data")
def make_data(ctx: StepContext) -> Outcome:
    """生成合成表。"""
    subenv.ensure_env(ctx.cfg)
    data = ctx.ws.dir / "data"
    if not (data / "train.csv").is_file():
        subenv.run_module(
            ctx.cfg,
            ctx.cfg.direction("data"),
            module="dgen.synthetic",
            args=["make", "--out", str(data), "--seed", str(ctx.param("seed", 0))],
            cwd=ctx.dir,
            log_dir=ctx.log_dir,
            name="make",
        )
    pkg = ctx.cfg.direction("data").package_dir(ctx.cfg.root)
    assert pkg is not None  # noqa: S101 -- data 方向一定有包，否则上面的命令已失败
    return Outcome(
        StepState.DONE,
        f"合成表 {data}",
        outputs={
            "rows_format": "csv",
            "train": str(data / "train.csv"),
            "pool": str(data / "pool.csv"),
            "val_rows": str(data / "val.csv"),
            "test_rows": str(data / "test.csv"),
            "rows_dir": str(data),
            "schema": str(pkg / "configs" / "schemas" / "demo_tabular.yaml"),
        },
    )


def _score(ctx: StepContext, train: str, model: str) -> Payload:
    scores = ctx.ws.dir / "scores"
    if not (scores / f"{model}_val.parquet").is_file():
        subenv.run_module(
            ctx.cfg,
            ctx.cfg.direction("data"),
            module="dgen.synthetic",
            args=[
                "score",
                "--train",
                train,
                "--data",
                str(ctx.find_output("rows_dir")),
                "--name",
                model,
                "--out",
                str(scores),
                "--seed",
                str(ctx.param("seed", 0)),
            ],
            cwd=ctx.dir,
            log_dir=ctx.log_dir,
            name=f"score-{model}",
        )
    return check_scores(ctx.cfg, scores, model, SPLITS, convert_csv=True)


@step_kind("synthetic.score")
def score(ctx: StepContext) -> Outcome:
    """给基线模型打分。"""
    model = str(ctx.param("model", "BASE"))
    report = _score(ctx, str(ctx.find_output("train")), model)
    return Outcome(
        StepState.DONE,
        f"{model}: val {cast('Payload', report['val'])['rows']} 行",
        outputs={
            "scores_dir": str(ctx.ws.dir / "scores"),
            "base_model": model,
            "models": {"base": model},
        },
    )


def arms_to_train(ctx: StepContext) -> list[Payload]:
    """要重训的配方：决策给了 ``recipes``（hash 列表）就只训这些，否则训所有还没有模型的。"""
    arms = cast("list[Payload]", ctx.find_output("arms"))
    have = cast("dict[str, str]", ctx.ws.latest_output("models") or {})
    want = ctx.param("recipes")
    if isinstance(want, list):
        return [a for a in arms if str(a["hash"]) in {str(w) for w in want}]
    return [a for a in arms if str(a["hash"]) not in have]


@step_kind("synthetic.retrain")
def retrain(ctx: StepContext) -> Outcome:
    """对每个训练文件各「重训 + 打分」一次。模型名 ``A<hash 前 8 位>``。"""
    arms = arms_to_train(ctx)
    models: dict[str, str] = dict(cast("dict[str, str]", ctx.find_output("models")))
    for arm in arms:
        name = f"A{str(arm['hash'])[:8]}"
        _score(ctx, str(arm["train_path"]), name)
        models[str(arm["hash"])] = name
    return Outcome(
        StepState.DONE,
        f"重训 {len(arms)} 组",
        outputs={"scores_dir": str(ctx.ws.dir / "scores"), "models": cast("Payload", models)},
    )
