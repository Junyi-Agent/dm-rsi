"""Data RSI（``subrsi/data`` 的 ``dgen``）的步骤。

``dgen.rsi``：SKILL.md 主流程 1–8 步：check → diagnose → 写配方（judge）→ generate → 抽检（人）→ build。
``dgen.compare``：第 10 步，重训后的新模型和基线在 val 上配对对比，选出交给 Decision Policy 的模型。

运行配置由 dm-rsi 生成（全部绝对路径，输出在流程目录里），不改 sub-RSI 的 configs/。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from dm_rsi import subenv, yamlfile
from dm_rsi.approvals import Kind, State
from dm_rsi.steps import judge
from dm_rsi.workspace import Outcome, StepState, step_kind

if TYPE_CHECKING:
    from pathlib import Path

    from xtra_rsi.payload import Payload, Value

    from dm_rsi.workspace import StepContext

DEFAULTS: Payload = {
    "diagnosis": {"threshold": "max_f1", "kmax": 6, "min_sil": 0.1, "n_ok": 2000, "seed": 0},
    "quality": {
        "checks": ["rules", "dedup", "distance", "discriminability", "cllm"],
        "filter": ["rules", "dedup", "cllm"],
        "cllm_threshold": 0.2,
    },
    "spotcheck": {"n": 50, "pass_ok_rate": 0.9},
    "compare": {
        "primary": "auprc",
        "metrics": ["auprc", "auroc", "f1", "brier"],
        "bootstrap": {"reps": 300, "seed": 20261121, "level": 0.95},
        "accept": {"min_lo": 0.0},
    },
    "seed": 0,
}


def _dgen(ctx: StepContext, name: str, *args: str, ok: tuple[int, ...] = (0,)) -> str:
    return subenv.run_module(
        ctx.cfg,
        ctx.cfg.direction("data"),
        module="dgen",
        args=list(args),
        cwd=ctx.dir,
        log_dir=ctx.log_dir,
        name=name,
        ok_codes=ok,
    )


def config_path(ctx: StepContext) -> Path:
    """本流程 Data RSI 的运行配置（data-rsi 步骤写，compare / final-test 读）。"""
    return ctx.ws.dir / "data-rsi" / "dgen.yaml"


def write_config(ctx: StepContext) -> Path:
    """生成 dgen 运行配置。``params.dgen`` 里的键覆盖默认值。"""
    path = config_path(ctx)
    if path.is_file():
        return path
    overrides = cast("Payload", ctx.param("dgen", {}) or {})
    cfg: Payload = {
        "run_name": ctx.ws.id,
        "out_dir": str(ctx.ws.dir / "data-rsi" / "out"),
        "schema": str(ctx.find_output("schema")),
        "data": {
            "train": str(ctx.find_output("train")),
            "pool": str(ctx.find_output("pool")),
            "leak_check_rows": [
                str(ctx.find_output("val_rows")),
                str(ctx.find_output("test_rows")),
            ],
            "val_rows": str(ctx.find_output("val_rows")),
            "test_rows": str(ctx.find_output("test_rows")),
            "scores_dir": str(ctx.find_output("scores_dir")),
            "model": str(ctx.find_output("base_model")),
        },
        **DEFAULTS,
        **overrides,
    }
    yamlfile.write(path, cfg)
    return path


def _archive(out: Path) -> list[dict[str, object]]:
    f = out / "archive.jsonl"
    if not f.is_file():
        return []
    return [
        cast("dict[str, object]", json.loads(line))
        for line in f.read_text().splitlines()
        if line.strip()
    ]


@step_kind("dgen.rsi")
def rsi(ctx: StepContext) -> Outcome:  # noqa: C901 -- 按 SKILL 主流程逐步写，拆开反而难对照
    """运行 Data RSI 主流程，到 build 为止。"""
    subenv.ensure_env(ctx.cfg)
    conf = write_config(ctx)
    out = ctx.ws.dir / "data-rsi" / "out"

    # 1. check：有 errors 时 dgen 退出码 1
    check = json.loads(_dgen(ctx, "check", "check", str(conf), ok=(0, 1)))
    if check.get("errors"):
        return Outcome(StepState.FAILED, f"dgen check 报错：{check['errors']}")

    # 2. diagnose：每个 run 只能跑一次
    diagnosis = out / "diagnosis" / "diagnosis.md"
    if not diagnosis.is_file():
        _dgen(ctx, "diagnose", "diagnose", str(conf))

    # 3–4. 写配方（judge）
    # new_recipes: 决策要求再写一批（例如上一批都没通过对比）；每次执行只写一次
    recipes = out / "recipes"
    wants_new = bool(ctx.param("new_recipes", False)) and not ctx.data.get("recipes_written")
    if not recipes.is_dir() or not any(recipes.glob("*.yaml")) or wants_new:
        judge.write_recipes(ctx, recipes, diagnosis)
        ctx.data["recipes_written"] = True

    # 5. generate：内容相同的配方只生成一次，所以重复调用安全
    _dgen(ctx, "generate", "generate", str(conf))
    arms = [r for r in _archive(out) if r.get("source") == "recipe"]
    if not arms:
        return Outcome(StepState.FAILED, f"generate 没有产出任何配方，见 {out}/report.md")

    # 7. 人工抽检（spotcheck: required 时等人填 spotcheck.csv 后批准）
    if str(ctx.param("spotcheck", "required")) == "required":
        sheets = [str(out / "arms" / str(a["hash"]) / "spotcheck.csv") for a in arms]
        a = ctx.gate(
            Kind.SPOTCHECK,
            f"Data RSI 人工抽检（{len(arms)} 个配方）",
            "spotcheck",
            {
                "sheets": cast("Value", sheets),
                "how": "在每个 spotcheck.csv 的 verdict 列填 ok/wrong_label/unrealistic/duplicate，ok ≥ 90% 算通过",
            },
            explain=(
                f"Data RSI 生成了 {len(arms)} 组新训练数据（流程 {ctx.ws.id}），进训练之前要人抽检。\n"
                "要你做的：打开下面每个 spotcheck.csv，在 verdict 列填 ok / wrong_label / unrealistic / duplicate"
                "（每份约 50 行），ok ≥ 90% 算通过。\n"
                + "\n".join(f"  {s}" for s in sheets)
                + "\n填完后批准（批准 = 抽检已完成），dm-rsi 会刷新质量报告并继续拼训练文件。\n"
                "拒绝：这一步失败、流程停下（例如生成的数据明显不对）。"
            ),
        )
        if a.state is not State.APPROVED and a.state is not State.DONE:
            return ctx.blocked_on(a, "人工抽检")
        if a.state is State.APPROVED:
            _dgen(ctx, "report", "report", str(conf))
            ctx.store.mark(a.id, ok=True, result={"report": str(out / "report.md")})

    # 8. build：每个配方一个训练文件
    built: list[Payload] = []
    max_arms = int(cast("int", ctx.param("max_arms", 4)))
    for arm in arms[:max_arms]:
        h, name = str(arm["hash"]), str(arm["name"])
        tdir = out / "train" / name
        if not (tdir / "build_report.json").is_file():
            _dgen(ctx, f"build-{h}", "build", str(conf), h, "--name", name)
        report = cast("dict[str, object]", json.loads((tdir / "build_report.json").read_text()))
        train = next((p for p in tdir.iterdir() if p.name.startswith("train.")), None)
        if train is None:
            return Outcome(StepState.FAILED, f"{tdir} 里没有 train.* 文件")
        stats = cast("dict[str, object]", arm.get("stats") or {})
        built.append(
            {
                "hash": h,
                "name": name,
                "train_path": str(train),
                "rows": cast("Value", report.get("rows")),
                "kept": cast("Value", stats.get("kept")),
            }
        )
    return Outcome(
        StepState.DONE,
        f"{len(built)} 个训练文件：" + ", ".join(f"{b['name']}({b['hash']})" for b in built),
        outputs={
            "arms": cast("Value", built),
            "dgen_config": str(conf),
            "dgen_report": str(out / "report.md"),
        },
    )


@step_kind("dgen.compare")
def compare(ctx: StepContext) -> Outcome:
    """在 val 上对比各重训模型和基线，选出交给 Decision Policy 的模型。

    选择规则：主指标（默认 auprc）CI 下界过 ``accept.min_lo``（dgen 的 ``passes_accept``）的里取差值最大的；
    都不过就用基线。val 只用来排序（SKILL 第 10 步），结论以 final-test 为准。
    """
    conf = config_path(ctx)
    models = cast("dict[str, str]", ctx.find_output("models"))
    base_key = "base"
    # dgen 拒绝覆盖已有的对比；每次执行用自己的 tag
    tag = str(ctx.param("tag", f"dmrsi-{ctx.step.name}"))
    res_path = ctx.ws.dir / "data-rsi" / "out" / "compare" / tag / "compare.json"
    if not res_path.is_file():
        runs = ",".join(f"{k}={v}" for k, v in models.items())
        _dgen(
            ctx,
            "compare",
            "compare",
            str(conf),
            "--runs",
            runs,
            "--baseline",
            base_key,
            "--tag",
            tag,
        )
    res = cast("dict[str, object]", json.loads(res_path.read_text()))
    result = cast("dict[str, object]", res["result"])
    primary = str(res.get("primary", "auprc"))
    diffs = [
        d
        for d in cast("list[dict[str, object]]", result.get("diffs") or [])
        if d.get("metric") == primary
    ]
    passing = [d for d in diffs if d.get("passes_accept")]
    winner_key = (
        str(max(passing, key=lambda d: cast("float", d["diff"]))["run"]) if passing else base_key
    )
    summary = {
        str(d["run"]): {"diff": d["diff"], "lo": d["lo"], "hi": d["hi"], "pass": d["passes_accept"]}
        for d in diffs
    }
    return Outcome(
        StepState.DONE,
        f"{primary}：{'选中 ' + winner_key if passing else '没有配方通过，沿用基线'}",
        outputs={
            "model": models[winner_key],
            "model_key": winner_key,
            "compare": str(res_path),
            "compare_summary": cast("Value", summary),
            "compared": cast("Value", sorted(models)),
            "scores_dir": str(ctx.find_output("scores_dir")),
        },
    )
