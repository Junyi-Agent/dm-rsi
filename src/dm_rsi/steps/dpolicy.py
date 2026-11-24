"""Decision Policy RSI（``subrsi/decision-policy`` 的 ``dpolicy``）的步骤。

``dpolicy.rsi``：SKILL.md 主流程 2–7 步：check → init → (写提案（judge）→ evolve) × rounds，
``replicates`` > 1 时同一配置只改 ``evolve.seed`` 跑多个副本，再 ``consensus``（1.6 的接受规则）。

运行配置由 dm-rsi 生成（绝对路径，输出在流程目录里），不改 sub-RSI 的 configs/。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import TYPE_CHECKING, cast

from dm_rsi import subenv, versions, yamlfile
from dm_rsi.steps import judge
from dm_rsi.workspace import Outcome, StepState, step_kind

if TYPE_CHECKING:
    from pathlib import Path

    from xtra_rsi.payload import Payload, Value

    from dm_rsi.workspace import StepContext

DEFAULTS: Payload = {
    "protocol": {"by": "step", "select_frac": 0.4},
    "objective": {"metric": "f1_e2e", "min_coverage": 0.0},
    "final_metric": "f1_e2e",
    "costs": {"fp": 1.0, "fn": 10.0, "defer": 0.5},
    "bootstrap": {"reps": 300, "seed": 20261121, "level": 0.95},
    "accept": {"min_lo": 0.0},
}

_FEATURES = r"""
import sys
import pandas as pd
out, paths = sys.argv[1], sys.argv[2:]
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
drop = {"id", "label", "y", "step"}
cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
df[["id", *cols]].drop_duplicates("id").to_parquet(out, index=False)
print(",".join(cols))
"""


def _dpolicy(ctx: StepContext, name: str, *args: str) -> str:
    return subenv.run_module(
        ctx.cfg,
        ctx.cfg.direction("decision-policy"),
        module="dpolicy",
        args=list(args),
        cwd=ctx.dir,
        log_dir=ctx.log_dir,
        name=name,
    )


def _features(ctx: StepContext) -> tuple[str | None, list[str]]:
    """二次判断用的特征表。``features: from_rows`` 时从 val/test 行（csv）里取数值列，不含标签。"""
    mode = ctx.param("features", "none")
    if mode == "none" or mode is None:
        return None, []
    if mode == "from_outputs":
        # 上游步骤（elliptic.build_data）已经生成了特征表，列名从表里读
        return _with_columns(ctx, str(ctx.find_output("features_path")))
    if mode != "from_rows":
        path = str(mode)
        cols = cast("list[str]", ctx.param("feature_columns", []))
        return path, cols
    out = ctx.dir / "features.parquet"
    cols_file = ctx.dir / "features.columns"
    if not out.is_file():
        proc = subprocess.run(
            [
                str(subenv.python(ctx.cfg)),
                "-c",
                _FEATURES,
                str(out),
                str(ctx.find_output("val_rows")),
                str(ctx.find_output("test_rows")),
            ],
            capture_output=True,
            text=True,
            check=True,
            env={**subenv.base_env(ctx.dir), "PATH": "/usr/bin:/bin"},
        )
        cols_file.write_text(proc.stdout.strip())
    return str(out), [c for c in cols_file.read_text().strip().split(",") if c]


_COLUMNS = r"""
import sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
print(",".join(c for c in df.columns if c != "id"))
"""


def _with_columns(ctx: StepContext, path: str) -> tuple[str | None, list[str]]:
    proc = subprocess.run(
        [str(subenv.python(ctx.cfg)), "-c", _COLUMNS, path],
        capture_output=True,
        text=True,
        check=True,
        env={**subenv.base_env(ctx.dir), "PATH": "/usr/bin:/bin"},
    )
    return path, [c for c in proc.stdout.strip().split(",") if c]


def dp_dir(ctx: StepContext, model: str) -> Path:
    """某个模型的 Decision Policy 运行目录。同一个模型多次执行 dp_rsi 是在同一个 dpolicy run 上多跑几轮。"""
    return ctx.ws.dir / "dp-rsi" / model


def config_path(ctx: StepContext, model: str, replicate: int) -> Path:
    """副本 ``replicate`` 的运行配置。"""
    return dp_dir(ctx, model) / f"dpolicy_r{replicate}.yaml"


def _write_config(
    ctx: StepContext, model: str, replicate: int, feat: tuple[str | None, list[str]]
) -> Path:
    path = config_path(ctx, model, replicate)
    if path.is_file():
        return path
    pkg = ctx.cfg.direction("decision-policy").package_dir(ctx.cfg.root)
    assert pkg is not None  # noqa: S101
    data: Payload = {
        "scores_dir": str(ctx.find_output("scores_dir")),
        "name": model,
        "val_split": "val",
        "test_split": "test",
        "step_from_id": bool(ctx.param("step_from_id", False)),
    }
    if feat[0]:
        data["features_path"] = feat[0]
        data["feature_columns"] = cast("Value", feat[1])
    overrides = cast("Payload", ctx.param("dpolicy", {}) or {})
    evolve: Payload = {
        "iterations": int(cast("int", ctx.param("iterations", 3))),
        "auto_candidates": 6,
        "seed": replicate,
    }
    cfg: Payload = {
        "run_name": f"{ctx.ws.id}-{model}-r{replicate}",
        "out_dir": str(dp_dir(ctx, model) / f"r{replicate}"),
        "data": data,
        **DEFAULTS,
        "baseline": str(pkg / "configs" / "policies" / "baseline_f1.yaml"),
        "evolve": evolve,
        **overrides,
    }
    yamlfile.write(path, cfg)
    return path


@step_kind("dpolicy.rsi")
def rsi(ctx: StepContext) -> Outcome:
    """运行 Decision Policy RSI，到 evolve / consensus 为止（final-test 是单独的流程，要审批）。

    用对比选出的模型（``model``）的分数。``rounds`` 是这次执行要**再**跑几轮（提案 + evolve）：
    同一个模型多次执行 dp_rsi 会接着上次的 incumbent 继续。
    """
    subenv.ensure_env(ctx.cfg)
    model = str(ctx.find_output("model"))
    base = dp_dir(ctx, model)
    feat = _features(ctx)
    replicates = int(cast("int", ctx.param("replicates", 1)))
    rounds = int(cast("int", ctx.param("rounds", 1)))
    first = int(
        cast(
            "int",
            ctx.data.setdefault("first_round", len(list((base / "r0").glob(".dmrsi-round*.done")))),
        )
    )
    confs: list[Path] = []
    for r in range(replicates):
        conf = _write_config(ctx, model, r, feat)
        confs.append(conf)
        out = base / f"r{r}"
        _dpolicy(ctx, f"check-r{r}", "check", str(conf))
        if not (out / "incumbent.yaml").is_file():
            _dpolicy(ctx, f"init-r{r}", "init", str(conf))
        for k in range(first, first + rounds):
            done = out / f".dmrsi-round{k}.done"
            if done.is_file():
                continue
            proposals = out / "proposals"
            if r == 0:
                inc = cast(
                    "Payload", json.loads((out / "archive.jsonl").read_text().splitlines()[-1])
                )
                diag = out / "eval" / str(inc.get("hash")) / "diagnosis.md"
                # 每轮重新写提案（SKILL 第 7 步：不要只靠自动变异）；旧提案移到 proposals.round<k-1>
                if k > 0 and proposals.is_dir():
                    shutil.move(proposals, out / f"proposals.round{k - 1}")
                judge.write_proposals(ctx, proposals, diag, feat[1])
            else:
                # 副本用和 r0 相同的提案，只改 seed
                src = base / "r0" / "proposals"
                if proposals.is_dir():
                    shutil.rmtree(proposals)
                shutil.copytree(src, proposals)
            _dpolicy(ctx, f"evolve-r{r}-k{k}", "evolve", str(conf))
            done.write_text(yamlfile.now())
    outputs: Payload = {
        "dp_model": model,
        "dp_dir": str(base),
        "dp_configs": [str(c) for c in confs],
        "incumbent": str(base / "r0" / "incumbent.yaml"),
        "dp_report": str(base / "r0" / "report.md"),
    }
    note = f"模型 {model}：{replicates} 个副本 × 第 {first + 1}–{first + rounds} 轮"
    if replicates > 1:
        text = _dpolicy(ctx, "consensus", "consensus", *map(str, confs))
        (ctx.dir / "consensus.txt").write_text(text)
        pick = base / "r0" / "consensus.yaml"
        outputs["consensus"] = str(pick) if pick.is_file() else None
        note += "，共识：" + ("有" if pick.is_file() else "没有候选过半通过")
    v = versions.register(ctx.ws, ctx.step.name, outputs)
    note += f"；版本判定：{v['decision']}" + (f" → {v['version']}" if v.get("version") else "")
    note += f"（{v['reason']}）"
    outputs["version_decision"] = cast(
        "Value", {k: v[k] for k in ("decision", "version", "reason")}
    )
    return Outcome(StepState.DONE, note, outputs=outputs)
