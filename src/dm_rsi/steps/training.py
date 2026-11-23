"""Elliptic++ × elephant-1 的真实流程步骤（ADR 0001 P2）。Training 方向还没有 sub-RSI，dm-rsi 直接调度 onejev-harness。

复用 ``subrsi/decision-policy/pipelines/elliptic_elephant/`` 的脚本（只读），环境变量由 dm-rsi 拼出来，
所有产物写在 ``$DMRSI_HOME``：

- ``elliptic.build_data``（CPU）：harness 的 ``setup.sh``（venv 不存在时）→ ``01_build_data.sh`` →
  从 train 里抽 ``train_sub``（模型训练用）和 ``pool``（训练期、模型没见过，给 Data RSI 诊断用）→
  ``04_policy_features.py``（给 Decision Policy 二次判断的特征）。
- ``elliptic.train``（GPU，每次都要审批）：CPU 上 ``ojh prepare`` + ``ojh resolve`` 得到 lock，再把
  ``python -m harness.runner <lock>`` 作为前台作业交给 GPU 后端（taskmgr 分配卡，machine 配置用
  ``gpu_source: preassigned``）。不用 ``ojh run``：它在 tmux 里后台跑，taskmgr 看不到 GPU 占用。
- ``elliptic.score``（CPU 合并 + GPU 打分）：``02b_merge_lora.py``（本机 CPU）→ GPU 作业里
  ``05_check_merged.py``（|Δp| < 0.005）+ ``03_score_checkpoint.sh``（pool / val / test）。
- ``elliptic.retrain``：对 Data RSI 的每个训练文件各做一次 train + score（每次训练都要审批）。

提交到要审批的后端（ssh / pbs）时：训练的审批里已经写明后端，算同一次确认；打分另外登记
``remote_submit`` 审批。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from dm_rsi import yamlfile
from dm_rsi.approvals import Kind, State
from dm_rsi.backends import requires_approval
from dm_rsi.backends.base import JobRef, JobSpec, JobState
from dm_rsi.backends.local import LocalBackend
from dm_rsi.steps.contracts import check_scores
from dm_rsi.steps.synthetic import arms_to_train  # 两种模板共用同一个选择规则
from dm_rsi.workspace import Outcome, StepState, WorkspaceError, step_kind

if TYPE_CHECKING:
    from xtra_rsi.payload import Payload, Value

    from dm_rsi.config import Config
    from dm_rsi.workspace import StepContext

ROWS_SUBDIR = "data/onejev/addr_txhist_noul"
SCORE_SPLITS = ("pool", "val", "test")
MERGE_MAX_DIFF = 0.005


def paths(cfg: Config) -> dict[str, Path]:
    """流程脚本要的路径（对应 env.example.sh 的变量）。"""
    dp = cfg.direction("decision-policy")
    assert dp.submodule is not None  # noqa: S101
    repo = cfg.root / dp.submodule
    ojh_repo = repo / "external" / "onejev-harness"
    return {
        "DMRSI_HOME": cfg.home,
        "DMRSI_REPO": repo,
        "OJH_REPO": ojh_repo,
        "OJH": ojh_repo / ".venv" / "bin" / "ojh",
        "OJH_HOME": cfg.home / "ojh",
        "DMRSI_DATA": cfg.home / "elliptic",
        "CFD": repo / "external" / "crypto-fraud-detection" / "crypto-fraud-detection",
        "JEV_PYTHON": cfg.home / "jev-venv" / "bin" / "python",
        "DMRSI_PYDEPS": cfg.home / "pydeps",
        "ONEJEV_ROOT": ojh_repo / "vendor" / "OneJev",
    }


def script_env(cfg: Config, scores: Path | None = None) -> dict[str, str]:
    """流程脚本的环境变量。"""
    env = {k: str(v) for k, v in paths(cfg).items()}
    if scores is not None:
        env["DMRSI_SCORES"] = str(scores)
    env["OMP_NUM_THREADS"] = "8"
    return env


def _pipeline_script(cfg: Config, name: str) -> Path:
    return paths(cfg)["DMRSI_REPO"] / "pipelines" / "elliptic_elephant" / name


def _run_local(
    ctx: StepContext, name: str, argv: list[str], *, cwd: Path, env: dict[str, str]
) -> str:
    proc = LocalBackend().run(JobSpec(name=name, argv=argv, cwd=cwd, log_dir=ctx.log_dir, env=env))
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        msg = f"{name} 失败（{proc.returncode}），日志 {ctx.log_dir}/{name}.err：\n{tail}"
        raise WorkspaceError(msg)
    return proc.stdout


_SPLIT_POOL = r"""
import json, random, sys
from pathlib import Path
rows, n, seed = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
lines = (rows / "train.jsonl").read_text().splitlines()
idx = list(range(len(lines)))
random.Random(seed).shuffle(idx)
sub = sorted(idx[:n]); pool = sorted(idx[n:])
(rows / f"train_sub{n // 1000}k.jsonl").write_text("\n".join(lines[i] for i in sub) + "\n")
(rows / "pool.jsonl").write_text("\n".join(lines[i] for i in pool) + "\n")
print(json.dumps({"train_sub": len(sub), "pool": len(pool)}))
"""


@step_kind("elliptic.build_data")
def build_data(ctx: StepContext) -> Outcome:
    """准备 harness 环境和 Elliptic++ 数据（CPU）。"""
    p = paths(ctx.cfg)
    env = script_env(ctx.cfg)
    if not p["OJH_REPO"].joinpath("setup.sh").is_file():
        return Outcome(
            StepState.BLOCKED,
            "onejev-harness 没有检出：在 dm-rsi 根目录运行 git -C subrsi/decision-policy "
            "submodule update --init external/onejev-harness external/crypto-fraud-detection",
        )
    if not p["OJH"].is_file():
        _run_local(ctx, "setup", ["bash", "setup.sh"], cwd=p["OJH_REPO"], env=env)
    rows = p["DMRSI_DATA"] / ROWS_SUBDIR
    if not (rows / "test.jsonl").is_file() or not (rows / "val_sub4000.jsonl").is_file():
        build = str(_pipeline_script(ctx.cfg, "01_build_data.sh"))
        _run_local(ctx, "build-data", ["bash", build], cwd=p["DMRSI_REPO"], env=env)
    n = int(cast("int", ctx.param("train_sub", 40000)))
    sub = rows / f"train_sub{n // 1000}k.jsonl"
    if not sub.is_file() or not (rows / "pool.jsonl").is_file():
        argv = [str(p["JEV_PYTHON"]), "-c", _SPLIT_POOL, str(rows), str(n), "0"]
        _run_local(ctx, "split-pool", argv, cwd=ctx.dir, env=env)
    features = ctx.ws.dir / "features.parquet"
    if not features.is_file():
        feat = str(_pipeline_script(ctx.cfg, "04_policy_features.py"))
        _run_local(
            ctx,
            "features",
            [str(p["JEV_PYTHON"]), feat, str(rows), str(features)],
            cwd=ctx.dir,
            env=env,
        )
    dg = ctx.cfg.direction("data").package_dir(ctx.cfg.root)
    assert dg is not None  # noqa: S101
    schema = ctx.param("dgen_schema", str(dg / "configs" / "schemas" / "elliptic_addr.yaml"))
    return Outcome(
        StepState.DONE,
        f"数据 {rows}（train_sub {n} 行 + pool）",
        outputs={
            "rows_format": "jsonl",
            "rows_dir": str(rows),
            "train": str(sub),
            "pool": str(rows / "pool.jsonl"),
            "val_rows": str(rows / "val.jsonl"),
            "test_rows": str(rows / "test.jsonl"),
            "val_sub": str(rows / "val_sub4000.jsonl"),
            "schema": str(schema),
            "features_path": str(features),
        },
    )


def machine_yaml(ctx: StepContext) -> Path:
    """onejev-harness 的机器配置：GPU 由调度系统预分配（taskmgr 设好 CUDA_VISIBLE_DEVICES）。"""
    p = paths(ctx.cfg)
    path = p["OJH_HOME"] / "machine.dmrsi.yaml"
    if not path.is_file():
        yamlfile.write(
            path,
            {
                "python": str(p["OJH_REPO"] / ".venv" / "bin" / "python"),
                "model_dir": str(ctx.param("model_dir", "/ssd1/junyi/jev/models")),
                "out_root": str(p["OJH_HOME"] / "out"),
                "gpu_source": "preassigned",
                "forbidden_write_roots": [str(p["DMRSI_REPO"])],
                "env": {"HF_HUB_OFFLINE": "1"},
            },
        )
    return path


def _exp_yaml(ctx: StepContext, model: str, train: str) -> tuple[Path, str]:
    p = paths(ctx.cfg)
    name = f"{ctx.ws.id}-{model}".lower()
    path = p["OJH_HOME"] / "work" / name / "exp.yaml"
    if not path.is_file():
        yamlfile.write(
            path,
            {
                "name": name,
                "base": str(ctx.param("base", "elephant-1")),
                "recipe": str(_pipeline_script(ctx.cfg, "recipes/elephant_lora_ddp_2g.yaml")),
                "gpus": int(cast("int", ctx.param("train_gpus", 2))),
                "data": {
                    "adapter": "onejev_rows",
                    "train": train,
                    "eval": {"val": {"train_labels": str(ctx.find_output("val_sub"))}},
                },
                "fields": {"label": "target.yes", "group": "meta.state_id", "text": "state_json"},
                "evaluator": {"type": "binary_decision", "gate": True},
                "seeds": [int(cast("int", ctx.param("seed", 0)))],
                "overrides": {},
            },
        )
    return path, name


def _ojh_python(ctx: StepContext) -> str:
    return str(
        ctx.param("ojh_python", str(paths(ctx.cfg)["OJH_REPO"] / ".venv" / "bin" / "python"))
    )


def _job(ctx: StepContext, key: str) -> JobRef | None:
    d = ctx.data.get(key)
    return JobRef.from_dict(cast("Payload", d)) if isinstance(d, dict) else None


def train_model(ctx: StepContext, model: str, train: str) -> tuple[StepState, str, str | None]:
    """训练一个模型。返回 (状态, 说明, run_dir)。状态不是 DONE 时调用方原样返回。"""
    p = paths(ctx.cfg)
    exp, name = _exp_yaml(ctx, model, train)
    gpus = int(cast("int", ctx.param("train_gpus", 2)))
    jkey = f"train_job/{model}"
    job = _job(ctx, jkey)
    a = ctx.gate(
        Kind.TRAIN,
        f"训练 {name}（elephant-1 LoRA，{gpus} 卡，约 2.4 GPU 小时）",
        f"train/{model}",
        {
            "exp": str(exp),
            "train_rows": train,
            "backend": str(ctx.param("gpu_backend", "xtrah200")),
        },
        explain=(
            f"要在后端 {ctx.param('gpu_backend', 'xtrah200')} 上训练模型 {model}"
            f"（流程 {ctx.ws.id}，elephant-1 LoRA，{gpus} 张卡，按以往约 2.4 GPU 小时）。\n"
            f"训练数据：{train}\n实验配置：{exp}\n"
            "为什么问你：训练占共享 GPU 时间，ADR 0001 规定每次训练都要人确认。\n"
            "批准：dm-rsi 提交作业，训练结束后自动合并 LoRA、打分，流程继续。\n"
            "拒绝：这一步失败、流程停下；之后可以改参数重开流程。\n"
            "建议：确认 GPU 空闲、训练数据是你想要的那一份再批。"
        ),
    )
    if a.state not in (State.APPROVED, State.DONE):
        out = ctx.blocked_on(a, f"训练 {model}")
        return out.state, out.note, None
    if job is None:
        env = script_env(ctx.cfg)
        m = str(machine_yaml(ctx))
        ojh = str(ctx.param("ojh_bin", str(p["OJH"])))
        _run_local(
            ctx,
            f"prepare-{model}",
            [ojh, "prepare", str(exp), "--machine", m],
            cwd=exp.parent,
            env=env,
        )
        out = _run_local(
            ctx,
            f"resolve-{model}",
            [ojh, "resolve", str(exp), "--machine", m],
            cwd=exp.parent,
            env=env,
        )
        lock = out.strip().splitlines()[-1]
        backend = ctx.backend(str(ctx.param("gpu_backend", "xtrah200")))
        job = backend.submit(
            JobSpec(
                name=f"{ctx.ws.id}-train-{model}",
                argv=[_ojh_python(ctx), "-m", "harness.runner", lock],
                cwd=p["OJH_REPO"],
                log_dir=ctx.log_dir,
                env={"PYTHONPATH": str(p["OJH_REPO"])},
                gpus=gpus,
                timeout=int(cast("int", ctx.param("train_timeout", 6 * 3600))),
            )
        )
        ctx.data[jkey] = job.to_dict()
        ctx.data[f"lock/{model}"] = lock
        if a.state is State.APPROVED:
            ctx.store.mark(a.id, ok=True, result={"job": job.to_dict(), "lock": lock})
        return StepState.RUNNING, f"训练 {model} 已提交：{job.backend} {job.id}", None
    st = ctx.backend(job.backend).status(job)
    lock = Path(str(ctx.data[f"lock/{model}"]))
    run_dir = str(yamlfile.read(lock).get("run_dir"))
    if st in (JobState.QUEUED, JobState.RUNNING):
        return StepState.RUNNING, f"训练 {model}：{st}（{job.backend} {job.id}）", None
    if st is JobState.DONE and (Path(run_dir) / "DONE").is_file():
        return StepState.DONE, f"训练 {model} 完成：{run_dir}", run_dir
    return StepState.FAILED, f"训练 {model} 作业 {st}，看 {run_dir}/DONE 和 runner.log", None


_SCORE_JOB = r"""set -euo pipefail
RUN=$1 MERGED=$2 NAME=$3
PY=$OJH_REPO/.venv/bin/python
export TORCH_COMPILE_DISABLE=1
PYTHONPATH=$DMRSI_PYDEPS "$PY" "$DMRSI_REPO/pipelines/elliptic_elephant/05_check_merged.py" "$RUN" "$MERGED" 200
"$PY" - "$MERGED/merge_check.json" <<'P'
import json, sys
d = json.load(open(sys.argv[1]))["mean_abs_p_diff"]
sys.exit(0 if d == d and d < __MAXDIFF__ else f"merged model disagrees with the harness scorer: mean |dp| = {d}")
P
SCORE_SPLITS="__SPLITS__" bash "$DMRSI_REPO/pipelines/elliptic_elephant/03_score_checkpoint.sh" "$MERGED" "$NAME" cuda:0
"""


def _merged(run_dir: str) -> Path | None:
    return next(iter(sorted(Path(run_dir).glob("runs/*/merged-step*-cpu"))), None)


def score_model(ctx: StepContext, model: str, run_dir: str) -> tuple[StepState, str]:  # noqa: PLR0911 -- 每个出口对应一个流程状态
    """合并 LoRA（本机 CPU）→ GPU 作业里核对合并结果并给 pool / val / test 打分。"""
    p = paths(ctx.cfg)
    scores = ctx.ws.dir / "scores"
    if all((scores / f"{model}_{s}.parquet").is_file() for s in SCORE_SPLITS):
        check_scores(ctx.cfg, scores, model, list(SCORE_SPLITS))
        return StepState.DONE, f"{model} 已打分"
    merged = _merged(run_dir)
    if merged is None:
        env = {**script_env(ctx.cfg), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "16"}
        py = str(p["OJH_REPO"] / ".venv" / "bin" / "python")
        merge = str(_pipeline_script(ctx.cfg, "02b_merge_lora.py"))
        _run_local(
            ctx,
            f"merge-{model}",
            ["nice", "-n", "10", py, merge, run_dir],
            cwd=p["DMRSI_REPO"],
            env=env,
        )
        merged = _merged(run_dir)
        if merged is None:
            return StepState.FAILED, f"合并后找不到 {run_dir}/runs/*/merged-step*-cpu"
    jkey = f"score_job/{model}"
    job = _job(ctx, jkey)
    if job is None:
        script = ctx.dir / "score_job.sh"
        body = _SCORE_JOB.replace("__MAXDIFF__", str(MERGE_MAX_DIFF)).replace(
            "__SPLITS__", " ".join(SCORE_SPLITS)
        )
        script.write_text(body)
        backend = ctx.backend(str(ctx.param("gpu_backend", "xtrah200")))
        if requires_approval(backend):
            a = ctx.gate(
                Kind.REMOTE_SUBMIT,
                f"在远程后端 {backend.name} 上给 {model} 打分（1 卡）",
                f"score/{model}",
                {"backend": backend.name, "model": model},
                explain=(
                    f"要在远程后端 {backend.name} 上给模型 {model} 打分（1 张卡，pool / val / test 三份）。\n"
                    "为什么问你：提交到 ssh / PBS 机器是对外动作，ADR 0001 规定要人确认。\n"
                    "批准：上传作业脚本并提交；拒绝：这一步失败、流程停下。"
                ),
            )
            if a.state not in (State.APPROVED, State.DONE):
                out = ctx.blocked_on(a, f"远程打分 {model}")
                return out.state, out.note
            if a.state is State.APPROVED:
                ctx.store.mark(a.id, ok=True, result={"backend": backend.name})
        job = backend.submit(
            JobSpec(
                name=f"{ctx.ws.id}-score-{model}",
                argv=["bash", str(script), run_dir, str(merged), model],
                cwd=p["DMRSI_REPO"],
                log_dir=ctx.log_dir,
                env=script_env(ctx.cfg, scores),
                gpus=1,
                timeout=int(cast("int", ctx.param("score_timeout", 3 * 3600))),
            )
        )
        ctx.data[jkey] = job.to_dict()
        return StepState.RUNNING, f"打分 {model} 已提交：{job.backend} {job.id}"
    st = ctx.backend(job.backend).status(job)
    if st in (JobState.QUEUED, JobState.RUNNING):
        return StepState.RUNNING, f"打分 {model}：{st}"
    if st is not JobState.DONE:
        ctx.backend(job.backend).fetch(job, ctx.log_dir / f"score-{model}")
        return StepState.FAILED, f"打分 {model} 作业 {st}，日志在 {ctx.log_dir}/score-{model}"
    check_scores(ctx.cfg, scores, model, list(SCORE_SPLITS))
    return StepState.DONE, f"{model} 已打分"


@step_kind("elliptic.train")
def train(ctx: StepContext) -> Outcome:
    """训练基线模型（GPU，要审批）。"""
    model = str(ctx.param("model", "BASE"))
    st, note, run_dir = train_model(ctx, model, str(ctx.find_output("train")))
    if st is not StepState.DONE:
        return Outcome(st, note)
    return Outcome(
        StepState.DONE, note, outputs={"run_dirs": {model: run_dir}, "base_model": model}
    )


@step_kind("elliptic.score")
def score(ctx: StepContext) -> Outcome:
    """给基线模型打分（pool / val / test）。"""
    model = str(ctx.param("model", "BASE"))
    run_dir = str(cast("dict[str, str]", ctx.find_output("run_dirs"))[model])
    st, note = score_model(ctx, model, run_dir)
    if st is not StepState.DONE:
        return Outcome(st, note)
    return Outcome(
        StepState.DONE,
        note,
        outputs={
            "scores_dir": str(ctx.ws.dir / "scores"),
            "base_model": model,
            "models": {"base": model},
        },
    )


@step_kind("elliptic.retrain")
def retrain(ctx: StepContext) -> Outcome:
    """对 Data RSI 的每个训练文件：训练（要审批）+ 打分。一个一个来，同一时间只占一组卡。"""
    arms = arms_to_train(ctx)
    models: dict[str, Value] = dict(cast("dict[str, Value]", ctx.find_output("models")))
    for arm in arms:
        h = str(arm["hash"])
        name = f"A{h[:8]}"
        st, note, run_dir = train_model(ctx, name, str(arm["train_path"]))
        if st is not StepState.DONE or run_dir is None:
            return Outcome(st, note)
        st, note = score_model(ctx, name, run_dir)
        if st is not StepState.DONE:
            return Outcome(st, note)
        models[h] = name
    return Outcome(
        StepState.DONE,
        f"重训并打分 {len(arms)} 组",
        outputs={"scores_dir": str(ctx.ws.dir / "scores"), "models": models},
    )
