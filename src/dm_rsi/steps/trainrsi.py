"""Training RSI（Chendie，``trsi``）：在卡时预算内选训练方案（ADR 0001，2026-11-24 修订，方案 A）。

流程 ``train_rsi`` 排在 Data RSI 之前。选出的方案（基座 + 配方 + 带原因的 overrides）作为产物 ``train_scheme``，
之后的基线（``train_base``）和 Data RSI 各组数据的重训（``retrain``）都用它训练。

一次 ``run_step`` 推进一步，可以反复调用（训练要几个小时，期间返回 RUNNING）：

1. 第一次：从 sub-RSI 的 ``experiments/elliptic_txhist/`` 生成本次的 ``exp.yaml`` 和 ``run.yaml``（路径改成绝对路径，
   训练行来自 ``prepare_data`` 的产物），``trsi check --ojh``（不占卡）、``trsi init``。
2. 每次：``trsi collect`` → ``trsi next``（判探针、补种子、接受）→ 算出还要训练哪些（起点的种子、新候选的探针、
   探针通过后的其余种子）。
3. 每次启动训练前登记 ``train`` 审批（ADR 0001：每次训练都要人确认），批准后 ``trsi launch``（本机 ``ojh run``，
   立即返回）。卡从 ``train_gpu_ids`` 里挑没被本流程占用的。
   新候选的探针在登记审批**之前**先用 trsi 自己的规则检查（:func:`_precheck`：在不在 space 里、是不是死路），
   不合格的候选记下原因、跳过（算一次尝试，计入 ``max_candidates``），下次判断环节会看到原因，不让整段失败。
   失败的试验按 ``max_retries`` 重试（trsi 对 failed 的试验用同一条 ``launch`` 重跑），每次重试也要审批；
   重试用完仍失败，整段才失败。（ADR 0001，2026-11-24 第二次修订，借鉴 Borui 的「每轮尝试上限」。）
4. 没有在跑、也没有要启动的：``next`` 说 STOP、候选数到了 ``max_candidates``、或判断环节不再提新候选时结束，
   ``trsi report``；否则判断环节（``judge: agent`` 时起一个 agent 会话，按 trsi 的 SKILL.md 第 4–5 步诊断、
   写 1–2 个候选）写进 ``proposals/``。

trsi 的记录只读不写（``archive.jsonl``、``incumbent.yaml``、``trials/``），改动一律经它的命令。
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from dm_rsi import agent, subenv
from dm_rsi.approvals import Kind, State
from dm_rsi.steps.training import paths, script_env
from dm_rsi.workspace import Outcome, StepState, step_kind

if TYPE_CHECKING:
    from xtra_rsi.payload import Payload, Value

    from dm_rsi.config import Config
    from dm_rsi.workspace import StepContext

EXPERIMENT = "experiments/elliptic_txhist"
DIRECTION = "training"


def _load(path: Path) -> dict[str, Value]:
    """读 trsi 的 YAML（有整数、布尔键，例如 ``dead: {2: ...}``，不走 dm-rsi 的严格读取）。"""
    return cast("dict[str, Value]", yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _dump(path: Path, data: dict[str, Value]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)


def env_file(cfg: Config) -> Path:
    """生成 trsi 在调 ``ojh`` 前 source 的环境变量文件（变量名和 trsi 的 ``run.yaml`` 约定一致）。"""
    path = cfg.home / "dmrsi-env.sh"
    body = "# 由 dm-rsi 生成（steps/training.py 的 script_env），trsi 调 ojh 前 source\n" + "".join(
        f"export {k}={shlex.quote(v)}\n" for k, v in sorted(script_env(cfg).items())
    )
    if not path.is_file() or path.read_text(encoding="utf-8") != body:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return path


def _pkg(ctx: StepContext) -> Path:
    pkg = ctx.cfg.direction(DIRECTION).package_dir(ctx.cfg.root)
    if pkg is None:
        msg = "training 方向没有 sub-RSI 包（dm-rsi.yaml 的 submodule / package）"
        raise ValueError(msg)
    return pkg


def _trsi(ctx: StepContext, name: str, *args: str) -> str:
    return subenv.run_module(
        ctx.cfg,
        ctx.cfg.direction(DIRECTION),
        module="trsi",
        args=list(args),
        cwd=ctx.dir,
        log_dir=ctx.log_dir,
        name=name,
    )


def write_configs(ctx: StepContext) -> Path:
    """本次的 ``exp.yaml`` + ``run.yaml``（已有就不改：trsi 规定改规则要换 run_name，也就是开新的一段流程）。"""
    run_path = ctx.dir / "run.yaml"
    if run_path.is_file():
        return run_path
    src = _pkg(ctx) / EXPERIMENT
    exp = _load(src / "exp.yaml")
    rows = Path(str(ctx.find_output("rows_dir")))
    data = cast("dict[str, Value]", exp["data"])
    data["train"] = str(rows / "train.jsonl")  # 完整训练集：和 Chendie 的环境、Decision Policy 一致
    data["eval"] = {"val": {"train_labels": str(ctx.find_output("val_rows"))}}
    _dump(ctx.dir / "exp.yaml", exp)

    run = _load(src / "run.yaml")

    def ab(p: Value) -> str:
        return str((src / str(p)).resolve())

    run["run_name"] = f"{ctx.ws.id}-{ctx.step.name}"
    run["out_dir"] = str(ctx.param("trsi_out_dir", str(ctx.cfg.home / "trsi" / "elliptic-txhist")))
    run["exp"] = str(ctx.dir / "exp.yaml")
    ojh: dict[str, Value] = {
        "bin": str(ctx.param("ojh_bin", str(paths(ctx.cfg)["OJH"]))),
        "env_file": str(env_file(ctx.cfg)),
    }
    if ctx.param("ojh_machine"):
        ojh["machine"] = str(ctx.param("ojh_machine"))
    run["ojh"] = ojh
    start = cast("dict[str, Value]", run["start"])
    start["recipe"] = ab(start["recipe"])
    space = cast("dict[str, Value]", run["space"])
    space["recipes"] = [ab(r) for r in cast("list[Value]", space["recipes"])]
    gpus = cast("dict[str, Value]", run["gpus"])
    gpus["ids"] = cast("Value", ctx.param("train_gpu_ids", gpus["ids"]))
    budget = cast("dict[str, Value]", run["budget"])
    budget["gpu_hours"] = cast("Value", ctx.param("gpu_hours", budget["gpu_hours"]))
    _dump(run_path, run)
    return run_path


# ---- 读 trsi 的记录（只读） --------------------------------------------------------------------------------


class Records:
    """``<out_dir>/<run_name>/`` 下的记录和共享的 ``<out_dir>/trials/``。"""

    def __init__(self, run_path: Path) -> None:
        """读一次快照。"""
        self.run: dict[str, Value] = _load(run_path)
        self.out = Path(str(self.run["out_dir"]))
        self.home = self.out / str(self.run["run_name"])
        p = self.home / "archive.jsonl"
        lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
        self.events: list[dict[str, Value]] = [json.loads(x) for x in lines if x.strip()]
        self.trials: dict[str, dict[str, Value]] = {}
        for e in self.of("use"):
            t = self.out / "trials" / f"{e['trial']}.json"
            if t.is_file() and str(e["trial"]) not in self.trials:
                self.trials[str(e["trial"])] = json.loads(t.read_text(encoding="utf-8"))

    def of(self, kind: str) -> list[dict[str, Value]]:
        """某类事件。"""
        return [e for e in self.events if e.get("event") == kind]

    @property
    def seeds(self) -> list[int]:
        """规则里的种子，第一个是探针。"""
        rules = cast("dict[str, Value]", self.run.get("rules") or {})
        return [int(cast("int", s)) for s in cast("list[Value]", rules.get("seeds") or [0, 1, 2])]

    @property
    def start_cid(self) -> str | None:
        """起点方案的编号（init 时第一次设为当前方案）。"""
        inc = self.of("incumbent")
        return str(inc[0]["candidate"]) if inc else None

    def incumbent(self) -> dict[str, Value]:
        """``incumbent.yaml``。"""
        return _load(self.home / "incumbent.yaml")

    def running_gpus(self) -> set[int]:
        """本 run 里还在跑的试验占着的卡。"""
        busy: set[int] = set()
        launched = {str(e["trial"]): e for e in self.of("launch")}
        for tid, rec in self.trials.items():
            if rec.get("status") == "running" and tid in launched:
                busy |= {int(cast("int", g)) for g in cast("list[Value]", launched[tid]["gpus"])}
        return busy


def _recipe_gpus(recipe: Path) -> int:
    return int(cast("int", _load(recipe).get("gpus", 1)))


def _item(  # noqa: PLR0913 -- 一次启动的全部描述
    label: str, proposal: str | None, seed: int, why: str, cand: Value, *, key: str | None = None
) -> dict[str, Value]:
    return {
        "label": label,
        "proposal": proposal,
        "seed": seed,
        "why": why,
        "cand": cand,
        "key": key or f"launch/{label}/s{seed}",
    }


def _start_seeds(rec: Records) -> list[dict[str, Value]]:
    if not rec.start_cid:
        return []
    why = "起点（团队配方）的种子，不计预算"
    return [
        _item("start", None, s, why, rec.run["start"])
        for s in rec.seeds
        if f"{rec.start_cid}-s{s}" not in rec.trials
    ]


def _new_probes(ctx: StepContext, rec: Records) -> list[dict[str, Value]]:
    launched = cast("list[Value]", ctx.data.setdefault("launched_proposals", []))
    return [
        _item(p.name, str(p), rec.seeds[0], "新候选的探针种子", cast("Value", _load(p)))
        for p in sorted((rec.home / "proposals").glob("*.yaml"))
        if p.name not in launched
    ]


def _more_seeds(rec: Records) -> list[dict[str, Value]]:
    out: list[dict[str, Value]] = []
    verdicts = {str(e["candidate"]) for e in rec.of("verdict")}
    inc_cid = str(rec.incumbent().get("candidate_id"))
    for e in rec.of("propose"):
        cid = str(e["candidate"])
        probes = [x for x in rec.of("probe") if x["candidate"] == cid and x["against"] == inc_cid]
        if cid in verdicts or cid == inc_cid or not probes or probes[-1]["verdict"] != "more_seeds":
            continue
        why = f"探针过了门槛（比当前方案高 {float(cast('float', probes[-1].get('gain') or 0)):+.4f}），补种子"
        out.extend(
            _item(Path(str(e["proposal"])).name, str(e["proposal"]), s, why, e["candidate_dict"])
            for s in rec.seeds[1:]
            if f"{cid}-s{s}" not in rec.trials
        )
    return out


_PRECHECK = r"""
import json, sys
from pathlib import Path
from trsi.candidate import dead_ends, load_proposal, outside_space
from trsi.config import load_run
run = load_run(Path(sys.argv[1]))
try:
    c, _ = load_proposal(Path(sys.argv[2]), run)
except Exception as e:  # 候选文件本身不合法
    print(json.dumps([f"候选文件不合法：{type(e).__name__}: {e}"]))
    raise SystemExit(0)
probs = [f"超出批准的 space：{x}" for x in outside_space(c, run.space)]
probs += [f"死路：{x}" for x in dead_ends(c, run.space)]
print(json.dumps(probs))
"""


def _precheck(ctx: StepContext, run_path: Path, proposal: Path) -> list[str]:
    """用 trsi 自己的规则检查新候选（不启动、不写 trsi 的记录），返回问题列表。"""
    out = subenv.run_code(
        ctx.cfg, ctx.cfg.direction(DIRECTION), _PRECHECK, [str(run_path), str(proposal)]
    )
    return [str(x) for x in cast("list[Value]", json.loads(out))]


def _refuse_bad_probes(
    ctx: StepContext, run_path: Path, probes: list[dict[str, Value]]
) -> list[dict[str, Value]]:
    """审批之前筛掉 trsi 会拒绝的候选：记进 ``refused``，算一次尝试（进 ``launched_proposals``）。"""
    launched = cast("list[Value]", ctx.data.setdefault("launched_proposals", []))
    refused = cast("dict[str, Value]", ctx.data.setdefault("refused", {}))
    ok: list[dict[str, Value]] = []
    for item in probes:
        probs = _precheck(ctx, run_path, Path(str(item["proposal"])))
        if probs:
            refused[str(item["label"])] = cast("Value", probs)
            launched.append(str(item["label"]))
        else:
            ok.append(item)
    return ok


def _retries(ctx: StepContext, rec: Records) -> tuple[list[dict[str, Value]], list[str]]:
    """失败的试验：还能重试的 → 启动项；重试用完的 → 说明。"""
    max_retries = int(cast("int", ctx.param("max_retries", 1)))
    proposals = {str(e["candidate"]): str(e["proposal"]) for e in rec.of("propose")}
    todo: list[dict[str, Value]] = []
    exhausted: list[str] = []
    for tid, t in rec.trials.items():
        if t.get("status") != "failed":
            continue
        err = cast("dict[str, Value]", t.get("result") or {}).get("error")
        attempts = int(cast("int", t.get("attempts") or 1))
        if attempts > max_retries:
            exhausted.append(f"{tid}（{attempts} 次都失败，最后一次：{err}）")
            continue
        cid = str(t.get("candidate_id"))
        proposal = None if cid == rec.start_cid else proposals.get(cid)
        if proposal is None and cid != rec.start_cid:
            exhausted.append(f"{tid}（找不到它的候选文件，不能重试：{err}）")
            continue
        label = "start" if proposal is None else Path(proposal).name
        why = (
            f"重试：这个试验上次失败（{err}），第 {attempts + 1} 次尝试，最多 {max_retries + 1} 次。"
            "先看 run 目录的 DONE、runner.log，确认是偶发问题（OOM、卡被占、节点故障）再批；"
            "如果是配置本身的问题，重试也会失败，拒绝即可"
        )
        cand = t.get("candidate") or (rec.run["start"] if proposal is None else None)
        seed = int(cast("int", t["seed"]))
        todo.append(
            _item(label, proposal, seed, why, cand, key=f"retry/{label}/s{seed}/{attempts}")
        )
    return todo, exhausted


def _launches(ctx: StepContext, rec: Records, run_path: Path) -> list[dict[str, Value]]:
    """还要启动的训练：失败试验的重试 → 起点的种子 → 新候选的探针 → 探针通过的候选的其余种子。"""
    retries, _ = _retries(ctx, rec)
    probes = _refuse_bad_probes(ctx, run_path, _new_probes(ctx, rec))
    out = [*retries, *_start_seeds(rec), *probes, *_more_seeds(rec)]
    for item in out:
        recipe = Path(str(cast("dict[str, Value]", item["cand"])["recipe"]))
        item["recipe"] = str(
            recipe if recipe.is_absolute() else (run_path.parent / recipe).resolve()
        )
    return out


def _describe(cand: dict[str, Value]) -> str:
    ovs = cast("dict[str, Value]", cand.get("overrides") or {})
    lines = [f"基座 {cand.get('base')}，配方 {Path(str(cand.get('recipe'))).name}"]
    for k, o in sorted(ovs.items()):
        od = cast("dict[str, Value]", o)
        lines.append(f"  {k} = {od.get('value')}（{od.get('reason')}）")
    if cand.get("note"):
        lines.append(f"  想法：{cand['note']}")
    if cand.get("prediction"):
        lines.append(f"  预测：{cand['prediction']}")
    return "\n".join(lines)


_BUDGET_RE = re.compile(r"budget\s+([\d.]+)\s*/\s*([\d.]+)")


def _launch(
    ctx: StepContext, run_path: Path, item: dict[str, Value], gpus: list[int], budget: str
) -> StepState:
    cand = cast("dict[str, Value]", item["cand"])
    seed = int(cast("int", item["seed"]))
    est = cast(
        "dict[str, Value]",
        cast("dict[str, Value]", Records(run_path).run["budget"]).get("estimate") or {},
    )
    hours = est.get(str(cand.get("base")))
    a = ctx.gate(
        Kind.TRAIN,
        f"Training RSI 训练 {item['label']} 种子 {seed}（{cand.get('base')}，卡 {gpus}）",
        str(item["key"]),
        {
            "run": str(run_path),
            "proposal": cast("Value", item["proposal"]),
            "seed": seed,
            "gpus": cast("Value", gpus),
        },
        explain=(
            f"Training RSI（Chendie 的 trsi）要在本机 {gpus} 号卡上训练一个种子（流程 {ctx.ws.id}/{ctx.step.name}）。\n"
            f"为什么训练：{item['why']}\n方案：{_describe(cand)}\n"
            f"种子：{seed}；预计约 {hours if hours is not None else '?'} 卡时（run.yaml 的 budget.estimate）。\n"
            f"预算：{budget}\n"
            f"范围（trsi 只接受范围内的候选）：{run_path} 的 space。\n"
            "为什么问你：训练占共享 GPU 时间，ADR 0001 规定每次训练都要人确认。\n"
            "批准：dm-rsi 运行 trsi launch（ojh run 在 tmux 里训练，立即返回），之后每轮 collect / next。\n"
            + (
                "拒绝：不再重试，这段流程失败、停下。\n"
                if str(item["key"]).startswith("retry/")
                else "拒绝：这段流程失败、停下；可以改参数（train_gpu_ids、gpu_hours）重开。\n"
            )
            + "建议：先 nvidia-smi 确认这几张卡空闲（harness 也会检查），方案的理由站得住再批。"
        ),
    )
    if a.state not in (State.APPROVED, State.DONE):
        return StepState.BLOCKED
    args = ["launch", str(run_path)]
    args += ["--start"] if item["proposal"] is None else [str(item["proposal"])]
    args += ["--seed", str(seed), "--gpus", ",".join(str(g) for g in gpus)]
    out = _trsi(ctx, f"launch-{item['label']}-s{seed}", *args)
    launched = cast("list[Value]", ctx.data.setdefault("launched_proposals", []))
    if (
        item["proposal"] is not None
        and seed == Records(run_path).seeds[0]
        and str(item["label"]) not in launched
    ):
        launched.append(str(item["label"]))
    if a.state is State.APPROVED:
        ctx.store.mark(a.id, ok=True, result={"stdout": out[-2000:]})
    return StepState.RUNNING


def _finish(ctx: StepContext, run_path: Path, why: str) -> Outcome:
    _trsi(ctx, "report", "report", str(run_path))
    rec = Records(run_path)
    inc = rec.incumbent()
    cid = str(inc["candidate_id"])
    cand = cast("dict[str, Value]", inc["candidate"])
    ckpts: dict[str, Value] = {}
    for tid, t in rec.trials.items():
        if tid.startswith(f"{cid}-s") and t.get("status") == "done":
            r = cast("dict[str, Value]", t.get("result") or {})
            ckpts[str(t["seed"])] = {
                "run_dir": t.get("run_dir"),
                "checkpoint": r.get("checkpoint"),
                "metrics": r.get("metrics"),
            }
    scheme: Payload = {
        "candidate_id": cid,
        "base": cand["base"],
        "recipe": cand["recipe"],
        "overrides": cast("Value", cand.get("overrides") or {}),
        "note": cast("Value", cand.get("note") or ""),
        "is_start": cid == rec.start_cid,
    }
    refused = cast("dict[str, Value]", ctx.data.get("refused") or {})
    if refused:
        why += f"；{len(refused)} 个候选被 trsi 的规则拒绝：{', '.join(refused)}"
    return Outcome(
        StepState.DONE,
        f"训练方案定为 `{cid}`（{cand['base']} {Path(str(cand['recipe'])).stem}）：{why}",
        outputs={
            "train_scheme": scheme,
            "train_rsi_report": str(rec.home / "report.md"),
            "train_rsi_home": str(rec.home),
            "train_checkpoints": ckpts,
        },
    )


def _propose(ctx: StepContext, rec: Records) -> list[str]:
    """判断环节：诊断当前方案、写 1–2 个候选。default 模式不提候选（只跑起点）。"""
    if str(ctx.param("judge", "default")) != "agent":
        return []
    proposals = rec.home / "proposals"
    reads = sorted(
        {str(Path(str(t["run_dir"])).parent) for t in rec.trials.values() if t.get("run_dir")}
    )
    res = agent.judge(
        ctx,
        direction=DIRECTION,
        task=(
            f"Training RSI 的 run 在 {rec.home}（report.md、archive.jsonl、incumbent.yaml），试验的 harness run 目录在 "
            f"{', '.join(reads) or '（无）'}（每个种子的 RESULT.md）。按 SKILL.md 主流程第 4–5 步：先写诊断，"
            f"再在 {proposals} 写 1–2 个候选 YAML（带 note 和 prediction，必须在 run.yaml 的 space 里）。"
            "如果剩下的问题都属于数据或评测、不值得再试训练方案，就不写文件，在回答里说明。只写候选，不运行任何命令。"
            + _refused_note(ctx)
        ),
        writes=proposals,
        reads=[rec.home, *map(Path, reads)],
        allow_empty=True,
    )
    return cast("list[str]", res.get("files") or [])


def _refused_note(ctx: StepContext) -> str:
    refused = cast("dict[str, list[str]]", ctx.data.get("refused") or {})
    if not refused:
        return ""
    lines = "；".join(f"{k}：{'、'.join(v)}" for k, v in refused.items())
    return (
        f"\n之前写的这些候选被 trsi 的规则拒绝了（没有训练，但算一次尝试），别再写同样的：{lines}"
    )


def _refresh(ctx: StepContext, run_path: Path) -> str:
    """第一次 check + init；每次 collect + next。返回 next 的输出。"""
    if not ctx.data.get("checked"):
        ojh = ["--ojh"] if ctx.param("trsi_check_ojh", True) else []
        _trsi(ctx, "check", "check", str(run_path), *ojh)
        _trsi(ctx, "init", "init", str(run_path))
        ctx.data["checked"] = True
    _trsi(ctx, "collect", "collect", str(run_path))
    nxt = _trsi(ctx, "next", "next", str(run_path))
    (ctx.dir / "next.txt").write_text(nxt, encoding="utf-8")
    return nxt


def _failed(ctx: StepContext, rec: Records) -> Outcome | None:
    """重试用完仍失败的试验 → 整段失败。还能重试的由 :func:`_retries` 排进启动列表。"""
    _, exhausted = _retries(ctx, rec)
    if not exhausted:
        return None
    return Outcome(
        StepState.FAILED,
        f"训练失败，重试用完（max_retries）：{'; '.join(exhausted)}"
        "（看 run 目录的 DONE、runner.log；修好后 dm-rsi work reset 重跑）",
    )


def _launch_what_fits(
    ctx: StepContext, run_path: Path, todo: list[dict[str, Value]], nxt: str
) -> tuple[bool, int]:
    """按顺序启动放得下的训练，每个先审批；一次只问一个。返回 (在等审批, 启动了几个)。"""
    rec = Records(run_path)
    m = _BUDGET_RE.search(nxt)
    budget = f"已花 {m.group(1)} / {m.group(2)} 卡时" if m else "（next 没有打印预算）"
    free = [g for g in _gpu_ids(rec) if g not in rec.running_gpus()]
    launched = 0
    for item in todo:
        n = _recipe_gpus(Path(str(item["recipe"])))
        if len(free) < n:
            break
        if _launch(ctx, run_path, item, free[:n], budget) is StepState.BLOCKED:
            return True, launched
        launched += 1
        free = free[n:]
    return False, launched


def _gpu_ids(rec: Records) -> list[int]:
    gpus = cast("dict[str, Value]", rec.run["gpus"])
    return [int(cast("int", g)) for g in cast("list[Value]", gpus["ids"])]


@step_kind("trsi.rsi")
def rsi(ctx: StepContext) -> Outcome:  # noqa: PLR0911 -- 每个出口对应一个流程状态
    """推进一步 Training RSI。"""
    run_path = write_configs(ctx)
    nxt = _refresh(ctx, run_path)
    rec = Records(run_path)
    failed = _failed(ctx, rec)
    if failed is not None:
        return failed
    stop = "STOP" in nxt
    todo = _launches(ctx, rec, run_path)
    if stop:  # 预算不够：只补起点的种子（不计预算），候选不再启动
        todo = [t for t in todo if t["proposal"] is None]
    blocked, launched = _launch_what_fits(ctx, run_path, todo, nxt)
    rec = Records(run_path)
    running = [tid for tid, t in rec.trials.items() if t.get("status") == "running"]
    if blocked:
        return Outcome(StepState.BLOCKED, f"等人批准下一次训练（在跑 {len(running)} 个）")
    if running:
        more = f"（本轮启动 {launched} 个）" if launched else ""
        return Outcome(StepState.RUNNING, f"在跑 {len(running)} 个试验：{', '.join(running)}{more}")
    if todo:
        return Outcome(
            StepState.FAILED,
            f"train_gpu_ids {_gpu_ids(rec)} 放不下下一个试验（配方 {Path(str(todo[0]['recipe'])).name}）：改参数重开",
        )
    if stop:
        return _finish(ctx, run_path, "预算不够再跑一个种子")
    n_props = len(cast("list[Value]", ctx.data.get("launched_proposals") or []))
    if n_props >= int(cast("int", ctx.param("max_candidates", 4))):
        return _finish(ctx, run_path, f"已试 {n_props} 个候选（max_candidates）")
    if not _propose(ctx, rec):
        return _finish(ctx, run_path, "不再提新候选")
    return Outcome(StepState.RUNNING, "写了新候选，下一轮启动探针")
