"""sub-RSI → dm-rsi 的交接（ADR 0001，2026-11-24 第三次修订）。

**dm-rsi 只在这个模块里读 sub-RSI 写的文件。** 其他模块要用 sub-RSI 的结果，调这里的函数，拿到带类型的对象；
``tests/test_handoff.py`` 会检查别的模块没有直接读 sub-RSI 的内部文件名。

每个交接有两个来源：

1. **正式交接文件** ``handoff.yaml``（格式见 ``docs/handoff.md``，``handoff: 1``）：各方向在自己的运行目录里写，
   dm-rsi 只认它的字段，不关心内部记录怎么存。存在就用它。
2. **兼容读取**：还没有正式文件时，从 sub-RSI 现在的内部记录（``archive.jsonl``、``incumbent.yaml``、
   ``compare.json``……）推出同样的对象。内部格式一变，只有这里要改。

两种来源都过同一套字段检查（:func:`need`），缺字段或类型不对时报 :class:`ContractError`，写明文件、字段、
实际拿到了什么。流程失败后按「契约不符转成 issue」处理，不在 dm-rsi 里悄悄修补。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from dm_rsi.steps.contracts import ContractError

if TYPE_CHECKING:
    from xtra_rsi.payload import Value

HANDOFF = "handoff.yaml"
VERSION = 1

_NUM = (int, float)
_OPT_STR = (str, type(None))
_TYPE_NAMES = {
    str: "字符串",
    int: "整数",
    float: "小数",
    bool: "布尔",
    dict: "映射",
    list: "列表",
    type(None): "空",
}


def _tname(t: type | tuple[type, ...]) -> str:
    ts = t if isinstance(t, tuple) else (t,)
    return " 或 ".join(_TYPE_NAMES.get(x, x.__name__) for x in ts)


def need(obj: object, where: str, **fields: type | tuple[type, ...]) -> dict[str, Value]:
    """``obj`` 是映射且有这些字段、类型对；否则报 :class:`ContractError`（列出全部问题）。"""
    if not isinstance(obj, dict):
        msg = f"{where}：应该是映射，实际是 {type(obj).__name__}"
        raise ContractError(msg)
    d = cast("dict[str, Value]", obj)
    bad: list[str] = []
    for k, t in fields.items():
        if k not in d:
            if not (isinstance(t, tuple) and type(None) in t):
                bad.append(f"缺字段 {k}")
            continue
        v = d[k]
        ok = isinstance(v, t) and not (isinstance(v, bool) and bool not in _as_tuple(t))
        if not ok:
            bad.append(f"{k} 应该是{_tname(t)}，实际是 {type(v).__name__}（{str(v)[:60]}）")
    if bad:
        msg = f"{where}：" + "；".join(bad)
        raise ContractError(msg)
    return d


def _as_tuple(t: type | tuple[type, ...]) -> tuple[type, ...]:
    return t if isinstance(t, tuple) else (t,)


def _yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        msg = f"读 {path} 失败：{e}"
        raise ContractError(msg) from e


def _jsonl(path: Path) -> list[dict[str, Value]]:
    if not path.is_file():
        return []
    out: list[dict[str, Value]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(need(json.loads(line), f"{path} 第 {i} 行"))
        except json.JSONDecodeError as e:
            msg = f"{path} 第 {i} 行不是 JSON：{e}"
            raise ContractError(msg) from e
    return out


def _json(path: Path) -> dict[str, Value]:
    try:
        return need(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, json.JSONDecodeError) as e:
        msg = f"读 {path} 失败：{e}"
        raise ContractError(msg) from e


def _official(run_dir: Path, kind: str) -> dict[str, Value] | None:
    """``<run_dir>/handoff.yaml``；没有就 None。版本和种类不对时报错。"""
    p = run_dir / HANDOFF
    if not p.is_file():
        return None
    d = need(_yaml(p), str(p), handoff=int, kind=str)
    if d["handoff"] != VERSION or d["kind"] != kind:
        msg = f"{p}：dm-rsi 认 handoff: {VERSION}、kind: {kind}，文件里是 handoff: {d['handoff']}、kind: {d['kind']}"
        raise ContractError(msg)
    return d


# ---- Training（trsi）---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainScheme:
    """Training RSI 选出的训练方案和它各种子的 checkpoint。"""

    candidate_id: str
    base: str
    recipe: str
    overrides: dict[str, Value]
    note: str
    is_start: bool
    checkpoints: dict[str, dict[str, Value]]
    source: str
    """``handoff.yaml`` 或 ``compat``（从内部记录推出）。"""

    def to_scheme(self) -> dict[str, Value]:
        """流程产物 ``train_scheme``。"""
        return {
            "candidate_id": self.candidate_id,
            "base": self.base,
            "recipe": self.recipe,
            "overrides": cast("Value", self.overrides),
            "note": self.note,
            "is_start": self.is_start,
        }


_TRIAL_EVENTS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "use": {"trial": str},
    "launch": {"trial": str, "gpus": list},
    "incumbent": {"candidate": str},
    "propose": {"candidate": str, "proposal": str, "candidate_dict": dict},
    "probe": {"candidate": str, "against": str, "verdict": str},
    "verdict": {"candidate": str},
}


class TrsiRecords:
    """trsi 一个 run 的记录快照：``<out_dir>/<run_name>/`` 和共享的 ``<out_dir>/trials/``（只读）。

    trainrsi 编排训练要用 trsi 的过程记录（事件、试验状态）；这里检查每类事件用到的字段，
    trsi 改了记录格式时报清楚的契约错误，而不是在编排逻辑深处 KeyError。
    """

    def __init__(self, run: dict[str, Value], run_path: Path) -> None:
        """``run`` 是 run.yaml 的内容。"""
        need(run, str(run_path), out_dir=str, run_name=str)
        self.run: dict[str, Value] = run
        self.out = Path(str(run["out_dir"]))
        self.home = self.out / str(run["run_name"])
        self.events: list[dict[str, Value]] = _jsonl(self.home / "archive.jsonl")
        for i, e in enumerate(self.events, 1):
            kind = str(e.get("event"))
            if kind in _TRIAL_EVENTS:
                need(e, f"{self.home}/archive.jsonl 第 {i} 个事件（{kind}）", **_TRIAL_EVENTS[kind])
        self.trials: dict[str, dict[str, Value]] = {}
        for e in self.of("use"):
            tid = str(e["trial"])
            t = self.out / "trials" / f"{tid}.json"
            if t.is_file() and tid not in self.trials:
                self.trials[tid] = need(_json(t), str(t), status=str, seed=int)

    def of(self, kind: str) -> list[dict[str, Value]]:
        """某类事件。"""
        return [e for e in self.events if e.get("event") == kind]

    @property
    def report(self) -> Path:
        """Trsi 给人看的报告。"""
        return self.home / "report.md"

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
        """当前方案（``incumbent.yaml``）。"""
        p = self.home / "incumbent.yaml"
        d = need(_yaml(p), str(p), candidate_id=str, candidate=dict)
        need(d["candidate"], f"{p} 的 candidate", base=str, recipe=str)
        return d

    def running_gpus(self) -> set[int]:
        """本 run 里还在跑的试验占着的卡。"""
        busy: set[int] = set()
        launched = {str(e["trial"]): e for e in self.of("launch")}
        for tid, rec in self.trials.items():
            if rec.get("status") == "running" and tid in launched:
                busy |= {int(cast("int", g)) for g in cast("list[Value]", launched[tid]["gpus"])}
        return busy

    def scheme(self) -> TrainScheme:
        """结束时的交接：优先 ``<home>/handoff.yaml``，否则从 incumbent 和试验记录推出。"""
        off = _official(self.home, "training")
        if off is not None:
            return _scheme_from_official(off, self.home / HANDOFF)
        inc = self.incumbent()
        cid = str(inc["candidate_id"])
        cand = cast("dict[str, Value]", inc["candidate"])
        ckpts: dict[str, dict[str, Value]] = {}
        for tid, t in self.trials.items():
            if tid.startswith(f"{cid}-s") and t.get("status") == "done":
                r = need(t.get("result") or {}, f"试验 {tid} 的 result", checkpoint=_OPT_STR)
                ckpts[str(t["seed"])] = {
                    "run_dir": t.get("run_dir"),
                    "checkpoint": r.get("checkpoint"),
                    "metrics": r.get("metrics"),
                }
        return TrainScheme(
            candidate_id=cid,
            base=str(cand["base"]),
            recipe=str(cand["recipe"]),
            overrides=cast("dict[str, Value]", cand.get("overrides") or {}),
            note=str(cand.get("note") or ""),
            is_start=cid == self.start_cid,
            checkpoints=ckpts,
            source="compat",
        )


def _scheme_from_official(d: dict[str, Value], p: Path) -> TrainScheme:
    need(d, str(p), scheme=dict, checkpoints=dict)
    s = need(
        d["scheme"],
        f"{p} 的 scheme",
        candidate_id=str,
        base=str,
        recipe=str,
        overrides=dict,
        is_start=bool,
        note=_OPT_STR,
    )
    ckpts: dict[str, dict[str, Value]] = {}
    for seed, c in cast("dict[str, Value]", d["checkpoints"]).items():
        ckpts[str(seed)] = need(
            c, f"{p} 的 checkpoints.{seed}", run_dir=str, checkpoint=str, metrics=(dict, type(None))
        )
    for k, o in cast("dict[str, Value]", s["overrides"]).items():
        need(o, f"{p} 的 scheme.overrides.{k}", value=(str, int, float, bool), reason=str)
    return TrainScheme(
        candidate_id=str(s["candidate_id"]),
        base=str(s["base"]),
        recipe=str(s["recipe"]),
        overrides=cast("dict[str, Value]", s["overrides"]),
        note=str(s.get("note") or ""),
        is_start=bool(s["is_start"]),
        checkpoints=ckpts,
        source=HANDOFF,
    )


# ---- Data（dgen）-------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """Data RSI 生成的一组训练数据（一个配方）。"""

    hash: str
    name: str
    stats: dict[str, Value] = field(default_factory=dict[str, "Value"])


def dgen_arms(out: Path) -> list[Arm]:
    """``generate`` 产出的配方（按生成顺序）。"""
    arch = out / "archive.jsonl"
    arms: list[Arm] = []
    for i, r in enumerate(_jsonl(arch), 1):
        if r.get("source") != "recipe":
            continue
        need(r, f"{arch} 第 {i} 条（recipe）", hash=str, name=str, stats=(dict, type(None)))
        arms.append(
            Arm(str(r["hash"]), str(r["name"]), cast("dict[str, Value]", r.get("stats") or {}))
        )
    return arms


@dataclass(frozen=True)
class Build:
    """``build`` 拼好的一个训练文件。"""

    train_path: Path
    rows: int | None


def dgen_build(tdir: Path) -> Build | None:
    """``<out>/train/<name>/``：还没 build 时 None。"""
    rep_path = tdir / "build_report.json"
    if not rep_path.is_file():
        return None
    rep = need(_json(rep_path), str(rep_path), rows=(int, type(None)))
    train = next((p for p in tdir.iterdir() if p.name.startswith("train.")), None)
    if train is None:
        msg = f"{tdir} 有 build_report.json，但没有 train.* 文件"
        raise ContractError(msg)
    return Build(train, cast("int | None", rep.get("rows")))


@dataclass(frozen=True)
class Diff:
    """对比里一个模型、一个指标相对基线的差。"""

    run: str
    metric: str
    diff: float
    lo: float | None
    hi: float | None
    passes_accept: bool


@dataclass(frozen=True)
class Compare:
    """``dgen compare`` 的结果。"""

    path: Path
    primary: str
    n: int
    n_pos: int
    table: dict[str, dict[str, Value]]
    """模型 → 各指标（含 threshold）。"""
    diffs: list[Diff]


def compare_path(out: Path, tag: str) -> Path:
    """一次对比的结果文件（存在说明这个 tag 已经对比过）。"""
    return out / "compare" / tag / "compare.json"


def dgen_compare(out: Path, tag: str) -> Compare:
    """读一次对比。"""
    p = compare_path(out, tag)
    d = need(_json(p), str(p), result=dict, primary=(str, type(None)))
    res = need(d["result"], f"{p} 的 result", n=int, n_pos=int, table=list, diffs=list)
    primary = str(d.get("primary") or "auprc")
    table: dict[str, dict[str, Value]] = {}
    for i, row in enumerate(cast("list[Value]", res["table"])):
        r = need(row, f"{p} 的 result.table[{i}]", run=str)
        table[str(r["run"])] = r
    diffs: list[Diff] = []
    for i, x in enumerate(cast("list[Value]", res["diffs"])):
        dd = need(
            x,
            f"{p} 的 result.diffs[{i}]",
            run=str,
            metric=str,
            diff=_NUM,
            lo=(int, float, type(None)),
            hi=(int, float, type(None)),
        )
        if dd["metric"] == primary:
            need(dd, f"{p} 的 result.diffs[{i}]（主指标 {primary}）", passes_accept=bool)
        diffs.append(
            Diff(
                run=str(dd["run"]),
                metric=str(dd["metric"]),
                diff=float(cast("float", dd["diff"])),
                lo=None if dd["lo"] is None else float(cast("float", dd["lo"])),
                hi=None if dd["hi"] is None else float(cast("float", dd["hi"])),
                passes_accept=bool(dd.get("passes_accept")),
            )
        )
    return Compare(
        p, primary, int(cast("int", res["n"])), int(cast("int", res["n_pos"])), table, diffs
    )


def dgen_final_test(out: Path) -> Path | None:
    """``dgen final-test`` 的结果文件（没跑过时 None）。"""
    return (
        next((out / "final_test").rglob("final_test.json"), None)
        if (out / "final_test").is_dir()
        else None
    )


# ---- Decision Policy（dpolicy）-----------------------------------------------------------------------------


@dataclass(frozen=True)
class Incumbent:
    """Decision Policy 一个 run 当前选中的策略。"""

    name: str
    hash: str
    objective: float
    is_baseline: bool
    evaluated: int
    """评估过的候选数（不含基线）。"""
    baseline_objective: float | None
    accepted: int
    """被接受的候选数。"""
    diff_vs_previous: float | None
    """最后一次接受时相对前一个 incumbent 的差。"""
    source: str


def dpolicy_incumbent(run_dir: Path) -> Incumbent | None:
    """``<dp_dir>/r<k>``：优先 ``handoff.yaml``，否则从 ``archive.jsonl`` 推出；还没 init 时 None。"""
    off = _official(run_dir, "decision-policy")
    if off is not None:
        p = run_dir / HANDOFF
        need(off, str(p), incumbent=dict, baseline=dict, evaluated=int, accepted=int)
        inc = need(off["incumbent"], f"{p} 的 incumbent", name=str, hash=str, objective=_NUM)
        base = need(off["baseline"], f"{p} 的 baseline", objective=_NUM)
        return Incumbent(
            name=str(inc["name"]),
            hash=str(inc["hash"]),
            objective=float(cast("float", inc["objective"])),
            is_baseline=int(cast("int", off["accepted"])) == 0,
            evaluated=int(cast("int", off["evaluated"])),
            baseline_objective=float(cast("float", base["objective"])),
            accepted=int(cast("int", off["accepted"])),
            diff_vs_previous=cast("float | None", inc.get("diff_vs_previous")),
            source=HANDOFF,
        )
    arch = run_dir / "archive.jsonl"
    recs = _jsonl(arch)
    if not recs:
        return None
    for i, r in enumerate(recs, 1):
        need(r, f"{arch} 第 {i} 条", name=str, hash=str, objective=_NUM, source=(str, type(None)))
    base = next((r for r in recs if r.get("source") == "baseline"), None)
    acc = [r for r in recs if r.get("accepted") and r.get("source") != "baseline"]
    inc = acc[-1] if acc else base
    if inc is None:
        msg = f"{arch}：既没有基线策略，也没有被接受的候选"
        raise ContractError(msg)
    vs = cast("dict[str, Value]", inc.get("vs_incumbent") or {})
    return Incumbent(
        name=str(inc["name"]),
        hash=str(inc["hash"]),
        objective=float(cast("float", inc["objective"])),
        is_baseline=not acc,
        evaluated=len(recs) - 1,
        baseline_objective=None if base is None else float(cast("float", base["objective"])),
        accepted=len(acc),
        diff_vs_previous=cast("float | None", vs.get("diff")),
        source="compat",
    )


def dpolicy_diagnosis(run_dir: Path) -> Path:
    """当前 incumbent 的诊断（写提案时给判断环节读）。

    不能取 archive 的最后一条：那是最后评估的候选，常常没被接受，诊断的不是当前策略。
    """
    inc = dpolicy_incumbent(run_dir)
    if inc is None:
        msg = f"{run_dir}：还没有 incumbent（dpolicy init 之后应该至少有基线策略）"
        raise ContractError(msg)
    p = run_dir / "eval" / inc.hash / "diagnosis.md"
    if not p.is_file():
        msg = f"{p} 不存在：incumbent {inc.name}（{inc.hash}）没有诊断"
        raise ContractError(msg)
    return p


def dpolicy_initialised(run_dir: Path) -> bool:
    """``dpolicy init`` 是否已经做过。"""
    return (run_dir / "incumbent.yaml").is_file()


def dpolicy_consensus(dp_dir: Path) -> Path | None:
    """多个副本的共识策略（没有候选过半通过时 None）。"""
    p = dp_dir / "r0" / "consensus.yaml"
    return p if p.is_file() else None


def dpolicy_report(run_dir: Path) -> Path:
    """Dpolicy 给人看的报告。"""
    return run_dir / "report.md"


def dpolicy_final_test_path(run_dir: Path) -> Path:
    """``dpolicy final-test`` 的结果文件。"""
    return run_dir / "final_test.json"


def dpolicy_final_test(run_dir: Path) -> dict[str, dict[str, Value]]:
    """final-test 的结果：策略名 → {objective, vs_baseline}。"""
    p = dpolicy_final_test_path(run_dir)
    d = need(_json(p), str(p), policies=dict)
    out: dict[str, dict[str, Value]] = {}
    for k, v in cast("dict[str, Value]", d["policies"]).items():
        pol = need(v, f"{p} 的 policies.{k}", objective=(int, float, type(None)))
        out[str(k)] = {"objective": pol.get("objective"), "vs_baseline": pol.get("vs_baseline")}
    return out
