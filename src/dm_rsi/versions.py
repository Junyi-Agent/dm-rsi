"""版本登记和停止条件（ADR 0001，2026-11-24 第二次修订，借鉴 Borui 的 ``rsi promote`` / ``rsi next``）。

一个工作区里，每完成一次 ``dp_rsi`` 就得到一个端到端的候选版本：训练方案 + Data 对比选出的模型 + Decision
Policy 的 incumbent，带 val 上的目标值（dpolicy 的 ``objective``，默认 f1_e2e）。代码按固定规则判定它是否升级：

- 还没有版本：升级为 V1。
- 否则目标值比当前最佳严格高出 ``stop.min_delta`` 才升级为 V<n+1>，不然记为保留（keep），连续保留数 +1。

停止条件（任一成立，:func:`stop_reason`）：已评估 ``stop.max_rounds`` 个候选版本；连续 ``stop.patience`` 个没有升级。
停止后，行动阶段只接受 ``final_test``（:mod:`dm_rsi.profile`），final-test 测的是**当前最佳版本**，不是最近一次。

记录在 ``$DMRSI_HOME/work/<id>/versions.yaml``，只由代码写，不手改。判定只用 val；val 被反复用来选择，
数会偏乐观（见 ADR「待定：val 和评测集分开」），结论以 final-test 为准。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from dm_rsi import yamlfile
from dm_rsi.steps import handoff

if TYPE_CHECKING:
    from xtra_rsi.payload import Payload, Value

    from dm_rsi.workspace import Workspace

DEFAULT_STOP: Payload = {"min_delta": 0.0, "patience": 2, "max_rounds": 4}


def path(ws: Workspace) -> Path:
    """``versions.yaml``。"""
    return ws.dir / "versions.yaml"


def load(ws: Workspace) -> Payload:
    """读登记表；没有时是空表。"""
    p = path(ws)
    if not p.is_file():
        return {"best": None, "streak": 0, "entries": []}
    return yamlfile.read(p)


def settings(ws: Workspace) -> Payload:
    """生效的停止设置：工作区参数 ``stop`` 覆盖默认值。"""
    return {**DEFAULT_STOP, **cast("Payload", ws.params.get("stop") or {})}


def dp_incumbent(dp_dir: Path) -> Payload | None:
    """Decision Policy 第 0 个副本的 incumbent（经 :mod:`dm_rsi.steps.handoff`）。"""
    inc = handoff.dpolicy_incumbent(dp_dir / "r0")
    if inc is None:
        return None
    return {
        "name": inc.name,
        "hash": inc.hash,
        "objective": inc.objective,
        "is_baseline": inc.is_baseline,
        "evaluated": inc.evaluated,
        "source": inc.source,
    }


def register(ws: Workspace, step: str, outputs: Payload) -> Payload:
    """登记一次 ``dp_rsi`` 的结果并判定升级 / 保留；同一次执行只登记一次（幂等），返回这条记录。"""
    reg = load(ws)
    entries = cast("list[Payload]", reg["entries"])
    for e in entries:
        if e["step"] == step:
            return e
    inc = dp_incumbent(Path(str(outputs["dp_dir"])))
    if inc is None:
        msg = f"{step}：{outputs['dp_dir']}/r0 还没有 incumbent（dpolicy 没 init？），不能登记版本"
        raise ValueError(msg)
    objective = float(cast("float", inc["objective"]))
    min_delta = float(cast("float", settings(ws)["min_delta"]))
    best = _entry(reg, cast("str | None", reg.get("best")))
    art = ws.artifacts()
    if best is None:
        decision, reason = "promote", "还没有版本，第一个端到端结果直接成为 V1"
    else:
        best_obj = float(cast("float", best["objective"]))
        delta = objective - best_obj
        if delta > min_delta:
            decision = "promote"
            reason = f"val 目标值 {objective:.4f} 比当前最佳 {best['version']} 的 {best_obj:.4f} 高 {delta:+.4f}（> min_delta {min_delta}）"
        else:
            decision = "keep"
            reason = f"val 目标值 {objective:.4f} 比当前最佳 {best['version']} 的 {best_obj:.4f} 差 {delta:+.4f}（要 > min_delta {min_delta}），保留 {best['version']}"
    n_versions = sum(1 for e in entries if e.get("version"))
    entry: Payload = {
        "round": len(entries) + 1,
        "step": step,
        "decision": decision,
        "version": f"V{n_versions + 1}" if decision == "promote" else None,
        "reason": reason,
        "objective": objective,
        "incumbent": cast("Value", inc),
        "dp_model": outputs["dp_model"],
        "dp_dir": outputs["dp_dir"],
        "dp_configs": outputs["dp_configs"],
        "consensus": outputs.get("consensus"),
        "model_key": cast("Value", art.get("model_key")),
        "train_scheme": cast(
            "Value", cast("Payload", art.get("train_scheme") or {}).get("candidate_id")
        ),
        "at": yamlfile.now(),
    }
    entries.append(entry)
    if decision == "promote":
        reg["best"] = entry["version"]
        reg["streak"] = 0
    else:
        reg["streak"] = int(cast("int", reg.get("streak") or 0)) + 1
    yamlfile.write(path(ws), reg)
    return entry


def _entry(reg: Payload, version: str | None) -> Payload | None:
    if version is None:
        return None
    return next(
        (e for e in cast("list[Payload]", reg["entries"]) if e.get("version") == version), None
    )


def best(ws: Workspace) -> Payload | None:
    """当前最佳版本的记录。"""
    reg = load(ws)
    return _entry(reg, cast("str | None", reg.get("best")))


def stop_reason(ws: Workspace) -> str | None:
    """停止条件命中时返回理由。"""
    reg = load(ws)
    s = settings(ws)
    rounds = len(cast("list[Value]", reg["entries"]))
    streak = int(cast("int", reg.get("streak") or 0))
    if rounds >= int(cast("int", s["max_rounds"])):
        return f"已评估 {rounds} 个候选版本（stop.max_rounds = {s['max_rounds']}）"
    if streak >= int(cast("int", s["patience"])):
        return f"连续 {streak} 个候选版本没有升级（stop.patience = {s['patience']}）"
    return None


def summary(ws: Workspace) -> Payload:
    """给观察阶段和 ``dm-rsi status`` 看的摘要。"""
    reg = load(ws)
    b = best(ws)
    return {
        "best": None
        if b is None
        else {"version": b["version"], "objective": b["objective"], "dp_model": b["dp_model"]},
        "rounds": len(cast("list[Value]", reg["entries"])),
        "streak": reg.get("streak") or 0,
        "settings": settings(ws),
        "stop": stop_reason(ws),
        "history": [
            f"第 {e['round']} 轮 {e['step']}：{e['decision']}"
            + (f" → {e['version']}" if e.get("version") else "")
            + f"（{float(cast('float', e['objective'])):.4f}）"
            for e in cast("list[Payload]", reg["entries"])
        ],
    }
