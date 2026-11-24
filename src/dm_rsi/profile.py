"""xtra-rsi profile ``dm``：dm-rsi 的 meta loop（ADR 0001，2026-11-23 修订）。

``rsi run --profile dm --iterations N --run-dir ...``（``--node root`` 时 agent 花费记到节点预算）

分工：**观察、推理、决策是 dm-rsi 自己的；行动执行决策选中的那一段具体流程**（数据、训练、打分、对比、
Decision Policy、final-test，见 :mod:`dm_rsi.flows`）。流程执行记在工作区（:mod:`dm_rsi.workspace`），
执行过程（调了哪些 sub-RSI 命令、提交了哪些作业）作为行动阶段的 ``events`` 写进本轮的 ``action.yaml``，
``rsi viz`` 面板上展开行动阶段就能看到。

- observe：工作区的产物、每次流程执行的状态、待审批、submodule、issue、上一轮的 lessons。
- reasoning：findings。还没结束的执行 → 继续它或等人；失败的流程 → 标出归属方向，停下等人处理；
  否则按 :func:`dm_rsi.flows.suggest` 指出缺什么。
- decision：``plan``，``action`` 是 ``flow``（带 ``flow``、``args``）、``resume``（带 ``step``）、
  或 ``wait``。同一时间只有一段流程在执行。dm-rsi 不给各方向起草 issue（2026-11-24，用户决定）：
  流程失败时 loop 停下，由人决定怎么处理、要不要和对应方向沟通。
- action：执行 plan；``flow`` 的名字和参数在 :data:`dm_rsi.flows.ARGS` 之外的一律拒绝。
- feedback：这段流程的结果、lessons；``wait`` 时 ``stop: true``，loop 停下等人。

版本和停止条件（:mod:`dm_rsi.versions`）：每次 ``dp_rsi`` 完成后代码判定升级 / 保留；停止条件命中后推理阶段
直接指向 ``final_test``，行动阶段拒绝其他流程（LLM 决策也绕不过去）。

工作区：``DMRSI_WORKSPACE``（默认 ``main``），模板：``DMRSI_TEMPLATE``（默认 ``synthetic``，只在新建工作区时用）。
默认（``DMRSI_META=rules``）五个阶段都是规则实现；``DMRSI_META=llm`` 时推理和决策换成 ``claude -p``
（读 ``harness/<stage>/AGENTS.md``），行动仍由代码执行并校验。
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, cast

from xtra_rsi import Profile, Stage, StageFn
from xtra_rsi.command_stage import CommandStage

from dm_rsi import backends as backends_mod
from dm_rsi import config, flows, issues, steps, submodules, versions, workspace
from dm_rsi.approvals import ApprovalStore, State
from dm_rsi.github import GhCli, GitHub, GitHubError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from xtra_rsi import IterationRecord, StageContext
    from xtra_rsi.payload import Payload, Value

_ = steps  # 导入即注册流程实现

ACTIONS = ("flow", "resume", "wait")

# 流程 → 出问题时归哪个方向（issue 发给谁）；None = dm-rsi 自己或环境的问题
OWNER: dict[str, str | None] = {
    "prepare_data": "training",
    "train_rsi": "training",
    "train_base": "training",
    "score_base": "training",
    "data_rsi": "data",
    "retrain": "training",
    "data_compare": "data",
    "dp_rsi": "decision-policy",
    "final_test": None,
}


def _cfg() -> config.Config:
    return config.load()


def _gh() -> GitHub:
    return GhCli()


def _ws(cfg: config.Config) -> workspace.Workspace:
    template = os.environ.get("DMRSI_TEMPLATE", "synthetic")
    ws_id = os.environ.get("DMRSI_WORKSPACE", "main")
    return workspace.open_or_create(cfg, ws_id, template, flows.template_params(template))


def make_backends(cfg: config.Config) -> Mapping[str, backends_mod.Backend]:
    """``dm-rsi.yaml`` 里配置的全部后端（构造时不连服务器）。"""
    out: dict[str, backends_mod.Backend] = {}
    for name, conf in cfg.backends.items():
        try:
            out[name] = backends_mod.make(name, conf, state_dir=cfg.home / "backends" / name)
        except backends_mod.BackendError:
            continue
    return out


def _carry(up: Payload) -> Payload:
    return {
        k: up[k] for k in ("lessons", "usage_offset", "workspace", "template", "pending") if k in up
    }


# ---- observe -------------------------------------------------------------------------------


_KEEP = {"arms", "models", "compared", "model", "dp_model", "base_model", "rows_dir", "dp_headline"}
_MAX_STR = 120
_MAX_LIST = 8


def _short(v: Value) -> Value:
    if isinstance(v, str) and len(v) > _MAX_STR:
        return v[: _MAX_STR - 3] + "..."
    if isinstance(v, list) and len(v) > _MAX_LIST:
        return [*v[:_MAX_LIST], f"...（共 {len(v)} 项）"]
    return v


def observe(ctx: StageContext) -> Payload:
    """收集当前状态。"""
    cfg = _cfg()
    ws = _ws(cfg)
    store = ApprovalStore(cfg.approvals_dir)
    approvals = {a.id: a for a in store.all()}
    execs: list[Value] = []
    for s in ws.steps:
        st = ws.step_state(s.name)
        aid = cast("Payload", st.get("data") or {}).get("approval")
        a = approvals.get(str(aid)) if aid else None
        execs.append(
            {
                "step": s.name,
                "flow": s.flow,
                "state": st.get("state"),
                "note": _short(cast("Value", st.get("note"))),
                "approval": {"id": a.id, "state": a.state.value} if a else None,
            }
        )
    pending = [a for a in approvals.values() if a.state is State.PENDING]
    out: Payload = {
        "workspace": ws.id,
        "template": ws.template,
        # 决策规则要用的键原样保留，其余的截短（只给人和 LLM 看）
        "artifacts": {k: v if k in _KEEP else _short(v) for k, v in ws.artifacts().items()},
        "executions": execs,
        "pending": [{"id": a.id, "kind": a.kind.value, "title": a.title} for a in pending],
        "versions": versions.summary(ws),
        "submodules": [
            s.to_dict() for s in submodules.states(cfg, online=bool(os.environ.get("DMRSI_ONLINE")))
        ],
        "lessons": cast("Value", ctx.upstream.get("lessons") or []),
        "usage_offset": cast("Value", ctx.upstream.get("usage_offset") or 0),
    }
    try:
        gh = _gh()
        out["inbox"] = [
            {"number": i.number, "title": i.title, "author": i.author}
            for i in issues.inbox(cfg, gh)
        ]
        out["outbound"] = [
            {"title": i.title, "state": i.state, "url": i.url} for i in issues.outbound(cfg, gh)
        ]
    except (GitHubError, OSError) as e:
        out["github_error"] = str(e)
    return out


# ---- reasoning -----------------------------------------------------------------------------


def _finding(kind: str, e: Payload, suggest: str, **extra: Value) -> Payload:
    return {
        "kind": kind,
        "step": e["step"],
        "flow": e["flow"],
        "evidence": e.get("note"),
        "suggest": suggest,
        **extra,
    }


def _open_findings(execs: list[Payload]) -> list[Value]:
    out: list[Value] = []
    for e in execs:
        state = e.get("state")
        if state in ("pending", "running"):
            out.append(_finding("open_execution", e, "resume"))
        elif state == "blocked":
            a = cast("Payload | None", e.get("approval"))
            if a is not None and a.get("state") == "pending":
                out.append(_finding("waiting_human", e, "wait"))
            else:
                # 审批已经有结果：继续推进，让流程自己处理批准 / 拒绝
                out.append(_finding("open_execution", e, "resume"))
    return out


def reasoning(ctx: StageContext) -> Payload:
    """整理 findings，每条带建议动作。"""
    up = ctx.upstream
    execs = cast("list[Payload]", up.get("executions") or [])
    findings = _open_findings(execs)
    latest: dict[str, Payload] = {}
    for e in execs:
        latest[str(e["flow"])] = e
    failed = [e for e in latest.values() if e.get("state") == "failed"]
    for e in failed:
        owner = OWNER.get(str(e["flow"]))
        findings.append(_finding("flow_failed", e, "wait", owner=owner))
    nxt = flows.suggest(
        cast("Payload", up.get("artifacts") or {}), cast("str | None", up.get("template"))
    )
    stop = cast("Payload", up.get("versions") or {}).get("stop")
    arts = cast("Payload", up.get("artifacts") or {})
    if stop and "dp_headline" not in arts and (nxt is None or nxt[0] != "final_test"):
        # 停止条件命中：不再探索，直接去 final-test（测当前最佳版本）
        nxt = ("final_test", {}, f"停止条件：{stop}；final-test 测当前最佳版本")
    if nxt is not None:
        flow, args, why = nxt
        if flow in {str(e["flow"]) for e in failed}:
            findings.append(
                {
                    "kind": "blocked_by_failure",
                    "flow": flow,
                    "evidence": f"下一段应该是 {flow}（{why}），但它上次失败了，等人处理（dm-rsi work reset）",
                    "suggest": "wait",
                }
            )
        else:
            findings.append(
                {
                    "kind": "next_flow",
                    "flow": flow,
                    "args": args,
                    "evidence": why,
                    "suggest": "flow",
                }
            )
    elif not any(e.get("state") in ("pending", "running", "blocked") for e in execs):
        findings.append(
            {
                "kind": "finished",
                "evidence": "工作区的流程都完成了（含 final-test）",
                "suggest": "wait",
            }
        )
    for sm in cast("list[Payload]", up.get("submodules") or []):
        if sm.get("dirty"):
            findings.append(
                {
                    "kind": "submodule_dirty",
                    "path": sm["path"],
                    "evidence": sm["dirty"],
                    "suggest": "wait",
                }
            )
        if sm.get("behind"):
            evidence = f"{sm['pinned']} → {sm['upstream']}"
            findings.append(
                {
                    "kind": "submodule_behind",
                    "path": sm["path"],
                    "evidence": evidence,
                    "suggest": "wait",
                }
            )
    findings.extend(
        {
            "kind": "inbox_issue",
            "evidence": f"#{i['number']} {i['title']}（{i['author']}）",
            "suggest": "wait",
        }
        for i in cast("list[Payload]", up.get("inbox") or [])
    )
    return {**_carry(up), "findings": findings}


# ---- decision ------------------------------------------------------------------------------


def decision(ctx: StageContext) -> Payload:
    """选一件事：先把进行中的流程推进完（同一时间只执行一段），再处理失败，再开始下一段流程。"""
    up = ctx.upstream
    findings = cast("list[Payload]", up.get("findings") or [])
    by_kind: dict[str, list[Payload]] = {}
    for f in findings:
        by_kind.setdefault(str(f.get("kind")), []).append(f)
    fails = by_kind.get("flow_failed", [])
    plan: Payload
    if by_kind.get("open_execution"):
        f = by_kind["open_execution"][0]
        plan = {
            "action": "resume",
            "step": f["step"],
            "why": f"继续 {f['step']}：{f.get('evidence')}",
        }
    elif by_kind.get("waiting_human"):
        f = by_kind["waiting_human"][0]
        plan = {"action": "wait", "why": f"{f['step']} 等人：{f.get('evidence')}"}
    elif fails:
        # 失败了就停下等人：dm-rsi 不起草 issue，也不绕过失败去跑别的流程
        f = fails[0]
        owner = f"，归 {f['owner']} 方向" if f.get("owner") else "，dm-rsi 自己或环境的问题"
        plan = {
            "action": "wait",
            "why": f"{f['flow']} 失败（{f['step']}{owner}）：{f.get('evidence')}。"
            f"看 $DMRSI_HOME/work/{up.get('workspace')}/{f['step']}/ 的日志，"
            f"处理后 dm-rsi work reset {up.get('workspace')} {f['step']} 重来",
        }
    elif by_kind.get("next_flow"):
        f = by_kind["next_flow"][0]
        plan = {
            "action": "flow",
            "flow": f["flow"],
            "args": f.get("args") or {},
            "why": f.get("evidence"),
        }
    else:
        kinds = ("finished", "blocked_by_failure", "flow_failed")
        reasons = [str(f.get("evidence")) for f in findings if f.get("kind") in kinds]
        plan = {"action": "wait", "why": "；".join(reasons) or "没有可做的事"}
    return {**_carry(up), "plan": plan}


# ---- action --------------------------------------------------------------------------------


def _usage_since(ws: workspace.Workspace, offset: int) -> tuple[list[Value], int]:
    f = ws.dir / "usage.jsonl"
    if not f.is_file():
        return [], offset
    lines = f.read_text().splitlines()
    new = [json.loads(line) for line in lines[offset:] if line.strip()]
    return [{"cost_usd": float(e.get("cost_usd") or 0.0)} for e in new], len(lines)


def _run_flow(plan: Payload, up: Payload, out: Payload) -> Payload:
    cfg = _cfg()
    ws = _ws(cfg)
    if plan.get("action") == "flow":
        try:
            kind, params = flows.resolve(
                ws.template, str(plan.get("flow")), cast("Payload", plan.get("args") or {})
            )
        except flows.FlowError as e:
            return {**out, "result": {"error": str(e)}}
        if ws.open_steps():
            msg = f"还有没结束的执行 {[s.name for s in ws.open_steps()]}，同一时间只执行一段流程"
            return {**out, "result": {"error": msg}}
        stop = versions.stop_reason(ws)
        if stop and plan.get("flow") != "final_test":
            msg = f"停止条件已命中（{stop}），只能执行 final_test，拒绝 {plan.get('flow')}"
            return {**out, "result": {"error": msg}}
        name = ws.add(str(plan["flow"]), kind, params).name
    else:
        name = str(plan.get("step"))
        if ws.step_state(name).get("state") == "running":
            time.sleep(float(os.environ.get("DMRSI_POLL_S", "300")))  # 等后台作业，不空转
    res = ws.run_step(name, ApprovalStore(cfg.approvals_dir), make_backends(cfg))
    usage, offset = _usage_since(ws, int(cast("int", up.get("usage_offset") or 0)))
    out = {**out, "result": res, "events": res.get("events") or [], "usage_offset": offset}
    if usage:
        out["_usage"] = usage
    return out


def action(ctx: StageContext) -> Payload:
    """执行 plan。行动 = 执行一段具体流程（或等待）。"""
    up = ctx.upstream
    plan = cast("Payload", up.get("plan") or {})
    act = str(plan.get("action"))
    out: Payload = {**_carry(up), "action": act}
    if act not in ACTIONS:
        return {**out, "result": {"error": f"动作 {act!r} 不在动作表 {ACTIONS} 里，拒绝执行"}}
    if act in ("flow", "resume"):
        return _run_flow(plan, up, out)
    waiting = [f"{p['id']} {p['title']}" for p in cast("list[Payload]", up.get("pending") or [])]
    return {**out, "result": {"why": plan.get("why"), "waiting": cast("Value", waiting)}}


# ---- feedback ------------------------------------------------------------------------------


def feedback(ctx: StageContext) -> Payload:
    """评估本轮，记下 lessons；``wait`` 时停下等人。"""
    up = ctx.upstream
    act = str(up.get("action"))
    res = cast("Payload", up.get("result") or {})
    lessons = [str(x) for x in cast("list[Value]", up.get("lessons") or [])]
    if act == "wait":
        verdict = "waiting"
    elif "error" in res:
        verdict = "refused"
    elif act in ("flow", "resume"):
        verdict = str(res.get("state"))
    else:
        verdict = act
    return {
        "lessons": cast("Value", lessons[-50:]),
        "usage_offset": up.get("usage_offset") or 0,
        "verdict": verdict,
        "flows_done": 1 if res.get("state") == "done" else 0,
        "last": cast("Value", {"action": act, "result": res}),
        "stop": act == "wait",
        "reason": str(res.get("why")) if act == "wait" else None,
    }


# ---- profile -------------------------------------------------------------------------------


def _flows_done(records: Sequence[IterationRecord]) -> float:
    return float(sum(int(cast("int", r.last.get("flows_done") or 0)) for r in records))


def profile() -> Profile:
    """``rsi --profile dm``。"""
    stages: dict[Stage, StageFn] = {
        Stage.OBSERVE: observe,
        Stage.REASONING: reasoning,
        Stage.DECISION: decision,
        Stage.ACTION: action,
        Stage.FEEDBACK: feedback,
    }
    if os.environ.get("DMRSI_META") == "llm":
        root = config.find_root()
        for s in (Stage.REASONING, Stage.DECISION):
            stages[s] = CommandStage(
                ["python", "-m", "dm_rsi.stage_llm"],
                agents_md=root / "harness" / s.value / "AGENTS.md",
                timeout=900,
            )
    return Profile(
        name="dm",
        stages=stages,
        seed={},
        metrics={"flows_done": _flows_done},
        goal={
            "target": f"DM-RSI 端到端流程（工作区 {os.environ.get('DMRSI_WORKSPACE', 'main')}）",
            "metric": "完成的流程段数；完成后看 val / test 上的 f1_e2e",
        },
    )
