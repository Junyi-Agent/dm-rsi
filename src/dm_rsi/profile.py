"""xtra-rsi profile ``dm``：dm-rsi 的 meta loop（ADR 0001「Meta loop」，P3）。

``rsi run --profile dm --node root --iterations N --run-dir ...``

每一轮：

- observe：流程状态、待审批、submodule、issue 收件箱和已发出的 issue、上一轮的 lessons。
- reasoning：把观察整理成 findings（步骤失败、等审批、等作业、submodule 落后或被写脏、新 issue……），
  每条带证据和建议动作。
- decision：从固定动作表里选**一件**事：``advance``（推进某个流程）、``new_run``（没有流程时按模板新建）、
  ``draft_issue``（把归属某个方向的失败起草成 issue，进审批队列）、``wait``（只剩人要做的事）。
- action：执行。对外的动作只会产生审批请求；动作不在动作表里时拒绝执行。
- feedback：本轮推进了几步、新增了哪些审批、哪些流程完成及其 final-test 结果；没有可做的事时 ``stop: true``，
  loop 停在审批点等人。

默认（``DMRSI_META=rules``）五个阶段都是确定性的规则实现。``DMRSI_META=llm`` 时 reasoning 和 decision
换成 ``claude -p``（``CommandStage``，读 ``harness/<stage>/AGENTS.md``），action 仍然由代码执行并校验动作表。
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, cast

from xtra_rsi import Profile, Stage, StageFn
from xtra_rsi.command_stage import CommandStage

from dm_rsi import config, issues, pipeline, submodules, templates
from dm_rsi.approvals import ApprovalStore, State
from dm_rsi.github import GhCli, GitHub, GitHubError
from dm_rsi.pipeline_cli import make_backends

if TYPE_CHECKING:
    from collections.abc import Sequence

    from xtra_rsi import IterationRecord, StageContext
    from xtra_rsi.payload import Payload, Value

ACTIONS = ("advance", "new_run", "draft_issue", "wait")

# 步骤种类 → 出问题时归哪个方向（issue 发给谁）
OWNER = {
    "dgen.rsi": "data",
    "dgen.compare": "data",
    "dpolicy.rsi": "decision-policy",
    "elliptic.train": "training",
    "elliptic.score": "training",
    "elliptic.retrain": "training",
}


def _cfg() -> config.Config:
    return config.load()


def _gh() -> GitHub:
    return GhCli()


def _usage_since(run: pipeline.Run, offset: int) -> tuple[list[Value], int]:
    f = run.dir / "usage.jsonl"
    if not f.is_file():
        return [], offset
    lines = f.read_text().splitlines()
    new = [json.loads(line) for line in lines[offset:] if line.strip()]
    return [{"cost_usd": float(e.get("cost_usd") or 0.0)} for e in new], len(lines)


# ---- observe -------------------------------------------------------------------------------


def observe(ctx: StageContext) -> Payload:
    """收集当前状态。"""
    cfg = _cfg()
    store = ApprovalStore(cfg.approvals_dir)
    runs: list[Value] = []
    for r in pipeline.list_runs(cfg):
        steps: list[Value] = []
        for s in r.steps:
            st = r.step_state(s.name)
            steps.append(
                {"name": s.name, "kind": s.kind, "state": st.get("state"), "note": st.get("note")}
            )
        runs.append(
            {
                "run": r.run_id,
                "template": r.template,
                "finished": bool(r.state.get("finished")),
                "steps": steps,
            }
        )
    out: Payload = {
        "runs": runs,
        "pending": [
            {"id": a.id, "kind": a.kind.value, "title": a.title}
            for a in store.all()
            if a.state is State.PENDING
        ],
        "submodules": [
            s.to_dict() for s in submodules.states(cfg, online=bool(os.environ.get("DMRSI_ONLINE")))
        ],
        "lessons": cast("Value", ctx.upstream.get("lessons") or []),
        "usage_offsets": cast("Value", ctx.upstream.get("usage_offsets") or {}),
        "template": cast(
            "Value", ctx.upstream.get("template") or os.environ.get("DMRSI_TEMPLATE", "synthetic")
        ),
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


def reasoning(ctx: StageContext) -> Payload:  # noqa: C901 -- 逐类 finding 平铺，和 AGENTS.local.md 的表一一对应
    """把观察整理成 findings，每条带建议动作。"""
    up = ctx.upstream
    findings: list[Value] = []
    for r in cast("list[Payload]", up.get("runs") or []):
        if r.get("finished"):
            continue
        for s in cast("list[Payload]", r.get("steps") or []):
            state = s.get("state")
            if state == "failed":
                owner = OWNER.get(str(s.get("kind")))
                findings.append(
                    {
                        "kind": "step_failed",
                        "run": r["run"],
                        "step": s["name"],
                        "owner": owner,
                        "evidence": s.get("note"),
                        "suggest": "draft_issue" if owner else "wait",
                    }
                )
                break
            if state == "blocked":
                findings.append(
                    {
                        "kind": "waiting_human",
                        "run": r["run"],
                        "step": s["name"],
                        "evidence": s.get("note"),
                        "suggest": "wait",
                    }
                )
                break
            if state in ("running", "pending"):
                findings.append(
                    {
                        "kind": "runnable",
                        "run": r["run"],
                        "step": s["name"],
                        "evidence": s.get("note"),
                        "suggest": "advance",
                    }
                )
                break
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
            findings.append(
                {
                    "kind": "submodule_behind",
                    "path": sm["path"],
                    "evidence": f"{sm['pinned']} → {sm['upstream']}",
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
    active = [r for r in cast("list[Payload]", up.get("runs") or []) if not r.get("finished")]
    if not active:
        findings.append(
            {
                "kind": "no_active_run",
                "suggest": "new_run",
                "evidence": f"没有进行中的流程，模板 {up.get('template')}",
            }
        )
    return {**_carry(up), "findings": findings}


def _carry(up: Payload) -> Payload:
    return {k: up[k] for k in ("lessons", "usage_offsets", "template", "pending") if k in up}


# ---- decision ------------------------------------------------------------------------------

_PRIORITY = {"advance": 0, "draft_issue": 1, "new_run": 2, "wait": 9}


def decision(ctx: StageContext) -> Payload:
    """选一件事。优先推进流程，其次把失败转成 issue，再其次新建流程；都没有就等人。"""
    up = ctx.upstream
    findings = cast("list[Payload]", up.get("findings") or [])
    lessons = [str(x) for x in cast("list[Value]", up.get("lessons") or [])]
    candidates = [
        f for f in findings if f.get("suggest") in _PRIORITY and f.get("suggest") != "wait"
    ]
    # 同一个失败已经起草过 issue（lessons 里记了）就不再起草
    candidates = [
        f
        for f in candidates
        if not (f.get("suggest") == "draft_issue" and _issue_key(f) in lessons)
    ]
    candidates.sort(key=lambda f: _PRIORITY[str(f["suggest"])])
    if not candidates:
        plan: Payload = {"action": "wait", "why": "只剩需要人做的事（审批、抽检）或没有可做的事"}
    else:
        f = candidates[0]
        plan = {"action": f["suggest"], "finding": f, "why": f"{f['kind']}：{f.get('evidence')}"}
    rejected = [{"finding": f.get("kind"), "action": f.get("suggest")} for f in candidates[1:]]
    return {**_carry(up), "plan": plan, "rejected": cast("Value", rejected)}


def _issue_key(f: Payload) -> str:
    return f"issue:{f.get('run')}/{f.get('step')}"


# ---- action --------------------------------------------------------------------------------


def action(ctx: StageContext) -> Payload:
    """执行 decision 选的动作（只允许动作表里的）。"""
    up = ctx.upstream
    plan = cast("Payload", up.get("plan") or {})
    act = str(plan.get("action"))
    if act not in ACTIONS:
        return {
            **_carry(up),
            "result": {"error": f"动作 {act!r} 不在动作表 {ACTIONS} 里，拒绝执行"},
        }
    cfg = _cfg()
    store = ApprovalStore(cfg.approvals_dir)
    finding = cast("Payload", plan.get("finding") or {})
    offsets = dict(cast("dict[str, int]", up.get("usage_offsets") or {}))
    out: Payload = {**_carry(up), "action": act}
    if act == "advance":
        run = pipeline.load(cfg, str(finding["run"]))
        res = run.advance(store, make_backends(cfg))
        usage, offsets[run.run_id] = _usage_since(run, offsets.get(run.run_id, 0))
        out["result"] = res
        if usage:
            out["_usage"] = usage
    elif act == "new_run":
        name = str(up.get("template") or "synthetic")
        steps, params = templates.TEMPLATES[name]()
        run_id = f"{name[:4]}-meta-{ctx.iteration:03d}-{os.getpid()}"
        pipeline.create(cfg, run_id, name, steps, params)
        out["result"] = {"created": run_id}
    elif act == "draft_issue":
        owner = str(finding.get("owner"))
        res = issues.draft(
            cfg,
            _gh(),
            store,
            direction=owner,
            key=f"{finding.get('run')}/{finding.get('step')}",
            title=f"dm-rsi 流程 {finding.get('run')} 的 {finding.get('step')} 步骤失败",
            body=f"dm-rsi 在调度流程时，`{finding.get('step')}` 步骤失败：\n\n```\n{finding.get('evidence')}\n```",
            evidence=[f"$DMRSI_HOME/runs/{finding.get('run')}/{finding.get('step')}/"],
        )
        out["result"] = (
            {"existing": res.existing.url}
            if res.existing
            else {"approval": res.approval.id if res.approval else None}
        )
        out["issue_key"] = _issue_key(finding)
    else:
        out["result"] = {
            "waiting": [
                f"{p['id']} {p['title']}" for p in cast("list[Payload]", up.get("pending") or [])
            ]
        }
    out["usage_offsets"] = cast("Value", offsets)
    return out


# ---- feedback ------------------------------------------------------------------------------


def feedback(ctx: StageContext) -> Payload:
    """评估本轮效果，记下 lessons；没有可做的事时停下等人。"""
    up = ctx.upstream
    act = str(up.get("action"))
    res = cast("Payload", up.get("result") or {})
    lessons = [str(x) for x in cast("list[Value]", up.get("lessons") or [])]
    if isinstance(up.get("issue_key"), str):
        lessons.append(str(up["issue_key"]))
    moved = len(cast("list[Value]", res.get("moved") or [])) if act == "advance" else 0
    progress = moved > 0 or act in ("new_run", "draft_issue")
    stop = act == "wait" or (
        act == "advance" and not moved and res.get("state") in ("blocked", "failed")
    )
    verdict = "progress" if progress else ("blocked" if stop else "no_change")
    return {
        "lessons": cast("Value", lessons[-50:]),
        "usage_offsets": up.get("usage_offsets") or {},
        "template": up.get("template"),
        "verdict": verdict,
        "moved": moved,
        "last": cast("Value", {"action": act, "result": res}),
        "stop": stop,
        "reason": "等人：审批或人工步骤" if stop else None,
    }


# ---- profile -------------------------------------------------------------------------------


def _progress(records: Sequence[IterationRecord]) -> float:
    return float(sum(int(cast("int", r.last.get("moved") or 0)) for r in records))


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
        seed={"template": os.environ.get("DMRSI_TEMPLATE", "synthetic")},
        metrics={"steps_moved": _progress},
        goal={
            "target": "DM-RSI 端到端流程",
            "metric": "推进的步骤数；完成的流程在 val 上的 f1_e2e",
        },
    )
