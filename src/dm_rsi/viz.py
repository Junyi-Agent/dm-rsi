"""xtra-rsi viz 插件（``rsi viz``）：给 dm meta loop 的卡片补上工作区进度、待审批，以及行动阶段执行的流程。

行动阶段的 ``events``（调用了哪些 sub-RSI 命令、提交了哪些作业）作为「内部过程」显示：展开某一轮就能看到
这一轮执行的是哪段流程、做了哪些事、结果如何。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from xtra_rsi.yamlio import YAMLError, load_mapping

from dm_rsi import config, workspace
from dm_rsi.approvals import ApprovalStore, State

if TYPE_CHECKING:
    from pathlib import Path

    from xtra_rsi.payload import Payload, Value


def _read(path: Path) -> Payload | None:
    if not path.is_file():
        return None
    try:
        return load_mapping(path.read_text(encoding="utf-8"), where=str(path))
    except YAMLError:
        return None


def describe(loop_dir: Path, *, detail: bool) -> Payload | None:
    """只认 profile 为 dm 的 loop。"""
    st = _read(loop_dir / "status.yaml")
    if st is None or st.get("profile") != "dm":
        return None
    cfg = config.load()
    notes: dict[str, Value] = {}
    inner: dict[str, Value] = {}
    done_per_iter: list[Value] = []
    ws_id: str | None = None
    for it in sorted(loop_dir.glob("iter-*")):
        key = str(int(it.name.split("-")[1]))
        obs = _read(it / "observe.yaml") or {}
        ws_id = str(obs.get("workspace") or ws_id or "")
        act = _read(it / "action.yaml") or {}
        fb = _read(it / "feedback.yaml") or {}
        res = cast("Payload", act.get("result") or {})
        what = str(act.get("action"))
        if what in ("flow", "resume"):
            label = f"执行 {res.get('step')}：{res.get('state')}"
        elif what == "draft_issue":
            label = f"起草 issue：{res.get('approval') or res.get('existing')}"
        else:
            label = f"{what}：{res.get('why') or res.get('error') or ''}"
        notes[key] = {"action": label[:160]}
        done_per_iter.append(fb.get("flows_done") or 0)
        if detail:
            events = cast("list[Value]", act.get("events") or [])
            tail: list[Value] = [
                {"kind": "text", "text": str(res.get("note") or res.get("why") or "")}
            ]
            inner[key] = [*events, *tail]
    ws = workspace.load(cfg, ws_id) if ws_id else None
    pending = [a for a in ApprovalStore(cfg.approvals_dir).all() if a.state is State.PENDING]
    done = sum(1 for s in ws.steps if ws.step_state(s.name).get("state") == "done") if ws else 0
    out: Payload = {
        "title": f"dm-rsi · 工作区 {ws_id}" if ws_id else "dm-rsi meta loop",
        "group": "dm-rsi",
        "kpis": [
            {"label": "完成的流程段", "value": done},
            {"label": "待审批", "value": len(pending)},
        ],
        "series": [{"label": "每轮完成的流程段", "values": done_per_iter}],
        "stage_notes": notes,
    }
    if detail:
        inner.setdefault("summary", [{"kind": "text", "text": ws.summary()}] if ws else [])
        out["inner"] = inner
    return out
