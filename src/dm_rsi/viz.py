"""xtra-rsi viz 插件（``rsi viz``）：给 dm meta loop 的卡片补上流程进度、待审批和每轮动作。"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from xtra_rsi.yamlio import YAMLError, load_mapping

from dm_rsi import config, pipeline
from dm_rsi.approvals import ApprovalStore, State

if TYPE_CHECKING:
    from pathlib import Path

    from xtra_rsi.payload import Payload, Value


def _status(loop_dir: Path) -> Payload | None:
    f = loop_dir / "status.yaml"
    if not f.is_file():
        return None
    try:
        return load_mapping(f.read_text(encoding="utf-8"), where=str(f))
    except YAMLError:
        return None


def describe(loop_dir: Path, *, detail: bool) -> Payload | None:
    """只认 profile 为 dm 的 loop。"""
    st = _status(loop_dir)
    if st is None or st.get("profile") != "dm":
        return None
    cfg = config.load()
    runs = pipeline.list_runs(cfg)
    pending = [a for a in ApprovalStore(cfg.approvals_dir).all() if a.state is State.PENDING]
    done_steps = sum(
        1 for r in runs for s in r.steps if r.step_state(s.name).get("state") == "done"
    )
    all_steps = sum(len(r.steps) for r in runs)
    notes: dict[str, Value] = {}
    moved: list[Value] = []
    for it in sorted(loop_dir.glob("iter-*")):
        fb = it / "feedback.yaml"
        if not fb.is_file():
            continue
        obj = load_mapping(fb.read_text(encoding="utf-8"), where=str(fb))
        last = cast("Payload", obj.get("last") or {})
        key = str(int(it.name.split("-")[1]))
        notes[key] = {
            "action": f"{last.get('action')}：{obj.get('verdict')}，推进 {obj.get('moved')} 步"
        }
        moved.append(obj.get("moved") or 0)
    out: Payload = {
        "title": "dm-rsi meta loop",
        "group": "dm-rsi",
        "kpis": [
            {"label": "流程", "value": len(runs)},
            {"label": "已完成步骤", "value": done_steps, "target": all_steps},
            {"label": "待审批", "value": len(pending)},
        ],
        "series": [{"label": "每轮推进步数", "values": moved}],
        "stage_notes": notes,
    }
    if detail:
        items: list[Value] = [{"kind": "text", "text": r.summary()} for r in runs]
        items.extend({"kind": "text", "text": f"待审批 {a.id} {a.title}"} for a in pending)
        out["inner"] = {"0": items}
    return out
