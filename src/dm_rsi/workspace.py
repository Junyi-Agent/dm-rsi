"""工作区：meta loop 的「行动」阶段在这里执行流程（ADR 0001，2026-11-23 修订）。

dm-rsi 自己做观察、推理、决策；**行动**执行决策选中的那一段具体流程（数据、训练、打分、对比、
Decision Policy……）。工作区记录这些流程的执行和产物：

- ``$DMRSI_HOME/work/<id>/workspace.yaml``：模板、参数、按时间追加的流程执行列表（``NNN-<flow>``）。
- ``state.yaml``：每次流程执行的状态、产物、作业引用、审批 id。
- ``NNN-<flow>/``：这次执行的工作目录和日志；跨执行共享的产物（数据、分数、sub-RSI 的运行目录）
  放在工作区根目录下的固定位置（``data/``、``scores/``、``data-rsi/``、``dp-rsi/``）。

流程之间不声明依赖：一次执行要什么产物，就在整个工作区里找**最新完成**的那一份（:meth:`StepContext.find_output`）。
缺什么、什么过时了，由决策根据观察判断（:mod:`dm_rsi.flows`）。

流程执行必须幂等：同一次执行可以被反复推进（等作业、等审批），不会重复做已经做完的事。
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, cast

from dm_rsi import yamlfile
from dm_rsi.approvals import Approval, ApprovalStore, Kind, State

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from xtra_rsi.payload import Payload, Value

    from dm_rsi.backends.base import Backend
    from dm_rsi.config import Config

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class WorkspaceError(ValueError):
    """工作区或流程执行不合法。"""


class StepState(StrEnum):
    """一次流程执行的状态。"""

    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"


OPEN_STATES = (StepState.PENDING, StepState.RUNNING, StepState.BLOCKED)


@dataclass(frozen=True)
class StepDef:
    """一次流程执行：名字 ``NNN-<flow>``、实现种类、参数。"""

    name: str
    flow: str
    kind: str
    params: Payload = field(default_factory=dict[str, "Value"])

    def to_dict(self) -> Payload:
        """写盘用。"""
        return {"name": self.name, "flow": self.flow, "kind": self.kind, "params": self.params}


@dataclass(frozen=True)
class Outcome:
    """一次推进的结果。``outputs`` 是产物，之后的流程用 :meth:`StepContext.find_output` 读。"""

    state: StepState
    note: str = ""
    outputs: Payload = field(default_factory=dict[str, "Value"])
    data: Payload = field(default_factory=dict[str, "Value"])
    """执行自己的中间状态（例如作业引用），下次推进时原样拿回。"""


class StepFn(Protocol):
    """流程实现。"""

    def __call__(self, ctx: StepContext, /) -> Outcome:
        """推进一次。"""
        ...


STEP_KINDS: dict[str, StepFn] = {}


def step_kind(name: str) -> Callable[[StepFn], StepFn]:
    """注册流程实现。"""

    def deco(fn: StepFn) -> StepFn:
        STEP_KINDS[name] = fn
        return fn

    return deco


class StepContext:
    """流程实现能用到的一切。"""

    def __init__(
        self,
        ws: Workspace,
        step: StepDef,
        data: Payload,
        store: ApprovalStore,
        backends: Mapping[str, Backend],
    ) -> None:
        """只由 :meth:`Workspace.run_step` 构造。"""
        self.ws: Workspace = ws
        self.cfg: Config = ws.cfg
        self.step: StepDef = step
        self.data: Payload = data
        self.store: ApprovalStore = store
        self.backends: Mapping[str, Backend] = backends

    @property
    def dir(self) -> Path:
        """这次执行的工作目录。"""
        d = self.ws.dir / self.step.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def log_dir(self) -> Path:
        """这次执行的日志目录。"""
        return self.dir / "logs"

    def param(self, key: str, default: Value = None) -> Value:
        """执行参数（决策给的），其次是工作区参数。"""
        if key in self.step.params:
            return self.step.params[key]
        return self.ws.params.get(key, default)

    def find_output(self, key: str) -> Value:
        """工作区里最新完成的、带 ``key`` 的产物。"""
        v = self.ws.latest_output(key)
        if v is None:
            msg = f"{self.step.name}：工作区里还没有产物 {key!r}（先执行产出它的流程）"
            raise WorkspaceError(msg)
        return v

    def gate(self, kind: Kind, title: str, key: str, detail: Payload, *, explain: str) -> Approval:
        """登记（或取回）一个审批。fingerprint 含工作区和执行名，同一次执行的同一件事只问一次。

        已经执行过（done）的审批也直接返回：推进后台作业时会再次走到这里，不能因此重新登记。

        ``explain`` 必填：给人看的说明，写清楚这是什么、为什么要人确认、批准 / 拒绝之后会怎样、建议怎么选。
        人只看 ``dm-rsi pending`` 就要能做决定，不需要去翻工作区。
        """
        fp = f"{self.ws.id}/{self.step.name}/{key}"
        prev = self.store.find(fp)
        if prev is not None and prev.state is State.DONE:
            return prev
        return self.store.request(
            kind,
            title,
            fp,
            {"workspace": self.ws.id, "step": self.step.name, **detail, "explain": explain},
        )

    def blocked_on(self, a: Approval, what: str) -> Outcome:
        """等审批时的 Outcome。被拒绝时执行失败，原因写进 note。"""
        if a.state is State.REJECTED:
            return Outcome(StepState.FAILED, f"{what}被拒绝（{a.id}）：{a.reason}")
        return Outcome(
            StepState.BLOCKED,
            f"等待审批 {a.id}：{what}（dm-rsi approve {a.id}）",
            data={"approval": a.id},
        )

    def backend(self, name: str) -> Backend:
        """按名字取后端。"""
        try:
            return self.backends[name]
        except KeyError:
            msg = f"没有配置后端 {name!r}（dm-rsi.yaml 的 backends）"
            raise WorkspaceError(msg) from None


@dataclass
class Workspace:
    """一个工作区。"""

    cfg: Config
    id: str
    template: str
    params: Payload
    steps: list[StepDef]
    state: Payload

    @property
    def dir(self) -> Path:
        """``$DMRSI_HOME/work/<id>``。"""
        return self.cfg.home / "work" / self.id

    def step_def(self, name: str) -> StepDef:
        """按名字取一次执行。"""
        for s in self.steps:
            if s.name == name:
                return s
        msg = f"工作区 {self.id} 没有执行 {name}"
        raise WorkspaceError(msg)

    def step_state(self, name: str) -> Payload:
        """一次执行的状态记录。"""
        steps = cast("dict[str, Payload]", self.state.setdefault("steps", {}))
        return steps.setdefault(name, {"state": StepState.PENDING.value})

    def latest_output(self, key: str) -> Value:
        """最新完成的执行里 ``key`` 的产物（按完成时间）。"""
        best: tuple[str, Value] | None = None
        for s in self.steps:
            st = self.step_state(s.name)
            if st.get("state") != StepState.DONE:
                continue
            out = cast("Payload", st.get("outputs") or {})
            if key in out:
                ended = str(st.get("ended_at") or "")
                if best is None or ended >= best[0]:
                    best = (ended, out[key])
        return None if best is None else best[1]

    def artifacts(self) -> Payload:
        """所有已完成执行的产物合并（新的覆盖旧的），给观察阶段用。"""
        done = [s for s in self.steps if self.step_state(s.name).get("state") == StepState.DONE]
        done.sort(key=lambda s: str(self.step_state(s.name).get("ended_at") or ""))
        out: Payload = {}
        for s in done:
            out.update(cast("Payload", self.step_state(s.name).get("outputs") or {}))
        return out

    def open_steps(self) -> list[StepDef]:
        """还没结束的执行（pending / running / blocked）。"""
        return [s for s in self.steps if self.step_state(s.name).get("state") in OPEN_STATES]

    def _save_spec(self) -> None:
        yamlfile.write(
            self.dir / "workspace.yaml",
            {
                "id": self.id,
                "template": self.template,
                "params": self.params,
                "steps": [s.to_dict() for s in self.steps],
            },
        )

    def save(self) -> None:
        """写 ``state.yaml``。"""
        self.state["updated_at"] = yamlfile.now()
        yamlfile.write(self.dir / "state.yaml", self.state)

    def add(self, flow: str, kind: str, params: Payload) -> StepDef:
        """追加一次流程执行。"""
        if kind not in STEP_KINDS:
            msg = f"未知流程实现 {kind!r}（{flow}）"
            raise WorkspaceError(msg)
        step = StepDef(f"{len(self.steps) + 1:03d}-{flow}", flow, kind, params)
        self.steps.append(step)
        self._save_spec()
        self.step_state(step.name)
        self.save()
        return step

    def run_step(self, name: str, store: ApprovalStore, backends: Mapping[str, Backend]) -> Payload:
        """推进一次执行，返回摘要（状态、说明、事件）。"""
        step = self.step_def(name)
        st = self.step_state(name)
        fn = STEP_KINDS.get(step.kind)
        if fn is None:
            msg = f"未知流程实现 {step.kind!r}（{step.name}）"
            raise WorkspaceError(msg)
        st.setdefault("started_at", yamlfile.now())
        ctx = StepContext(self, step, cast("Payload", st.get("data") or {}), store, backends)
        logs_before = _logs(ctx.log_dir)
        try:
            out = fn(ctx)
        except Exception as e:  # noqa: BLE001 -- 任何异常都记进状态，让人和 meta loop 看到
            out = Outcome(StepState.FAILED, f"{type(e).__name__}: {e}", data=ctx.data)
            (ctx.dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        st.update(
            {
                "state": out.state.value,
                "note": out.note,
                "data": {**ctx.data, **out.data},
                "updated_at": yamlfile.now(),
            }
        )
        if out.state is StepState.DONE:
            st["outputs"] = out.outputs
            st["ended_at"] = yamlfile.now()
        self.save()
        events: list[Value] = [
            {"kind": "tool", "text": f"{p.stem}（日志 {p}）"}
            for p in sorted(_logs(ctx.log_dir) - logs_before, key=lambda p: p.stat().st_mtime)
        ]
        return {
            "step": name,
            "flow": step.flow,
            "state": out.state.value,
            "note": out.note,
            "events": events,
            "outputs": list(out.outputs) if out.state is StepState.DONE else [],
        }

    def reset(self, name: str) -> None:
        """把一次执行重置为 pending（失败后修好了要重来时用）。产物目录不删。"""
        self.step_def(name)
        cast("dict[str, Payload]", self.state["steps"])[name] = {"state": StepState.PENDING.value}
        self.save()

    def summary(self) -> str:
        """一行摘要。"""
        marks = {"done": "✓", "running": "…", "blocked": "‖", "failed": "✗", "pending": "·"}
        parts = [
            f"{s.name}{marks.get(str(self.step_state(s.name).get('state')), '?')}"
            for s in self.steps
        ]
        return f"{self.id} [{self.template}] " + (" ".join(parts) or "（还没有执行任何流程）")


def _logs(d: Path) -> set[Path]:
    return set(d.glob("*.err")) if d.is_dir() else set()


def open_or_create(
    cfg: Config, ws_id: str, template: str, params: Payload | None = None
) -> Workspace:
    """取已有工作区，没有就新建。"""
    if (cfg.home / "work" / ws_id / "workspace.yaml").is_file():
        return load(cfg, ws_id)
    if not _ID_RE.match(ws_id):
        msg = f"工作区 id {ws_id!r} 只能用小写字母、数字和 -"
        raise WorkspaceError(msg)
    ws = Workspace(cfg, ws_id, template, dict(params or {}), [], {"id": ws_id, "steps": {}})
    ws._save_spec()  # noqa: SLF001 -- 同模块
    ws.save()
    return ws


def load(cfg: Config, ws_id: str) -> Workspace:
    """读已有工作区。"""
    d = cfg.home / "work" / ws_id
    if not (d / "workspace.yaml").is_file():
        msg = f"没有工作区 {ws_id}（{d}）"
        raise WorkspaceError(msg)
    spec = yamlfile.read(d / "workspace.yaml")
    steps = [
        StepDef(
            str(s["name"]), str(s["flow"]), str(s["kind"]), cast("Payload", s.get("params") or {})
        )
        for s in cast("list[Payload]", spec.get("steps") or [])
    ]
    state = (
        yamlfile.read(d / "state.yaml")
        if (d / "state.yaml").is_file()
        else {"id": ws_id, "steps": {}}
    )
    return Workspace(
        cfg, ws_id, str(spec["template"]), cast("Payload", spec.get("params") or {}), steps, state
    )


def list_workspaces(cfg: Config) -> list[Workspace]:
    """全部工作区。"""
    root = cfg.home / "work"
    if not root.is_dir():
        return []
    return [load(cfg, p.name) for p in sorted(root.iterdir()) if (p / "workspace.yaml").is_file()]
