"""流程（pipeline）：一串步骤，只通过文件契约交接（ADR 0001「流程由步骤组成」）。

一次流程 = ``$DMRSI_HOME/runs/<run_id>/``：

- ``run.yaml``：步骤列表（名字、种类、依赖、参数），创建后不改。
- ``state.yaml``：每个步骤的状态、产物、作业引用、审批 id、说明。
- ``<step>/``：步骤自己的工作目录和日志。

``advance`` 按顺序推进：跳过已完成的步骤，对第一个未完成的步骤调用它的实现；步骤返回
``done`` 就继续下一个，返回 ``running``（后台作业）/ ``blocked``（等人审批或人工操作）/
``failed`` 就停下。步骤实现必须幂等：重复调用不会重复做已经做完的事。
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

_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class PipelineError(ValueError):
    """流程定义不合法或状态不允许。"""


class StepState(StrEnum):
    """步骤状态。"""

    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class StepDef:
    """``run.yaml`` 里的一个步骤。"""

    name: str
    kind: str
    needs: tuple[str, ...] = ()
    params: Payload = field(default_factory=dict[str, "Value"])

    def to_dict(self) -> Payload:
        """写盘用。"""
        return {
            "name": self.name,
            "kind": self.kind,
            "needs": list(self.needs),
            "params": self.params,
        }


@dataclass(frozen=True)
class Outcome:
    """一次 advance 的结果。``outputs`` 是步骤的产物（契约），下游步骤读它。"""

    state: StepState
    note: str = ""
    outputs: Payload = field(default_factory=dict[str, "Value"])
    data: Payload = field(default_factory=dict[str, "Value"])
    """步骤自己的中间状态（例如作业引用），下次 advance 时原样拿回。"""


class StepFn(Protocol):
    """步骤实现。"""

    def __call__(self, ctx: StepContext, /) -> Outcome:
        """推进一次。"""
        ...


class StepContext:
    """步骤实现能用到的一切。"""

    def __init__(
        self,
        run: Run,
        step: StepDef,
        data: Payload,
        store: ApprovalStore,
        backends: Mapping[str, Backend],
    ) -> None:
        """只由 :meth:`Run.advance` 构造。"""
        self.run: Run = run
        self.cfg: Config = run.cfg
        self.step: StepDef = step
        self.data: Payload = data
        self.store: ApprovalStore = store
        self.backends: Mapping[str, Backend] = backends

    @property
    def dir(self) -> Path:
        """步骤的工作目录。"""
        d = self.run.dir / self.step.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def log_dir(self) -> Path:
        """步骤的日志目录。"""
        return self.dir / "logs"

    def param(self, key: str, default: Value = None) -> Value:
        """步骤参数，其次是流程级参数。"""
        if key in self.step.params:
            return self.step.params[key]
        return self.run.params.get(key, default)

    def outputs(self, step: str) -> Payload:
        """某个已完成步骤的产物。"""
        st = self.run.step_state(step)
        if st.get("state") != StepState.DONE:
            msg = f"{self.step.name} 需要 {step} 的产物，但它还没完成"
            raise PipelineError(msg)
        return cast("Payload", st.get("outputs") or {})

    def find_output(self, key: str) -> Value:
        """依赖链上（按 needs 从近到远）第一个带 ``key`` 的产物。"""
        seen: set[str] = set()
        queue = list(self.step.needs)
        while queue:
            name = queue.pop(0)
            if name in seen:
                continue
            seen.add(name)
            out = self.outputs(name)
            if key in out:
                return out[key]
            queue += list(self.run.step_def(name).needs)
        msg = f"{self.step.name}：依赖链上没有步骤产出 {key!r}"
        raise PipelineError(msg)

    def gate(self, kind: Kind, title: str, key: str, detail: Payload, *, explain: str) -> Approval:
        """登记（或取回）一个审批。fingerprint 含 run_id 和步骤名，同一流程的同一件事只问一次。

        已经执行过（done）的审批也直接返回：步骤轮询后台作业时会再次走到这里，不能因此重新登记。

        ``explain`` 必填：给人看的说明，写清楚这是什么、为什么要人确认、批准 / 拒绝之后会怎样、建议怎么选。
        人只看 ``dm-rsi pending`` 就要能做决定，不需要去翻流程目录。
        """
        fp = f"{self.run.run_id}/{self.step.name}/{key}"
        prev = self.store.find(fp)
        if prev is not None and prev.state is State.DONE:
            return prev
        return self.store.request(
            kind,
            title,
            fp,
            {"run": self.run.run_id, "step": self.step.name, **detail, "explain": explain},
        )

    def blocked_on(self, a: Approval, what: str) -> Outcome:
        """等审批时的 Outcome。被拒绝时步骤失败，原因写进 note。"""
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
            raise PipelineError(msg) from None


STEP_KINDS: dict[str, StepFn] = {}


def step_kind(name: str) -> Callable[[StepFn], StepFn]:
    """注册步骤实现。"""

    def deco(fn: StepFn) -> StepFn:
        STEP_KINDS[name] = fn
        return fn

    return deco


@dataclass
class Run:
    """一次流程。"""

    cfg: Config
    run_id: str
    steps: list[StepDef]
    params: Payload
    template: str
    state: Payload

    @property
    def dir(self) -> Path:
        """``$DMRSI_HOME/runs/<run_id>``。"""
        return self.cfg.runs_dir / self.run_id

    def step_def(self, name: str) -> StepDef:
        """按名字取步骤定义。"""
        for s in self.steps:
            if s.name == name:
                return s
        msg = f"流程 {self.run_id} 没有步骤 {name}"
        raise PipelineError(msg)

    def step_state(self, name: str) -> Payload:
        """步骤的状态记录。"""
        steps = cast("dict[str, Payload]", self.state.setdefault("steps", {}))
        return steps.setdefault(name, {"state": StepState.PENDING.value})

    def save(self) -> None:
        """写 ``state.yaml``。"""
        self.state["updated_at"] = yamlfile.now()
        yamlfile.write(self.dir / "state.yaml", self.state)

    def summary(self) -> str:
        """一行摘要：run_id、模板、每步状态。"""
        marks = {"done": "✓", "running": "…", "blocked": "‖", "failed": "✗", "pending": "·"}
        parts = [
            f"{s.name}{marks.get(str(self.step_state(s.name).get('state')), '?')}"
            for s in self.steps
        ]
        return f"{self.run_id} [{self.template}] " + " ".join(parts)

    def current(self) -> tuple[StepDef, Payload] | None:
        """第一个未完成的步骤。"""
        for s in self.steps:
            st = self.step_state(s.name)
            if st.get("state") != StepState.DONE:
                return s, st
        return None

    def advance(
        self, store: ApprovalStore, backends: Mapping[str, Backend], *, max_steps: int = 100
    ) -> Payload:
        """按顺序推进，直到完成、等待或失败。返回本次推进的摘要。"""
        moved: list[Value] = []
        for _ in range(max_steps):
            cur = self.current()
            if cur is None:
                self.state["finished"] = True
                self.save()
                return {"run": self.run_id, "moved": moved, "state": "done"}
            step, st = cur
            missing = [n for n in step.needs if self.step_state(n).get("state") != StepState.DONE]
            if missing:
                msg = f"{step.name} 的依赖 {missing} 未完成，但排在它前面的步骤都已完成：run.yaml 顺序有误"
                raise PipelineError(msg)
            fn = STEP_KINDS.get(step.kind)
            if fn is None:
                msg = f"未知步骤种类 {step.kind!r}（{step.name}）"
                raise PipelineError(msg)
            st.setdefault("started_at", yamlfile.now())
            ctx = StepContext(self, step, cast("Payload", st.get("data") or {}), store, backends)
            try:
                out = fn(ctx)
            except Exception as e:  # noqa: BLE001 -- 任何异常都记进 state，让人和 meta loop 看到
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
                moved.append(step.name)
            self.save()
            if out.state is not StepState.DONE:
                return {
                    "run": self.run_id,
                    "moved": moved,
                    "state": out.state.value,
                    "step": step.name,
                    "note": out.note,
                }
        return {
            "run": self.run_id,
            "moved": moved,
            "state": "running",
            "note": f"一次最多推进 {max_steps} 步",
        }

    def reset(self, step: str) -> None:
        """把某步及其后的步骤重置为 pending（失败后修好了要重跑时用）。产物目录不删。"""
        names = [s.name for s in self.steps]
        if step not in names:
            msg = f"没有步骤 {step}"
            raise PipelineError(msg)
        for n in names[names.index(step) :]:
            cast("dict[str, Payload]", self.state["steps"])[n] = {"state": StepState.PENDING.value}
        self.state.pop("finished", None)
        self.save()


def create(cfg: Config, run_id: str, template: str, steps: list[StepDef], params: Payload) -> Run:
    """新建流程。run_id 已存在时报错（不覆盖）。"""
    if not _RUN_ID_RE.match(run_id):
        msg = f"run_id {run_id!r} 只能用小写字母、数字和 -"
        raise PipelineError(msg)
    names = [s.name for s in steps]
    if len(set(names)) != len(names):
        msg = f"步骤名重复：{names}"
        raise PipelineError(msg)
    for i, s in enumerate(steps):
        if s.kind not in STEP_KINDS:
            msg = f"未知步骤种类 {s.kind!r}（{s.name}）"
            raise PipelineError(msg)
        bad = [n for n in s.needs if n not in names[:i]]
        if bad:
            msg = f"{s.name} 依赖的 {bad} 不在它前面"
            raise PipelineError(msg)
    d = cfg.runs_dir / run_id
    if (d / "run.yaml").exists():
        msg = f"流程 {run_id} 已存在：{d}"
        raise PipelineError(msg)
    spec: Payload = {
        "run_id": run_id,
        "template": template,
        "created_at": yamlfile.now(),
        "params": params,
        "steps": [s.to_dict() for s in steps],
    }
    yamlfile.write(d / "run.yaml", spec)
    run = Run(cfg, run_id, steps, params, template, {"run_id": run_id, "steps": {}})
    run.save()
    return run


def load(cfg: Config, run_id: str) -> Run:
    """读已有流程。"""
    d = cfg.runs_dir / run_id
    if not (d / "run.yaml").is_file():
        msg = f"没有流程 {run_id}（{d}）"
        raise PipelineError(msg)
    spec = yamlfile.read(d / "run.yaml")
    steps = [
        StepDef(
            name=str(s["name"]),
            kind=str(s["kind"]),
            needs=tuple(str(n) for n in cast("list[str]", s.get("needs") or [])),
            params=cast("Payload", s.get("params") or {}),
        )
        for s in cast("list[Payload]", spec["steps"])
    ]
    state = (
        yamlfile.read(d / "state.yaml")
        if (d / "state.yaml").is_file()
        else {"run_id": run_id, "steps": {}}
    )
    return Run(
        cfg,
        run_id,
        steps,
        cast("Payload", spec.get("params") or {}),
        str(spec.get("template")),
        state,
    )


def list_runs(cfg: Config) -> list[Run]:
    """全部流程，按创建顺序。"""
    if not cfg.runs_dir.is_dir():
        return []
    runs = [load(cfg, p.name) for p in sorted(cfg.runs_dir.iterdir()) if (p / "run.yaml").is_file()]
    return sorted(runs, key=lambda r: str(yamlfile.read(r.dir / "run.yaml").get("created_at")))
