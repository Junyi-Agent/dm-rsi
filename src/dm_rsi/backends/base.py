"""后端的公共类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from xtra_rsi.payload import Payload, Value


class BackendError(RuntimeError):
    """后端调用失败或配置不合法。"""


class JobState(StrEnum):
    """作业状态，各后端映射到这五种。"""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    UNKNOWN = "unknown"
    """查不到或无法判定，需要人看。"""


@dataclass(frozen=True)
class JobSpec:
    """要运行的命令。``argv`` 不经过 shell；需要 shell 时自己写 ``["bash", "-c", ...]``。"""

    name: str
    argv: list[str]
    cwd: Path
    log_dir: Path
    env: dict[str, str] = field(default_factory=dict[str, str])
    gpus: int = 0
    timeout: int | None = None


@dataclass(frozen=True)
class JobRef:
    """提交后拿到的作业引用，写进 ``state.yaml``，下次 advance 时用来查状态。"""

    backend: str
    id: str
    meta: Payload = field(default_factory=dict[str, "Value"])

    def to_dict(self) -> Payload:
        """写盘用。"""
        return {"backend": self.backend, "id": self.id, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: Payload) -> JobRef:
        """读盘。"""
        meta = d.get("meta")
        return cls(
            backend=str(d["backend"]), id=str(d["id"]), meta=meta if isinstance(meta, dict) else {}
        )


class Backend(Protocol):
    """执行后端。"""

    name: str

    def submit(self, spec: JobSpec) -> JobRef:
        """提交，立即返回。"""
        ...

    def status(self, job: JobRef) -> JobState:
        """查状态。"""
        ...

    def fetch(self, job: JobRef, dest: Path) -> None:
        """把作业的日志和结果拷到 ``dest``。"""
        ...

    def cancel(self, job: JobRef) -> None:
        """取消。"""
        ...
