"""执行后端（ADR 0001「执行后端可插拔」）。

- ``local``：本机 CPU，subprocess。
- ``taskmgr``：GPU，gpu-taskmgr 的 ``task`` CLI。
- ``ssh``：直接登录的远程机器（学校 A100 服务器）。提交要 ``remote_submit`` 审批。
- ``pbs``：PBS 集群（Borealis），经登录节点 ``qsub``。提交要 ``remote_submit`` 审批。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dm_rsi.backends.base import Backend, BackendError, JobRef, JobSpec, JobState
from dm_rsi.backends.local import LocalBackend
from dm_rsi.backends.ssh import PbsBackend, SshBackend
from dm_rsi.backends.taskmgr import TaskmgrBackend

if TYPE_CHECKING:
    from pathlib import Path

    from xtra_rsi.payload import Payload, Value

__all__ = [
    "Backend",
    "BackendError",
    "JobRef",
    "JobSpec",
    "JobState",
    "LocalBackend",
    "PbsBackend",
    "SshBackend",
    "TaskmgrBackend",
    "make",
    "requires_approval",
]


def requires_approval(backend: Backend) -> bool:
    """提交到这个后端要不要人确认（ssh / pbs 要）。"""
    return bool(getattr(backend, "requires_approval", False))


def _str(conf: Payload, key: str, name: str) -> str:
    v = conf.get(key)
    if not isinstance(v, str) or not v:
        msg = f"backends.{name}.{key} 必须是非空字符串"
        raise BackendError(msg)
    return v


def _int(v: Value, default: int) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else default


def make(name: str, conf: Payload, *, state_dir: Path) -> Backend:
    """按 ``dm-rsi.yaml`` 的 ``backends.<name>`` 构造后端。构造时不连服务器。"""
    kind = conf.get("kind")
    if kind == "local":
        return LocalBackend(name)
    if kind == "taskmgr":
        return TaskmgrBackend(
            name,
            host=_str(conf, "host", name),
            max_gpus=_int(conf.get("max_gpus"), 2),
            min_free_gb=_int(conf.get("min_free_gb"), 40),
            task_cli=str(conf.get("task_cli") or "task"),
            state_dir=state_dir,
        )
    if kind == "ssh":
        return SshBackend(
            name,
            host=_str(conf, "host", name),
            remote_dir=_str(conf, "remote_dir", name),
            max_gpus=_int(conf.get("max_gpus"), 2),
            ssh_cli=str(conf.get("ssh_cli") or "ssh"),
        )
    if kind == "pbs":
        return PbsBackend(
            name,
            host=_str(conf, "host", name),
            remote_dir=_str(conf, "remote_dir", name),
            project=_str(conf, "project", name),
            max_gpus=_int(conf.get("max_gpus"), 2),
            walltime=str(conf.get("walltime") or "02:00:00"),
            ssh_cli=str(conf.get("ssh_cli") or "ssh"),
        )
    msg = f"backends.{name}.kind 未知：{kind!r}（local / taskmgr / ssh / pbs）"
    raise BackendError(msg)
