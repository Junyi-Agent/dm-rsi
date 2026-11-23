"""sub-RSI submodule 的状态：固定在哪个 commit、上游是否有新 commit、工作区是否被弄脏。

submodule 只读（ADR 0001）。``dirty`` 非空说明有东西写进了 sub-RSI 的目录，是 dm-rsi 的 bug。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from xtra_rsi.payload import Payload

    from dm_rsi.config import Config


def _git(cwd: Path, *argv: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *argv], capture_output=True, text=True, check=False, timeout=60
    )
    if proc.returncode != 0:
        msg = f"git {' '.join(argv)} 失败：{proc.stderr.strip()}"
        raise RuntimeError(msg)
    return proc.stdout


@dataclass(frozen=True)
class SubmoduleState:
    """一个 sub-RSI submodule 的状态。"""

    direction: str
    path: str
    branch: str | None
    pinned: str
    upstream: str | None
    dirty: list[str]

    @property
    def behind(self) -> bool | None:
        """上游分支头和固定的 commit 不同。离线时为 None。"""
        return None if self.upstream is None else self.upstream != self.pinned

    def to_dict(self) -> Payload:
        """摘要。"""
        return {
            "direction": self.direction,
            "path": self.path,
            "branch": self.branch,
            "pinned": self.pinned[:12],
            "upstream": self.upstream[:12] if self.upstream else None,
            "behind": self.behind,
            "dirty": list(self.dirty),
        }


def states(cfg: Config, *, online: bool = True) -> list[SubmoduleState]:
    """所有有 submodule 的方向。``online`` 时用 ls-remote 查上游分支头。"""
    out: list[SubmoduleState] = []
    for d in cfg.directions.values():
        if d.submodule is None:
            continue
        sub = cfg.root / d.submodule
        pinned = _git(sub, "rev-parse", "HEAD").strip()
        dirty = [
            line[3:] for line in _git(sub, "status", "--porcelain").splitlines() if line.strip()
        ]
        upstream: str | None = None
        if online and d.branch:
            url = _git(cfg.root, "config", "-f", ".gitmodules", f"submodule.{d.name}.url").strip()
            line = _git(cfg.root, "ls-remote", url, f"refs/heads/{d.branch}").split()
            upstream = line[0] if line else None
        out.append(
            SubmoduleState(
                direction=d.name,
                path=d.submodule,
                branch=d.branch,
                pinned=pinned,
                upstream=upstream,
                dirty=dirty,
            )
        )
    return out
