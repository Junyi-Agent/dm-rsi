"""审批队列：对外、花钱、不可逆的动作先登记，人确认后才执行（ADR 0001「统一的审批队列」）。

每个请求一个文件 ``$DMRSI_HOME/approvals/<id>.yaml``。状态：

``pending`` → 人 ``approve`` → ``approved`` → 执行方 ``mark`` → ``done`` / ``failed``
``pending`` → 人 ``reject`` → ``rejected``

同一个 ``fingerprint`` 只登记一次：已有未结束（pending / approved）或被拒绝的请求时，返回原请求，
不重复打扰人。被拒绝后要重新申请，换一个 fingerprint（说明有了新证据）。

dm-rsi 启动的 agent（阶段命令、sub-RSI 会话）带环境变量 ``DMRSI_AGENT=1``，在这种环境里
``decide`` 一律拒绝：审批只能由人在自己的终端里做。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from dm_rsi import yamlfile

if TYPE_CHECKING:
    from pathlib import Path

    from xtra_rsi.payload import Payload

AGENT_ENV = "DMRSI_AGENT"
_ID_RE = re.compile(r"^a(\d{4,})$")


class Kind(StrEnum):
    """需要人确认的动作（ADR 0001）。"""

    ISSUE = "issue"
    """对外发 issue。"""
    TRAIN = "train"
    """启动一次训练（每次都要）。"""
    FINAL_TEST = "final_test"
    """``final-test`` 及 ``--again``。"""
    LLM_COST = "llm_cost"
    """付费 LLM 生成数据。"""
    REMOTE_SUBMIT = "remote_submit"
    """提交到 ssh / PBS 后端。"""
    SPOTCHECK = "spotcheck"
    """Data RSI 的人工抽检：人填完 spotcheck.csv 后批准（不是授权，是「人工步骤已完成」）。"""


class State(StrEnum):
    """审批状态。"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DONE = "done"
    FAILED = "failed"


OPEN = (State.PENDING, State.APPROVED)


class ApprovalError(ValueError):
    """请求不存在、状态不允许或调用方无权限。"""


@dataclass(frozen=True)
class Approval:
    """一个审批请求。"""

    id: str
    kind: Kind
    title: str
    fingerprint: str
    state: State
    detail: Payload
    created_at: str
    decided_at: str | None = None
    reason: str | None = None
    result: Payload | None = None

    @property
    def explain(self) -> str:
        """给人看的说明：是什么、为什么问、批准或拒绝之后会怎样、建议。"""
        return str(self.detail.get("explain") or "（没有说明）")

    def to_dict(self) -> Payload:
        """写盘用。"""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "title": self.title,
            "fingerprint": self.fingerprint,
            "state": self.state.value,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "reason": self.reason,
            "detail": self.detail,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, d: Payload) -> Approval:
        """读盘。"""
        return cls(
            id=str(d["id"]),
            kind=Kind(str(d["kind"])),
            title=str(d["title"]),
            fingerprint=str(d["fingerprint"]),
            state=State(str(d["state"])),
            detail=cast("Payload", d.get("detail") or {}),
            created_at=str(d["created_at"]),
            decided_at=cast("str | None", d.get("decided_at")),
            reason=cast("str | None", d.get("reason")),
            result=cast("Payload | None", d.get("result")),
        )


class ApprovalStore:
    """``$DMRSI_HOME/approvals/`` 下的审批请求。"""

    def __init__(self, path: Path) -> None:
        """``path`` 不存在时第一次写入时创建。"""
        self.dir: Path = path

    def _file(self, aid: str) -> Path:
        return self.dir / f"{aid}.yaml"

    def all(self) -> list[Approval]:
        """全部请求，按 id 排序。"""
        if not self.dir.is_dir():
            return []
        files = sorted(p for p in self.dir.glob("a*.yaml") if _ID_RE.match(p.stem))
        return [Approval.from_dict(yamlfile.read(p)) for p in files]

    def get(self, aid: str) -> Approval:
        """按 id 取。"""
        f = self._file(aid)
        if not _ID_RE.match(aid) or not f.is_file():
            msg = f"没有审批请求 {aid}"
            raise ApprovalError(msg)
        return Approval.from_dict(yamlfile.read(f))

    def find(self, fingerprint: str) -> Approval | None:
        """同一 fingerprint 最新的一个请求。"""
        hits = [a for a in self.all() if a.fingerprint == fingerprint]
        return hits[-1] if hits else None

    def request(self, kind: Kind, title: str, fingerprint: str, detail: Payload) -> Approval:
        """登记请求；同 fingerprint 已有未结束或被拒绝的请求时返回它。"""
        prev = self.find(fingerprint)
        if (
            prev is not None
            and prev.state is State.PENDING
            and (prev.title, prev.detail) != (title, detail)
        ):
            # 还没决定的请求：说明可以更新（例如补上更多上下文），id 不变
            new = Approval(**{**prev.__dict__, "title": title, "detail": detail})
            yamlfile.write(self._file(prev.id), new.to_dict())
            return new
        if prev is not None and (prev.state in OPEN or prev.state is State.REJECTED):
            return prev
        ids = [int(m.group(1)) for p in self.dir.glob("a*.yaml") if (m := _ID_RE.match(p.stem))]
        aid = f"a{(max(ids, default=0) + 1):04d}"
        a = Approval(
            id=aid,
            kind=kind,
            title=title,
            fingerprint=fingerprint,
            state=State.PENDING,
            detail=detail,
            created_at=yamlfile.now(),
        )
        yamlfile.write(self._file(aid), a.to_dict())
        return a

    def decide(self, aid: str, *, approve: bool, reason: str | None = None) -> Approval:
        """人的决定。agent 环境里拒绝调用。"""
        if os.environ.get(AGENT_ENV):
            msg = f"审批只能由人在自己的终端里做（检测到 {AGENT_ENV}）"
            raise ApprovalError(msg)
        a = self.get(aid)
        if a.state is not State.PENDING:
            msg = f"{aid} 状态为 {a.state}，只有 pending 的请求能批准或拒绝"
            raise ApprovalError(msg)
        if not approve and not reason:
            msg = "拒绝时要写原因（--reason），dm-rsi 会把它记进 lessons"
            raise ApprovalError(msg)
        new = Approval(
            **{
                **a.__dict__,
                "state": State.APPROVED if approve else State.REJECTED,
                "decided_at": yamlfile.now(),
                "reason": reason,
            }
        )
        yamlfile.write(self._file(aid), new.to_dict())
        return new

    def mark(self, aid: str, *, ok: bool, result: Payload) -> Approval:
        """执行方记录结果。只有 approved 的请求能被执行。"""
        a = self.get(aid)
        if a.state is not State.APPROVED:
            msg = f"{aid} 状态为 {a.state}，未经批准不能执行"
            raise ApprovalError(msg)
        new = Approval(
            **{**a.__dict__, "state": State.DONE if ok else State.FAILED, "result": result}
        )
        yamlfile.write(self._file(aid), new.to_dict())
        return new
