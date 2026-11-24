"""issue 双向流（ADR 0001）。

- dm-rsi → 各方向：``draft`` 登记一个 ``issue`` 类审批（草稿），人 ``dm-rsi approve <id>`` 后由
  ``send`` 发到这个方向收 issue 的仓库（``Config.issue_repo``），标题带方向前缀。正文末尾带指纹标记，用来去重、把 issue 对应回诊断。
- 各方向 → dm-rsi：他们在 dm-rsi 自己的仓库提 issue；``inbox`` 读出来交给 observe。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dm_rsi.approvals import Approval, ApprovalError, ApprovalStore, Kind, State

if TYPE_CHECKING:
    from xtra_rsi.payload import Payload

    from dm_rsi.config import Config
    from dm_rsi.github import GitHub, Issue

MARK_RE = re.compile(r"<!-- dm-rsi:fp=([0-9a-f]{12}) -->")


def fingerprint(direction: str, key: str) -> str:
    """方向 + 诊断键（例如 ``dp:confident-errors``）→ 12 位十六进制。"""
    return hashlib.sha256(f"{direction}\0{key}".encode()).hexdigest()[:12]


def marker(fp: str) -> str:
    """写在正文末尾的隐藏标记。"""
    return f"<!-- dm-rsi:fp={fp} -->"


def fingerprint_of(body: str) -> str | None:
    """从正文里取指纹。"""
    m = MARK_RE.search(body)
    return m.group(1) if m else None


@dataclass(frozen=True)
class DraftResult:
    """``draft`` 的结果：要么已有同指纹的 open issue，要么是一个审批请求。"""

    existing: Issue | None = None
    approval: Approval | None = None


def _body(cfg: Config, direction: str, body: str, evidence: list[str], fp: str) -> str:
    d = cfg.direction(direction)
    lines = [body.rstrip(), ""]
    if evidence:
        lines += ["**证据**", "", *[f"- {e}" for e in evidence], ""]
    who = f"@{d.github_login}" if d.github_login else d.owner
    lines += [
        "---",
        (
            f"由 [dm-rsi](https://github.com/{cfg.self_repo})（meta RSI）起草，经人确认后发出。"
            f"方向：{direction}（负责人 {who}）。"
            f"有问题或需要 dm-rsi 配合，请在 https://github.com/{cfg.self_repo}/issues 提。"
        ),
        marker(fp),
    ]
    return "\n".join(lines)


def draft(  # noqa: PLR0913 -- 都是 issue 的组成部分，keyword-only
    cfg: Config,
    gh: GitHub,
    store: ApprovalStore,
    *,
    direction: str,
    key: str,
    title: str,
    body: str,
    evidence: list[str] | None = None,
) -> DraftResult:
    """起草发给某个方向的 issue。同指纹的 issue 已经 open 时不起草。"""
    d = cfg.direction(direction)
    fp = fingerprint(direction, key)
    repo = cfg.issue_repo(direction)
    for issue in gh.list_issues(repo, state="open", search="dm-rsi:fp in:body"):
        if fingerprint_of(issue.body) == fp:
            return DraftResult(existing=issue)
    full_title = title if title.startswith(d.issue_prefix) else f"{d.issue_prefix} {title}"
    detail: Payload = {
        "repo": repo,
        "direction": direction,
        "key": key,
        "issue_title": full_title,
        "issue_body": _body(cfg, direction, body, evidence or [], fp),
    }
    detail["explain"] = (
        f"dm-rsi 起草了一个要发给 {direction} 方向（负责人 {d.owner}）的 issue。\n"
        f"发到：https://github.com/{repo}（公开仓库，对方和所有人都看得到）\n"
        f"标题：{full_title}\n正文：\n{_indent(str(detail['issue_body']))}\n"
        "批准：立即用你的 gh 账号发出。拒绝：不发；同一个诊断不会再起草，除非换了新证据。\n"
        "建议：看正文的证据是否站得住、是不是对方的问题（而不是 dm-rsi 自己的 bug），再批。"
    )
    a = store.request(Kind.ISSUE, f"发 issue 给 {direction}：{full_title}", fp, detail)
    return DraftResult(approval=a)


def _indent(text: str) -> str:
    return "\n".join(f"  | {line}" for line in text.splitlines())


def send(gh: GitHub, store: ApprovalStore, aid: str) -> Approval:
    """发出一个已批准的 issue 草稿。"""
    a = store.get(aid)
    if a.kind is not Kind.ISSUE:
        msg = f"{aid} 是 {a.kind} 类请求，不是 issue"
        raise ApprovalError(msg)
    if a.state is not State.APPROVED:
        msg = f"{aid} 状态为 {a.state}，批准后才能发出"
        raise ApprovalError(msg)
    try:
        url = gh.create_issue(
            str(a.detail["repo"]), str(a.detail["issue_title"]), str(a.detail["issue_body"])
        )
    except Exception as e:  # noqa: BLE001 -- 任何失败都记进审批记录，人能看到原因
        return store.mark(aid, ok=False, result={"error": str(e)})
    return store.mark(aid, ok=True, result={"url": url})


def inbox(cfg: Config, gh: GitHub) -> list[Issue]:
    """各方向提给 dm-rsi 的 open issue。"""
    return gh.list_issues(cfg.self_repo, state="open")


def outbound(cfg: Config, gh: GitHub) -> list[Issue]:
    """dm-rsi 发出去的 issue（带指纹标记），含已关闭的，用来跟踪是否被处理。"""
    return [
        i
        for repo in cfg.issue_repos
        for i in gh.list_issues(repo, state="all", search="dm-rsi:fp in:body")
        if fingerprint_of(i.body)
    ]
