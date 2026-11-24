"""``dm-rsi`` 命令。输出一律是 YAML（stdout），错误写 stderr、退出码非零。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from xtra_rsi.yamlio import dump

from dm_rsi import __version__, config, issues, submodules, versions, workspace
from dm_rsi.approvals import ApprovalError, ApprovalStore, Kind, State
from dm_rsi.github import GhCli, GitHub, GitHubError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from xtra_rsi.payload import Payload, Value

type Handler = Callable[[argparse.Namespace, config.Config, GitHub], int]


def _emit(obj: Payload) -> None:
    sys.stdout.write(dump(obj))


def _err(msg: str) -> None:
    sys.stderr.write(f"dm-rsi: {msg}\n")


def cmd_status(args: argparse.Namespace, cfg: config.Config, gh: GitHub) -> int:
    """总览：submodule、待审批、收件箱、流程 run。"""
    store = ApprovalStore(cfg.approvals_dir)
    out: Payload = {
        "version": __version__,
        "home": str(cfg.home),
        "submodules": [s.to_dict() for s in submodules.states(cfg, online=not args.offline)],
        "pending": [f"{a.id} {a.kind} {a.title}" for a in store.all() if a.state is State.PENDING],
        "workspaces": [w.summary() for w in workspace.list_workspaces(cfg)],
    }
    if not args.offline:
        try:
            out["inbox"] = [f"#{i.number} {i.title} ({i.author})" for i in issues.inbox(cfg, gh)]
            out["outbound"] = [f"{i.state} {i.title} {i.url}" for i in issues.outbound(cfg, gh)]
        except GitHubError as e:
            out["github_error"] = str(e)
    _emit(out)
    return 0


def cmd_pending(args: argparse.Namespace, cfg: config.Config, _gh: GitHub) -> int:
    """列出待审批的请求：默认是给人读的说明，``--yaml`` 输出完整记录。"""
    store = ApprovalStore(cfg.approvals_dir)
    pending = [a for a in store.all() if a.state is State.PENDING]
    if args.yaml:
        _emit({"pending": [a.to_dict() for a in pending]})
        return 0
    if not pending:
        sys.stdout.write("没有待审批的请求。\n")
        return 0
    for a in pending:
        sys.stdout.write(f"\n[{a.id}] {a.title}\n")
        sys.stdout.write("".join(f"    {line}\n" for line in a.explain.splitlines()))
        sys.stdout.write(
            f'    → 批准：dm-rsi approve {a.id}    拒绝：dm-rsi reject {a.id} --reason "..."\n'
        )
    return 0


def cmd_approvals(_args: argparse.Namespace, cfg: config.Config, _gh: GitHub) -> int:
    """列出全部审批请求（摘要）。"""
    store = ApprovalStore(cfg.approvals_dir)
    _emit({"approvals": [f"{a.id} {a.state} {a.kind} {a.title}" for a in store.all()]})
    return 0


def cmd_approve(args: argparse.Namespace, cfg: config.Config, gh: GitHub) -> int:
    """（人）批准。issue 类请求批准后立即发出。"""
    store = ApprovalStore(cfg.approvals_dir)
    a = store.decide(args.id, approve=True, reason=args.reason)
    if a.kind is Kind.ISSUE:
        a = issues.send(gh, store, a.id)
    _emit(a.to_dict())
    return 0 if a.state is not State.FAILED else 1


def cmd_reject(args: argparse.Namespace, cfg: config.Config, _gh: GitHub) -> int:
    """（人）拒绝，必须写原因。"""
    store = ApprovalStore(cfg.approvals_dir)
    _emit(store.decide(args.id, approve=False, reason=args.reason).to_dict())
    return 0


def cmd_issue_draft(args: argparse.Namespace, cfg: config.Config, gh: GitHub) -> int:
    """起草发给某个方向的 issue（进入审批队列，不会直接发出）。"""
    body = Path(args.body_file).read_text(encoding="utf-8") if args.body_file else args.body
    if not body:
        _err("需要 --body 或 --body-file")
        return 2
    res = issues.draft(
        cfg,
        gh,
        ApprovalStore(cfg.approvals_dir),
        direction=args.to,
        key=args.key,
        title=args.title,
        body=body,
        evidence=args.evidence,
    )
    if res.existing is not None:
        _emit({"existing": res.existing.url, "note": "同一诊断已有 open 的 issue，未起草"})
    elif res.approval is not None:
        _emit({**res.approval.to_dict(), "note": f"人运行 dm-rsi approve {res.approval.id} 后发出"})
    return 0


def cmd_issue_list(_args: argparse.Namespace, cfg: config.Config, gh: GitHub) -> int:
    """两个方向的 issue：收件箱（提给 dm-rsi 的）和 dm-rsi 发出去的。"""
    _emit(
        {
            "inbox": [
                {
                    "number": i.number,
                    "title": i.title,
                    "author": i.author,
                    "url": i.url,
                    "updated_at": i.updated_at,
                }
                for i in issues.inbox(cfg, gh)
            ],
            "outbound": [
                {"title": i.title, "state": i.state, "comments": i.comments, "url": i.url}
                for i in issues.outbound(cfg, gh)
            ],
        }
    )
    return 0


def cmd_work_list(_args: argparse.Namespace, cfg: config.Config, _gh: GitHub) -> int:
    """全部工作区。"""
    _emit({"workspaces": [w.summary() for w in workspace.list_workspaces(cfg)]})
    return 0


def cmd_work_show(args: argparse.Namespace, cfg: config.Config, _gh: GitHub) -> int:
    """一个工作区的流程执行。"""
    ws = workspace.load(cfg, args.id)
    rows: list[Value] = []
    for s in ws.steps:
        st = ws.step_state(s.name)
        row: Payload = {
            "step": s.name,
            "kind": s.kind,
            "state": st.get("state"),
            "note": st.get("note"),
        }
        if args.outputs:
            row["outputs"] = st.get("outputs")
        rows.append(row)
    _emit(
        {
            "workspace": ws.id,
            "template": ws.template,
            "dir": str(ws.dir),
            "executions": rows,
            "versions": versions.summary(ws),
        }
    )
    return 0


def cmd_work_reset(args: argparse.Namespace, cfg: config.Config, _gh: GitHub) -> int:
    """重置一次执行。"""
    workspace.load(cfg, args.id).reset(args.step)
    _emit({"reset": args.step, "note": "meta loop 下一轮会重新推进它（rsi run --profile dm ...）"})
    return 0


def build_parser() -> argparse.ArgumentParser:
    """命令行结构。"""
    p = argparse.ArgumentParser(
        prog="dm-rsi", description="DM-RSI meta RSI：统一入口，调度各方向的 sub-RSI"
    )
    p.add_argument("--version", action="version", version=f"dm-rsi {__version__}")
    p.add_argument("--home", type=Path, help="覆盖 $DMRSI_HOME / dm-rsi.yaml 里的 home")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="总览：submodule、待审批、收件箱、流程 run")
    s.add_argument("--offline", action="store_true", help="不访问网络（不查上游和 issue）")
    s.set_defaults(func=cmd_status)
    pe = sub.add_parser("pending", help="待审批的请求和说明")
    pe.add_argument("--yaml", action="store_true", help="输出完整记录（YAML）")
    pe.set_defaults(func=cmd_pending)
    sub.add_parser("approvals", help="全部审批请求").set_defaults(func=cmd_approvals)
    a = sub.add_parser("approve", help="（人）批准；issue 类请求批准后立即发出")
    a.add_argument("id")
    a.add_argument("--reason")
    a.set_defaults(func=cmd_approve)
    r = sub.add_parser("reject", help="（人）拒绝")
    r.add_argument("id")
    r.add_argument("--reason", required=True)
    r.set_defaults(func=cmd_reject)

    i = sub.add_parser("issue", help="issue 双向流")
    isub = i.add_subparsers(dest="issue_cmd", required=True)
    d = isub.add_parser("draft", help="起草发给某个方向的 issue（进审批队列）")
    d.add_argument("--to", required=True, help="方向：decision-policy / data / training")
    d.add_argument("--key", required=True, help="诊断键，用于去重，例如 dp:confident-errors")
    d.add_argument("--title", required=True)
    d.add_argument("--body")
    d.add_argument("--body-file")
    d.add_argument("--evidence", action="append", default=[], help="证据（路径、数字），可重复")
    d.set_defaults(func=cmd_issue_draft)
    isub.add_parser("list", help="收件箱和已发出的 issue").set_defaults(func=cmd_issue_list)

    w = sub.add_parser("work", help="工作区：行动阶段执行过的流程（只读查看，或重置失败的执行）")
    wsub = w.add_subparsers(dest="work_cmd", required=True)
    wsub.add_parser("list", help="全部工作区").set_defaults(func=cmd_work_list)
    ws_ = wsub.add_parser("show", help="一个工作区的每次流程执行和产物")
    ws_.add_argument("id")
    ws_.add_argument("--outputs", action="store_true", help="同时列出每次执行的产物")
    ws_.set_defaults(func=cmd_work_show)
    wr = wsub.add_parser("reset", help="把失败的执行重置为 pending，meta loop 下一轮会重新推进它")
    wr.add_argument("id")
    wr.add_argument("step")
    wr.set_defaults(func=cmd_work_reset)
    return p


def main(argv: Sequence[str] | None = None, *, gh: GitHub | None = None) -> int:
    """入口。"""
    args = build_parser().parse_args(argv)
    try:
        cfg = config.load(home=args.home)
        handler: Handler = args.func
        return handler(args, cfg, gh or GhCli())
    except (config.ConfigError, ApprovalError, GitHubError, workspace.WorkspaceError) as e:
        _err(str(e))
        return 1
