"""``dm-rsi run ...``：新建、推进、查看流程。"""

from __future__ import annotations

import argparse  # noqa: TC003 -- register() 运行时要用 argparse 的类型
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from xtra_rsi.yamlio import dump, loads

from dm_rsi import backends, pipeline, steps, templates
from dm_rsi.approvals import ApprovalStore

if TYPE_CHECKING:
    from collections.abc import Mapping

    from xtra_rsi.payload import Payload

    from dm_rsi.config import Config
    from dm_rsi.github import GitHub

_ = steps  # 导入即注册步骤种类


def make_backends(cfg: Config) -> Mapping[str, backends.Backend]:
    """``dm-rsi.yaml`` 里配置的全部后端（构造时不连服务器）。"""
    out: dict[str, backends.Backend] = {}
    for name, conf in cfg.backends.items():
        try:
            out[name] = backends.make(name, conf, state_dir=cfg.home / "backends" / name)
        except backends.BackendError as e:
            sys.stderr.write(f"dm-rsi: 跳过后端 {name}：{e}\n")
    return out


def _parse_set(items: list[str]) -> Payload:
    out: Payload = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            msg = f"--set 要写成 key=value：{it!r}"
            raise pipeline.PipelineError(msg)
        out[k] = loads(v, where=f"--set {k}")
    return out


def cmd_new(args: argparse.Namespace, cfg: Config, _gh: GitHub) -> int:
    """按模板新建流程。"""
    steps_, params = templates.TEMPLATES[args.template]()
    run_id = args.id or f"{args.template[:4]}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    run = pipeline.create(cfg, run_id, args.template, steps_, {**params, **_parse_set(args.set)})
    sys.stdout.write(
        dump({"run": run.run_id, "dir": str(run.dir), "steps": [s.name for s in run.steps]})
    )
    return 0


def cmd_advance(args: argparse.Namespace, cfg: Config, _gh: GitHub) -> int:
    """推进流程直到完成、等待或失败。"""
    run = pipeline.load(cfg, args.run)
    res = run.advance(
        ApprovalStore(cfg.approvals_dir), make_backends(cfg), max_steps=args.max_steps
    )
    sys.stdout.write(dump(res))
    return 1 if res.get("state") == "failed" else 0


def cmd_show(args: argparse.Namespace, cfg: Config, _gh: GitHub) -> int:
    """流程的定义和每步状态。"""
    run = pipeline.load(cfg, args.run)
    rows: list[Payload] = []
    for s in run.steps:
        st = run.step_state(s.name)
        row: Payload = {
            "step": s.name,
            "kind": s.kind,
            "state": st.get("state"),
            "note": st.get("note"),
        }
        if args.outputs:
            row["outputs"] = st.get("outputs")
        rows.append(row)
    sys.stdout.write(
        dump(
            {
                "run": run.run_id,
                "template": run.template,
                "params": run.params,
                "steps": cast("Payload", rows),
            }
        )
    )
    return 0


def cmd_list(_args: argparse.Namespace, cfg: Config, _gh: GitHub) -> int:
    """全部流程。"""
    sys.stdout.write(dump({"runs": [r.summary() for r in pipeline.list_runs(cfg)]}))
    return 0


def cmd_reset(args: argparse.Namespace, cfg: Config, _gh: GitHub) -> int:
    """把某步及之后的步骤重置为 pending。"""
    pipeline.load(cfg, args.run).reset(args.step)
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage] -- argparse 没有公开这个类型
    """注册 ``run`` 子命令。"""
    r = sub.add_parser("run", help="流程：新建、推进、查看")
    rs = r.add_subparsers(dest="run_cmd", required=True)
    n = rs.add_parser("new", help="按模板新建流程")
    n.add_argument("--template", choices=sorted(templates.TEMPLATES), required=True)
    n.add_argument("--id", help="run_id（默认按模板名和时间生成）")
    n.add_argument(
        "--set", action="append", default=[], help="流程级参数 key=value（值按 YAML 解析），可重复"
    )
    n.set_defaults(func=cmd_new)
    a = rs.add_parser("advance", help="推进直到完成、等待审批/作业或失败")
    a.add_argument("run")
    a.add_argument("--max-steps", type=int, default=100)
    a.set_defaults(func=cmd_advance)
    s = rs.add_parser("show", help="每步状态")
    s.add_argument("run")
    s.add_argument("--outputs", action="store_true", help="同时列出每步产物")
    s.set_defaults(func=cmd_show)
    rs.add_parser("list", help="全部流程").set_defaults(func=cmd_list)
    x = rs.add_parser("reset", help="把某步及之后的步骤重置为 pending")
    x.add_argument("run")
    x.add_argument("step")
    x.set_defaults(func=cmd_reset)
