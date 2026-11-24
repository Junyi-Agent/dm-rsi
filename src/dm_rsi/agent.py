"""判断步骤的 agent 会话（ADR 0001「两种步骤」，P3）。

sub-RSI 的 SKILL.md 里有一部分要 agent 判断：读诊断、写配方 / 策略提案。dm-rsi 在**流程目录**里起一个
``claude -p`` 会话（不在 submodule 里，submodule 只用 ``--add-dir`` 给它读），提示词 = 该 sub-RSI 的
SKILL.md 路径 + 本步骤的任务和限制。

限制：

- 只给读写文件的工具（Read、Write、Edit、Glob、Grep），不给 Bash，单次会话有花费上限：agent 不能自己跑 sub-RSI 的命令，
  更不能跑 ``final-test``；命令由 dm-rsi 的步骤执行。
- 环境变量 ``DMRSI_AGENT=1``：即使它设法调用 dm-rsi，审批也会被拒绝（:mod:`dm_rsi.approvals`）。
- 会话结束后检查：submodule 没被写脏、``writes`` 目录里确实有新文件，否则步骤失败。
- 用量追加到 ``<run>/usage.jsonl``（``cost_usd``），meta loop 的 action 阶段把它作为 ``_usage`` 交给
  xtra-rsi 记到节点预算上。
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING, cast

from dm_rsi import submodules, yamlfile
from dm_rsi.approvals import AGENT_ENV

if TYPE_CHECKING:
    from pathlib import Path

    from xtra_rsi.payload import Payload

    from dm_rsi.workspace import StepContext

TOOLS = "Read,Write,Edit,Glob,Grep"


class AgentError(RuntimeError):
    """agent 会话失败或越界。"""


def claude_argv(
    prompt: str, *, add_dirs: list[Path], model: str | None, budget_usd: float
) -> list[str]:
    """``claude -p`` 的命令行。``DMRSI_CLAUDE`` 可以替换可执行文件（测试用）。

    ``--tools`` 限定可用的内置工具（没有 Bash），``--allowedTools`` 让它们不需要逐次确认，
    ``--max-budget-usd`` 是单次会话的花费上限。
    """
    argv = [os.environ.get("DMRSI_CLAUDE", "claude"), "-p", "--output-format", "json"]
    argv += ["--tools", TOOLS, "--allowedTools", TOOLS, "--max-budget-usd", f"{budget_usd:g}"]
    for d in add_dirs:
        argv += ["--add-dir", str(d)]
    if model:
        argv += ["--model", model]
    return [*argv, prompt]


def _prompt(ctx: StepContext, skill: Path, task: str, writes: Path) -> str:
    return "\n".join(
        [
            f"你是 dm-rsi（DM-RSI-Framework 的 meta RSI）调度的一个判断步骤，流程 {ctx.ws.id}，步骤 {ctx.step.name}。",
            f"先读这个 sub-RSI 的说明：{skill}（以及它引用的 reference/ 文档）。",
            f"任务：{task}",
            f"只能往 {writes} 里写文件。不要修改 {skill.parent} 下的任何文件。",
            "没有 Bash：不要尝试运行命令，命令由 dm-rsi 执行。",
            "完成后用 3–5 句话总结你的诊断和每个文件的意图。",
        ]
    )


def judge(  # noqa: PLR0913 -- 都是会话的组成部分，keyword-only
    ctx: StepContext,
    *,
    direction: str,
    task: str,
    writes: Path,
    reads: list[Path] | None = None,
    allow_empty: bool = False,
) -> Payload:
    """在流程目录里跑一次判断会话，返回结果摘要（也写进步骤目录的 ``agent-<n>.json``）。

    ``reads``：sub-RSI 包以外还要读的目录。``allow_empty``：不写文件也是合法结论（例如"不再提候选"）。
    ``writes`` 总是最后一个 ``--add-dir``。
    """
    d = ctx.cfg.direction(direction)
    pkg = d.package_dir(ctx.cfg.root)
    if pkg is None:
        msg = f"方向 {direction} 没有 sub-RSI"
        raise AgentError(msg)
    before = {p.name for p in writes.glob("*")} if writes.is_dir() else set()
    writes.mkdir(parents=True, exist_ok=True)
    argv = claude_argv(
        _prompt(ctx, pkg / "SKILL.md", task, writes),
        add_dirs=[pkg, *(reads or []), writes],
        model=cast("str | None", ctx.param("agent_model")),
        budget_usd=float(cast("float", ctx.param("agent_budget_usd", 2.0))),
    )
    proc = subprocess.run(
        argv,
        cwd=ctx.ws.dir,
        env={**os.environ, AGENT_ENV: "1"},
        capture_output=True,
        text=True,
        timeout=int(cast("int", ctx.param("agent_timeout", 1800))),
        check=False,
    )
    n = len(list(ctx.dir.glob("agent-*.json"))) + 1
    log = ctx.dir / f"agent-{n}.json"
    log.write_text(proc.stdout or "{}", encoding="utf-8")
    (ctx.dir / f"agent-{n}.err").write_text(proc.stderr, encoding="utf-8")
    try:
        res = cast("dict[str, object]", json.loads(proc.stdout))
    except json.JSONDecodeError:
        res = {}
    cost = float(cast("float", res.get("total_cost_usd") or 0.0))
    _record_usage(ctx, direction, cost, res)
    if proc.returncode != 0 or res.get("is_error"):
        msg = f"agent 会话失败（{proc.returncode}），见 {log}"
        raise AgentError(msg)
    dirty = [s for s in submodules.states(ctx.cfg, online=False) if s.dirty]
    if dirty:
        msg = f"agent 写进了只读的 submodule：{[(s.path, s.dirty) for s in dirty]}"
        raise AgentError(msg)
    new = sorted({p.name for p in writes.glob("*")} - before)
    if not new and not allow_empty:
        msg = f"agent 没有在 {writes} 写任何文件，见 {log}"
        raise AgentError(msg)
    return {
        "files": cast("Payload", new),
        "cost_usd": cost,
        "summary": str(res.get("result") or "")[:2000],
    }


def _record_usage(ctx: StepContext, direction: str, cost: float, res: dict[str, object]) -> None:
    usage = cast("dict[str, object]", res.get("usage") or {})
    entry = {
        "time": yamlfile.now(),
        "run": ctx.ws.id,
        "step": ctx.step.name,
        "direction": direction,
        "cost_usd": cost,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }
    with (ctx.ws.dir / "usage.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
