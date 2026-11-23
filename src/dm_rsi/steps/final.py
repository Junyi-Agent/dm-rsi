"""``final_test``：Data 和 Decision Policy 各在 test 上跑一次（必须先经人批准，ADR 0001）。

两个 sub-RSI 的 final-test 每个 run 只允许一次；结果文件已存在时直接读，不会自动加 ``--again``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dm_rsi import subenv
from dm_rsi.approvals import Kind, State
from dm_rsi.steps import dgen as dgen_step
from dm_rsi.workspace import Outcome, StepState, step_kind

if TYPE_CHECKING:
    from xtra_rsi.payload import Payload, Value

    from dm_rsi.workspace import StepContext


def _val_summary(ctx: StepContext) -> list[str]:
    """汇总 val 上已经定下来的结果：Data 选了哪个模型，Decision Policy 的 incumbent 比基线好多少。"""
    lines: list[str] = []
    cmp_ = cast("dict[str, dict[str, object]]", ctx.find_output("compare_summary") or {})
    chosen = str(ctx.find_output("model_key"))
    names = {
        str(a["hash"]): str(a["name"])
        for a in cast("list[dict[str, object]]", ctx.find_output("arms") or [])
    }
    for k, v in cmp_.items():
        mark = "（选中）" if k == chosen else ""
        lines.append(
            f"  Data：配方 {names.get(k, k)}（{k}）的模型 vs 基线，val AUPRC 差 {float(cast('float', v['diff'])):+.3f}"
            f" [{float(cast('float', v['lo'])):+.3f}, {float(cast('float', v['hi'])):+.3f}]，"
            f"{'通过' if v['pass'] else '未通过'}{mark}"
        )
    if chosen == "base":
        lines.append("  Data：没有配方通过，Decision Policy 用的是基线模型")
    arch = Path(str(ctx.find_output("dp_dir"))) / "r0" / "archive.jsonl"
    if arch.is_file():
        recs = [json.loads(x) for x in arch.read_text().splitlines() if x.strip()]
        base = next((r for r in recs if r.get("source") == "baseline"), None)
        acc = [r for r in recs if r.get("accepted") and r.get("source") != "baseline"]
        if base is not None:
            line = f"  Decision Policy：基线策略 val(select) f1_e2e = {float(base['objective']):.3f}，评估了 {len(recs) - 1} 个候选，"
            if acc:
                inc = acc[-1]
                vs = cast("dict[str, float]", inc.get("vs_incumbent") or {})
                line += f"接受了 {len(acc)} 个，当前 incumbent {inc['name']} = {float(inc['objective']):.3f}（差 {vs.get('diff', 0):+.3f}）"
            else:
                line += "没有一个通过接受规则，incumbent 仍是基线策略"
            lines.append(line)
    return lines


def _explain(ctx: StepContext, models: dict[str, str], *, synthetic: bool) -> str:
    head = [
        f"工作区 {ctx.ws.id}（模板 {ctx.ws.template}）val 上的流程都跑完了，最后一段是在 test 上各测一次：",
        f"  - Data：{len(models) - 1} 个重训模型 vs 基线（dgen final-test）",
        "  - Decision Policy：基线策略 vs val 上选出的 incumbent（dpolicy final-test）",
        "为什么问你：test 每个 run 只能读一次，读了之后就不能再拿 test 的结果回头改选择（两个 sub-RSI 的规则）。",
        "val 上已经定下的结果：",
        *_val_summary(ctx),
        "批准：跑两次 final-test，结果写进 final_test.json，流程完成；之后这个 run 不能再测 test。",
        "拒绝：final-test 不跑，流程以失败结束；val 上的产物都还在。",
    ]
    if synthetic:
        head.append(
            "建议：这是合成数据（演示 / 验收 dm-rsi 用的假数据），test 也是假的，批准没有风险，也不产生研究结论。"
            "想看流程完整跑完就批，不关心就拒绝。"
        )
    else:
        head.append("建议：确认 val 上的选择已经定了、不会再改提案或配方，再批。")
    return "\n".join(head)


@step_kind("final_test")
def final_test(ctx: StepContext) -> Outcome:
    """审批 → dgen final-test → dpolicy final-test。"""
    models = cast("dict[str, str]", ctx.find_output("models"))
    dp_conf = Path(str(cast("list[str]", ctx.find_output("dp_configs"))[0]))
    consensus = dp_conf.parent / "r0" / "consensus.yaml"
    plan: Payload = {
        "data": f"dgen final-test：{len(models) - 1} 个重训模型 vs 基线，在 test 上一次",
        "decision_policy": f"dpolicy final-test：baseline vs incumbent{'（含共识选择）' if consensus.is_file() else ''}，在 test 上一次",
        "dgen_config": str(dgen_step.config_path(ctx)),
        "dpolicy_config": str(dp_conf),
    }
    synthetic = ctx.ws.template == "synthetic"
    a = ctx.gate(
        Kind.FINAL_TEST,
        f"流程 {ctx.ws.id} 的 final-test（{'合成数据' if synthetic else '真实数据'}，读 test，每个 run 只一次）",
        "final",
        plan,
        explain=_explain(ctx, models, synthetic=synthetic),
    )
    if a.state not in (State.APPROVED, State.DONE):
        return ctx.blocked_on(a, "final-test")

    results: Payload = {}
    dg_out = ctx.ws.dir / "data-rsi" / "out" / "final_test" / "final" / "final_test.json"
    if len(models) > 1:
        if not dg_out.is_file():
            runs = ",".join(f"{k}={v}" for k, v in models.items())
            subenv.run_module(
                ctx.cfg,
                ctx.cfg.direction("data"),
                module="dgen",
                args=[
                    "final-test",
                    str(dgen_step.config_path(ctx)),
                    "--allow-test",
                    *("--runs", runs, "--baseline", "base", "--tag", "final"),
                ],
                cwd=ctx.dir,
                log_dir=ctx.log_dir,
                name="dgen-final-test",
            )
        found = next((ctx.ws.dir / "data-rsi" / "out").rglob("final_test.json"), None)
        results["data"] = str(found) if found else None

    dp_json = dp_conf.parent / "r0" / "final_test.json"
    if not dp_json.is_file():
        args = ["final-test", str(dp_conf), "--allow-test"]
        if consensus.is_file():
            args += ["--policy", str(consensus)]
        subenv.run_module(
            ctx.cfg,
            ctx.cfg.direction("decision-policy"),
            module="dpolicy",
            args=args,
            cwd=ctx.dir,
            log_dir=ctx.log_dir,
            name="dpolicy-final-test",
        )
    dp = cast("dict[str, object]", json.loads(dp_json.read_text()))
    pols = cast("dict[str, dict[str, object]]", dp.get("policies") or {})
    headline = {
        k: {"objective": v.get("objective"), "vs_baseline": v.get("vs_baseline")}
        for k, v in pols.items()
    }
    results["decision_policy"] = str(dp_json)
    results["dp_headline"] = cast("Value", json.loads(json.dumps(headline, default=str)))
    if a.state is State.APPROVED:
        ctx.store.mark(a.id, ok=True, result=results)
    return Outcome(StepState.DONE, "final-test 完成", outputs=results)
