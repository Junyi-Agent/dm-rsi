"""sub-RSI 流程里「要 agent 做判断」的环节：写配方（Data）、写策略提案（Decision Policy）。

两种模式（步骤参数 ``judge``）：

- ``default``：确定性的默认做法，用 sub-RSI 自带的示例（P1，也是 agent 不可用时的兜底）。
- ``agent``：在 sub-RSI 的目录里起一个 agent 会话，按它的 SKILL.md 读诊断、写文件（P3，见 :mod:`dm_rsi.agent`）。

两种模式都只往 ``out_dir`` 里写 YAML，之后由 sub-RSI 的 CLI 检查和执行；写出来的东西不合法时 CLI 会拒绝。
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from xtra_rsi.yamlio import dump

from dm_rsi import agent

if TYPE_CHECKING:
    from pathlib import Path

    from dm_rsi.workspace import StepContext

DEFAULT_RECIPES = ("hard_all.yaml", "smote_baseline.yaml")
DEFAULT_PROPOSALS = ("example_abstain_second_stage.yaml", "example_rolling_platt.yaml")


def _mode(ctx: StepContext) -> str:
    mode = str(ctx.param("judge", "default"))
    if mode not in {"default", "agent"}:
        msg = f"judge 只能是 default / agent，收到 {mode!r}"
        raise ValueError(msg)
    return mode


def write_recipes(ctx: StepContext, out_dir: Path, diagnosis: Path) -> list[str]:
    """为 Data RSI 在 ``out_dir`` 写 1–4 个配方，返回文件名。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if _mode(ctx) == "agent":
        agent.judge(
            ctx,
            direction="data",
            task=(
                f"读 {diagnosis}，按 SKILL.md 主流程第 3–4 步写下诊断，并在 {out_dir} 写 1–4 个配方 YAML"
                "（每个带 note:，总带一个 method: hard 作参照）。只写配方，不运行任何 dgen 命令。"
            ),
            writes=out_dir,
        )
    else:
        pkg = ctx.cfg.direction("data").package_dir(ctx.cfg.root)
        assert pkg is not None  # noqa: S101
        for name in DEFAULT_RECIPES:
            shutil.copy(pkg / "configs" / "recipes" / name, out_dir / name)
    return sorted(p.name for p in out_dir.glob("*.yaml"))


def write_proposals(
    ctx: StepContext, out_dir: Path, diagnosis: Path, features: list[str]
) -> list[str]:
    """为 Decision Policy RSI 在 ``out_dir`` 写 1–4 个策略提案，返回文件名。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if _mode(ctx) == "agent":
        agent.judge(
            ctx,
            direction="decision-policy",
            task=(
                f"读 {diagnosis}，按 SKILL.md 主流程第 4–5 步写下诊断，并在 {out_dir} 写 1–4 个策略 YAML"
                f"（每个带 note:）。可用的特征列：{features or '无'}。只写提案，不运行任何 dpolicy 命令。"
            ),
            writes=out_dir,
        )
    else:
        pkg = ctx.cfg.direction("decision-policy").package_dir(ctx.cfg.root)
        assert pkg is not None  # noqa: S101
        for name in DEFAULT_PROPOSALS:
            shutil.copy(pkg / "configs" / "policies" / name, out_dir / name)
        if features:
            (out_dir / "dmrsi_abstain30_logreg_features.yaml").write_text(
                dump(
                    {
                        "name": "dmrsi_abstain30_logreg_features",
                        "calibration": {"method": "none"},
                        "decision": {"rule": "max_f1"},
                        "abstain": {"coverage": 0.7},
                        "second_stage": {
                            "enabled": True,
                            "model": "logreg",
                            "apply_to": "band",
                            "features": list(features),
                        },
                        "note": "dm-rsi 默认提案：拒判带交给能看到额外特征的二次判断（SKILL 经验：有新信息的二次判断才有收益）",
                    }
                ),
                encoding="utf-8",
            )
    return sorted(p.name for p in out_dir.glob("*.yaml"))
