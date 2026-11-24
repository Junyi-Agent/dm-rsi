"""流程动作表：决策从这里选「下一段流程」，行动执行它（ADR 0001，2026-11-23 修订）。

每个流程有名字（决策用）、按模板选的实现（:data:`dm_rsi.workspace.STEP_KINDS` 里的种类）、默认参数、
决策可以改的参数（``ARGS``）。

:func:`suggest` 是默认的决策规则：看工作区里已有的产物，缺什么、什么过时了，就建议补哪一段。
LLM 模式下决策可以不按这个顺序（例如重训、换配方、多跑几轮 DP），但只能选表里的流程、改表里的参数。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from xtra_rsi.payload import Payload, Value

# 流程名 → {模板: 实现种类}；None 表示这个模板不需要这一段
FLOWS: dict[str, dict[str, str | None]] = {
    "prepare_data": {"synthetic": "synthetic.make_data", "elliptic": "elliptic.build_data"},
    # Training RSI 选训练方案（方案 A）；合成模板不训练模型，不需要
    "train_rsi": {"synthetic": None, "elliptic": "trsi.rsi"},
    "train_base": {"synthetic": "synthetic.score", "elliptic": "elliptic.train"},
    "score_base": {"synthetic": None, "elliptic": "elliptic.score"},
    "data_rsi": {"synthetic": "dgen.rsi", "elliptic": "dgen.rsi"},
    "retrain": {"synthetic": "synthetic.retrain", "elliptic": "elliptic.retrain"},
    "data_compare": {"synthetic": "dgen.compare", "elliptic": "dgen.compare"},
    "dp_rsi": {"synthetic": "dpolicy.rsi", "elliptic": "dpolicy.rsi"},
    "final_test": {"synthetic": "final_test", "elliptic": "final_test"},
}

# 决策可以给的参数（其余的由模板默认值决定，决策改不了）
ARGS: dict[str, tuple[str, ...]] = {
    "prepare_data": ("train_sub",),
    "train_rsi": ("judge", "train_gpu_ids", "gpu_hours", "max_candidates"),
    "train_base": ("train_gpus",),
    "score_base": (),
    "data_rsi": ("judge", "new_recipes", "max_arms"),
    "retrain": ("recipes", "train_gpus"),
    "data_compare": ("tag",),
    "dp_rsi": ("judge", "rounds", "replicates", "iterations"),
    "final_test": (),
}

# 模板 → 工作区参数和各流程的默认参数
TEMPLATES: dict[str, Payload] = {
    "synthetic": {
        "params": {"judge": "default", "seed": 0},
        "flows": {
            "train_base": {"model": "BASE"},
            # 合成行不需要人工抽检；真实数据默认 required
            "data_rsi": {"spotcheck": "skip"},
            # dgen 的合成 val 只有 1 个 step，dpolicy 默认的 by: step 协议分不出 fit / select，合成数据上按行分组
            "dp_rsi": {
                "features": "from_rows",
                "iterations": 3,
                "dpolicy": {"protocol": {"by": "group", "group": "id", "select_frac": 0.4}},
            },
        },
    },
    "elliptic": {
        "params": {"judge": "agent", "gpu_backend": "xtrah200", "seed": 0},
        "flows": {
            "train_rsi": {"train_gpu_ids": [0, 1], "gpu_hours": 100, "max_candidates": 4},
            "train_base": {"model": "BASE"},
            "score_base": {"model": "BASE"},
            "data_rsi": {"spotcheck": "required"},
            "dp_rsi": {"replicates": 3, "rounds": 2, "iterations": 5, "features": "from_outputs"},
        },
    },
}


class FlowError(ValueError):
    """流程名、参数或模板不合法。"""


def resolve(template: str, flow: str, args: Payload | None) -> tuple[str, Payload]:
    """流程名 + 决策给的参数 → (实现种类, 参数)。不在表里的流程或参数直接拒绝。"""
    if template not in TEMPLATES:
        msg = f"未知模板 {template!r}（{', '.join(TEMPLATES)}）"
        raise FlowError(msg)
    if flow not in FLOWS:
        msg = f"流程 {flow!r} 不在动作表里（{', '.join(FLOWS)}）"
        raise FlowError(msg)
    kind = FLOWS[flow].get(template)
    if kind is None:
        msg = f"模板 {template} 不需要流程 {flow}"
        raise FlowError(msg)
    args = dict(args or {})
    bad = sorted(set(args) - set(ARGS[flow]))
    if bad:
        msg = f"流程 {flow} 不接受参数 {bad}（可以给：{list(ARGS[flow]) or '无'}）"
        raise FlowError(msg)
    defaults = cast("dict[str, Payload]", TEMPLATES[template]["flows"]).get(flow, {})
    return kind, {**defaults, **args}


def template_params(template: str) -> Payload:
    """模板的工作区参数。"""
    if template not in TEMPLATES:
        msg = f"未知模板 {template!r}（{', '.join(TEMPLATES)}）"
        raise FlowError(msg)
    return dict(cast("Payload", TEMPLATES[template]["params"]))


def suggest(art: Payload, template: str | None = None) -> tuple[str, Payload, str] | None:  # noqa: PLR0911 -- 按流程顺序逐条检查，每条一个出口
    """默认决策规则：(流程, 参数, 理由)；没有可做的就返回 None。``art`` 是工作区的产物。

    基本只看产物：合成模板的 train_base 一步就产出了打过分的基线（``models``），自然跳过 score_base。
    唯一看模板的是 train_rsi：模板不需要它（``FLOWS`` 里是 None）时跳过。
    """
    models = cast("dict[str, Value]", art.get("models") or {})
    arms = cast("list[Payload]", art.get("arms") or [])
    if "rows_dir" not in art:
        return "prepare_data", {}, "工作区里还没有数据"
    needs_scheme = template is None or FLOWS["train_rsi"].get(template) is not None
    if needs_scheme and "train_scheme" not in art and "base_model" not in art:
        return "train_rsi", {}, "还没有训练方案：先让 Training RSI 选基座、配方和超参"
    if "base_model" not in art:
        return "train_base", {}, "还没有基线模型"
    if not models:
        return "score_base", {}, "基线模型还没打分"
    if not arms:
        return "data_rsi", {}, "还没有 Data RSI 生成的训练数据"
    untrained = [str(a["hash"]) for a in arms if str(a["hash"]) not in models]
    if untrained:
        return "retrain", {}, f"{len(untrained)} 组新训练数据还没重训：{', '.join(untrained)}"
    compared = sorted(str(k) for k in cast("list[Value]", art.get("compared") or []))
    if compared != sorted(models):
        return (
            "data_compare",
            {},
            "有新模型还没和基线对比" if compared else "还没对比重训模型和基线",
        )
    if art.get("dp_model") != art.get("model"):
        why = (
            "Decision Policy 还没跑"
            if "dp_model" not in art
            else f"对比选出的模型变了（{art.get('dp_model')} → {art.get('model')}）"
        )
        return "dp_rsi", {}, why
    if "dp_headline" not in art:
        return "final_test", {}, "val 上的选择都定了，剩 final-test"
    return None
