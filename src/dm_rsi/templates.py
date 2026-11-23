"""流程模板。``dm-rsi run new --template <name>`` 用它生成 ``run.yaml``。

- ``synthetic``：sub-RSI 自带的合成数据，全程 CPU，用来验收 dm-rsi 本身（ADR 0001 P1 / P3）。
- ``elliptic``：Elliptic++ × elephant-1 的真实流程（P2），训练和打分走 GPU 后端，训练每次都要审批。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dm_rsi.pipeline import StepDef

if TYPE_CHECKING:
    from collections.abc import Callable

    from xtra_rsi.payload import Payload


def synthetic() -> tuple[list[StepDef], Payload]:
    """合成数据上的完整流程。"""
    steps = [
        StepDef("make-data", "synthetic.make_data"),
        StepDef("score-base", "synthetic.score", ("make-data",), {"model": "BASE"}),
        # 合成行不需要人工抽检；真实数据默认 required
        StepDef("data-rsi", "dgen.rsi", ("score-base",), {"spotcheck": "skip"}),
        StepDef("retrain", "synthetic.retrain", ("data-rsi",)),
        StepDef("data-compare", "dgen.compare", ("retrain",)),
        # dgen 的合成 val 只有 1 个 step，dpolicy 默认的 by: step 协议分不出 fit / select；
        # 合成数据上按行随机分组（by: group）。真实 Elliptic 的 val 有 5 个 step，用默认协议。
        StepDef(
            "dp-rsi",
            "dpolicy.rsi",
            ("data-compare",),
            {
                "features": "from_rows",
                "iterations": 3,
                "dpolicy": {"protocol": {"by": "group", "group": "id", "select_frac": 0.4}},
            },
        ),
        StepDef("final-test", "final_test", ("dp-rsi",)),
    ]
    return steps, {"judge": "default", "seed": 0}


def elliptic() -> tuple[list[StepDef], Payload]:
    """Elliptic++ × elephant-1：建数据 → 基线训练/打分 → Data RSI → 重训/打分 → 对比 → DP RSI → final-test。"""
    steps = [
        StepDef("build-data", "elliptic.build_data"),
        StepDef("train-base", "elliptic.train", ("build-data",), {"model": "BASE"}),
        StepDef("score-base", "elliptic.score", ("train-base",), {"model": "BASE"}),
        StepDef("data-rsi", "dgen.rsi", ("score-base",), {"spotcheck": "required"}),
        StepDef("retrain", "elliptic.retrain", ("data-rsi",)),
        StepDef("data-compare", "dgen.compare", ("retrain",)),
        StepDef(
            "dp-rsi",
            "dpolicy.rsi",
            ("data-compare",),
            {"replicates": 3, "rounds": 2, "iterations": 5, "features": "from_outputs"},
        ),
        StepDef("final-test", "final_test", ("dp-rsi",)),
    ]
    return steps, {"judge": "agent", "gpu_backend": "xtrah200", "seed": 0}


TEMPLATES: dict[str, Callable[[], tuple[list[StepDef], Payload]]] = {
    "synthetic": synthetic,
    "elliptic": elliptic,
}
