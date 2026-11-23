"""步骤实现。导入本包即把所有步骤种类注册到 :data:`dm_rsi.pipeline.STEP_KINDS`。"""

from dm_rsi.steps import dgen, dpolicy, final, synthetic, training

__all__ = ["dgen", "dpolicy", "final", "synthetic", "training"]
