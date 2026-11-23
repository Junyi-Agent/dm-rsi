"""ADR 0001 P1 验收：合成数据上用 sub-RSI 的真实 CLI 跑通整条流程，submodule 不被写入。"""

from __future__ import annotations

import subprocess

import pytest

from dm_rsi import pipeline
from dm_rsi.approvals import ApprovalStore, Kind, State
from dm_rsi.pipeline_cli import make_backends
from dm_rsi.templates import synthetic


def _dirty(cfg):
    out = []
    for d in cfg.directions.values():
        if d.submodule:
            out += subprocess.run(
                ["git", "-C", str(cfg.root / d.submodule), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
    return out


@pytest.mark.slow
def test_synthetic_pipeline_end_to_end(cfg_with_env):
    cfg = cfg_with_env
    steps, params = synthetic()
    run = pipeline.create(cfg, "e2e", "synthetic", steps, params)
    store = ApprovalStore(cfg.approvals_dir)
    res = run.advance(store, make_backends(cfg))
    assert res["state"] == "blocked", res
    assert res["step"] == "final-test"
    [a] = store.all()
    assert a.kind is Kind.FINAL_TEST
    # 人只看 pending 就能决定：说明里写了是合成数据、val 结果、批准 / 拒绝的后果
    for words in ("合成数据", "val 上已经定下的结果", "批准：", "拒绝：", "建议："):
        assert words in a.explain

    # 测试夹具里的临时审批队列，扮演人
    store.decide(a.id, approve=True)
    res = pipeline.load(cfg, "e2e").advance(store, make_backends(cfg))
    assert res["state"] == "done", res
    assert store.get(a.id).state is State.DONE

    run = pipeline.load(cfg, "e2e")
    assert len(run.step_state("data-rsi")["outputs"]["arms"]) == 2
    assert run.step_state("data-compare")["outputs"]["model"]
    final = run.step_state("final-test")["outputs"]
    assert set(final["dp_headline"]) >= {"baseline", "incumbent"}
    assert (run.dir / "dp-rsi" / "r0" / "final_test.json").is_file()
    assert _dirty(cfg) == []
