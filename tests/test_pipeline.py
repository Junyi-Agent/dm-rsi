from __future__ import annotations

import pytest

from dm_rsi import pipeline
from dm_rsi.approvals import ApprovalStore, Kind, State
from dm_rsi.pipeline import Outcome, StepDef, StepState, step_kind

CALLS: list[str] = []


@step_kind("test.ok")
def _ok(ctx):
    CALLS.append(ctx.step.name)
    return Outcome(StepState.DONE, "ok", outputs={"value": ctx.step.name})


@step_kind("test.read")
def _read(ctx):
    CALLS.append(ctx.step.name)
    return Outcome(StepState.DONE, outputs={"got": ctx.find_output("value")})


@step_kind("test.gate")
def _gate(ctx):
    a = ctx.gate(Kind.TRAIN, "启动训练", "train", {"gpus": 2}, explain="训练 2 卡")
    if a.state is not State.APPROVED:
        return ctx.blocked_on(a, "训练")
    return Outcome(StepState.DONE, outputs={"trained": True})


@step_kind("test.boom")
def _boom(ctx):
    msg = "坏了"
    raise RuntimeError(msg)


@pytest.fixture(autouse=True)
def _clear():
    CALLS.clear()


def test_advance_runs_in_order_and_passes_outputs(cfg):
    steps = [StepDef("a", "test.ok"), StepDef("b", "test.read", ("a",))]
    run = pipeline.create(cfg, "r1", "t", steps, {})
    res = run.advance(ApprovalStore(cfg.approvals_dir), {})
    assert res["state"] == "done"
    assert CALLS == ["a", "b"]
    run2 = pipeline.load(cfg, "r1")
    assert run2.step_state("b")["outputs"] == {"got": "a"}


def test_done_steps_are_not_rerun(cfg):
    run = pipeline.create(cfg, "r1", "t", [StepDef("a", "test.ok")], {})
    store = ApprovalStore(cfg.approvals_dir)
    run.advance(store, {})
    pipeline.load(cfg, "r1").advance(store, {})
    assert CALLS == ["a"]


def test_gate_blocks_until_human_approves(cfg):
    steps = [
        StepDef("a", "test.ok"),
        StepDef("g", "test.gate", ("a",)),
        StepDef("c", "test.ok", ("g",)),
    ]
    run = pipeline.create(cfg, "r1", "t", steps, {})
    store = ApprovalStore(cfg.approvals_dir)
    res = run.advance(store, {})
    assert res["state"] == "blocked"
    assert "a0001" in str(res["note"])
    # 再推进一次仍然等着，不会重复登记
    assert pipeline.load(cfg, "r1").advance(store, {})["state"] == "blocked"
    assert len(store.all()) == 1
    store.decide("a0001", approve=True)
    assert pipeline.load(cfg, "r1").advance(store, {})["state"] == "done"


def test_rejected_gate_fails_step(cfg):
    run = pipeline.create(cfg, "r1", "t", [StepDef("g", "test.gate")], {})
    store = ApprovalStore(cfg.approvals_dir)
    run.advance(store, {})
    store.decide("a0001", approve=False, reason="GPU 不够")
    res = pipeline.load(cfg, "r1").advance(store, {})
    assert res["state"] == "failed"
    assert "GPU 不够" in str(res["note"])


def test_exception_is_recorded_and_reset_reruns(cfg):
    run = pipeline.create(
        cfg, "r1", "t", [StepDef("a", "test.ok"), StepDef("x", "test.boom", ("a",))], {}
    )
    store = ApprovalStore(cfg.approvals_dir)
    res = run.advance(store, {})
    assert res["state"] == "failed"
    assert "坏了" in str(res["note"])
    assert (run.dir / "x" / "error.txt").is_file()
    run.reset("a")
    assert pipeline.load(cfg, "r1").step_state("a")["state"] == "pending"


def test_create_validates(cfg):
    with pytest.raises(pipeline.PipelineError, match="不在它前面"):
        pipeline.create(
            cfg, "r1", "t", [StepDef("b", "test.ok", ("a",)), StepDef("a", "test.ok")], {}
        )
    with pytest.raises(pipeline.PipelineError, match="未知步骤种类"):
        pipeline.create(cfg, "r2", "t", [StepDef("a", "nope")], {})
    with pytest.raises(pipeline.PipelineError, match="小写"):
        pipeline.create(cfg, "Bad_ID", "t", [StepDef("a", "test.ok")], {})
    pipeline.create(cfg, "r3", "t", [StepDef("a", "test.ok")], {})
    with pytest.raises(pipeline.PipelineError, match="已存在"):
        pipeline.create(cfg, "r3", "t", [StepDef("a", "test.ok")], {})
