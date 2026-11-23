from __future__ import annotations

import pytest

from dm_rsi import flows, workspace
from dm_rsi.approvals import ApprovalStore, Kind, State
from dm_rsi.workspace import Outcome, StepState, step_kind

CALLS: list[str] = []


@step_kind("test.ok")
def _ok(ctx):
    CALLS.append(ctx.step.name)
    return Outcome(StepState.DONE, "ok", outputs={"value": ctx.step.name})


@step_kind("test.read")
def _read(ctx):
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


def test_find_output_takes_latest_done_execution(cfg):
    ws = workspace.open_or_create(cfg, "w", "synthetic")
    store = ApprovalStore(cfg.approvals_dir)
    a = ws.add("x", "test.ok", {})
    ws.run_step(a.name, store, {})
    b = ws.add("x", "test.ok", {})
    ws.run_step(b.name, store, {})
    c = ws.add("y", "test.read", {})
    res = ws.run_step(c.name, store, {})
    assert res["state"] == "done"
    assert workspace.load(cfg, "w").step_state(c.name)["outputs"] == {"got": b.name}


def test_missing_output_fails_with_clear_message(cfg):
    ws = workspace.open_or_create(cfg, "w", "synthetic")
    s = ws.add("y", "test.read", {})
    res = ws.run_step(s.name, ApprovalStore(cfg.approvals_dir), {})
    assert res["state"] == "failed"
    assert "还没有产物 'value'" in str(res["note"])


def test_gate_blocks_until_human_approves_and_is_asked_once(cfg):
    ws = workspace.open_or_create(cfg, "w", "synthetic")
    store = ApprovalStore(cfg.approvals_dir)
    g = ws.add("g", "test.gate", {})
    assert ws.run_step(g.name, store, {})["state"] == "blocked"
    assert workspace.load(cfg, "w").run_step(g.name, store, {})["state"] == "blocked"
    [a] = store.all()
    assert a.explain == "训练 2 卡"
    store.decide(a.id, approve=True)
    assert workspace.load(cfg, "w").run_step(g.name, store, {})["state"] == "done"


def test_rejected_gate_fails(cfg):
    ws = workspace.open_or_create(cfg, "w", "synthetic")
    store = ApprovalStore(cfg.approvals_dir)
    g = ws.add("g", "test.gate", {})
    ws.run_step(g.name, store, {})
    store.decide("a0001", approve=False, reason="GPU 不够")
    res = ws.run_step(g.name, store, {})
    assert res["state"] == "failed"
    assert "GPU 不够" in str(res["note"])


def test_exception_recorded_and_reset(cfg):
    ws = workspace.open_or_create(cfg, "w", "synthetic")
    s = ws.add("x", "test.boom", {})
    res = ws.run_step(s.name, ApprovalStore(cfg.approvals_dir), {})
    assert res["state"] == "failed"
    assert (ws.dir / s.name / "error.txt").is_file()
    ws.reset(s.name)
    assert workspace.load(cfg, "w").open_steps()[0].name == s.name


def test_flows_resolve_rejects_unknown_flow_and_args():
    with pytest.raises(flows.FlowError, match="动作表"):
        flows.resolve("synthetic", "rm_rf", {})
    with pytest.raises(flows.FlowError, match="不接受参数"):
        flows.resolve("synthetic", "dp_rsi", {"dpolicy": {"protocol": {}}})
    kind, params = flows.resolve("synthetic", "dp_rsi", {"rounds": 2})
    assert kind == "dpolicy.rsi"
    assert params["rounds"] == 2
    assert params["dpolicy"]["protocol"]["by"] == "group"  # 模板默认值保留
    with pytest.raises(flows.FlowError, match="不需要"):
        flows.resolve("synthetic", "score_base", {})


def test_suggest_order():
    art: dict = {}
    order = []
    steps = [
        ("prepare_data", {"rows_dir": "/d"}),
        ("train_base", {"base_model": "BASE", "models": {"base": "BASE"}}),
        ("data_rsi", {"arms": [{"hash": "h1"}]}),
        ("retrain", {"models": {"base": "BASE", "h1": "A1"}}),
        ("data_compare", {"compared": ["base", "h1"], "model": "A1"}),
        ("dp_rsi", {"dp_model": "A1"}),
        ("final_test", {"dp_headline": {}}),
    ]
    for flow, produced in steps:
        nxt = flows.suggest(art)
        assert nxt is not None
        order.append(nxt[0])
        assert nxt[0] == flow
        art.update(produced)
    assert flows.suggest(art) is None
    # 对比选出的模型变了 → 重新跑 DP
    art["model"] = "A2"
    assert flows.suggest(art)[0] == "dp_rsi"
