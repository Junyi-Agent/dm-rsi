"""ADR 0001（修订）：meta loop。观察、推理、决策由 dm-rsi 做，行动执行决策选中的那段流程。"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from xtra_rsi import Stage, StageContext, run_loop

from dm_rsi import flows, profile, workspace
from dm_rsi.approvals import ApprovalStore, Kind, State
from dm_rsi.yamlfile import read

FAKE_CLAUDE = r"""#!__PY__
import json, shutil, sys
from pathlib import Path
args = sys.argv[1:]
dirs = [args[i + 1] for i, a in enumerate(args) if a == "--add-dir"]
pkg, writes = Path(dirs[0]), Path(dirs[-1])
assert "Bash" not in args[args.index("--tools") + 1]
if (pkg / "dgen").is_dir():
    shutil.copy(pkg / "configs" / "recipes" / "hard_all.yaml", writes / "agent_hard.yaml")
else:
    shutil.copy(pkg / "configs" / "policies" / "example_rolling_platt.yaml", writes / "agent_rolling.yaml")
print(json.dumps({"result": "诊断：……", "total_cost_usd": 0.12, "is_error": False,
                  "usage": {"input_tokens": 10, "output_tokens": 5}}))
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    p = tmp_path / "claude"
    p.write_text(FAKE_CLAUDE.replace("__PY__", sys.executable))
    p.chmod(0o755)
    monkeypatch.setenv("DMRSI_CLAUDE", str(p))
    return p


@pytest.fixture
def meta_env(cfg_with_env, gh, monkeypatch):
    monkeypatch.setenv("DMRSI_HOME", str(cfg_with_env.home))
    monkeypatch.setenv("DMRSI_WORKSPACE", "t")
    monkeypatch.setenv("DMRSI_TEMPLATE", "synthetic")
    monkeypatch.setattr(profile, "_gh", lambda: gh)
    return cfg_with_env


def _ctx(stage, upstream):
    return StageContext(stage=stage, iteration=0, upstream=upstream)


def _dirty(cfg):
    out = []
    for d in cfg.directions.values():
        if d.submodule:
            cmd = ["git", "-C", str(cfg.root / d.submodule), "status", "--porcelain"]
            out += subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()
    return out


def test_decision_priorities():
    nxt = {"kind": "next_flow", "flow": "data_rsi", "args": {}, "evidence": "缺", "suggest": "flow"}
    opened = {
        "kind": "open_execution",
        "step": "003-retrain",
        "flow": "retrain",
        "suggest": "resume",
    }
    waiting = {"kind": "waiting_human", "step": "003-retrain", "flow": "retrain", "suggest": "wait"}
    failed = {
        "kind": "flow_failed",
        "step": "004-dp_rsi",
        "flow": "dp_rsi",
        "owner": "decision-policy",
        "suggest": "wait",
    }

    def plan(*fs, lessons=()):
        up = {"findings": list(fs), "lessons": list(lessons)}
        return profile.decision(_ctx(Stage.DECISION, up))["plan"]

    assert plan(nxt) == {"action": "flow", "flow": "data_rsi", "args": {}, "why": "缺"}
    assert plan(opened, nxt)["action"] == "resume"  # 同一时间只执行一段
    assert plan(waiting, nxt)["action"] == "wait"
    # 失败了就停下等人，不起草 issue，也不绕过去跑下一段
    p = plan(failed, nxt)
    assert p["action"] == "wait"
    assert "dp_rsi 失败" in p["why"]
    assert "decision-policy" in p["why"]


def test_action_refuses_unknown_action_and_flow(meta_env):
    out = profile.action(_ctx(Stage.ACTION, {"plan": {"action": "rm -rf"}}))
    assert "拒绝" in out["result"]["error"]
    out = profile.action(
        _ctx(Stage.ACTION, {"plan": {"action": "flow", "flow": "train_everything"}})
    )
    assert "动作表" in out["result"]["error"]
    plan = {"action": "flow", "flow": "dp_rsi", "args": {"dpolicy": {}}}
    out = profile.action(_ctx(Stage.ACTION, {"plan": plan}))
    assert "不接受参数" in out["result"]["error"]


@pytest.mark.slow
def test_meta_loop_runs_flows_in_action_and_stops_at_gate(meta_env, tmp_path):
    loop = tmp_path / "loop"
    records = run_loop(profile.profile(), iterations=20, run_dir=loop)
    plans = [r.outputs[Stage.DECISION]["plan"] for r in records]
    flows_run = [p["flow"] for p in plans if p["action"] == "flow"]
    assert flows_run == [
        "prepare_data",
        "train_base",
        "data_rsi",
        "retrain",
        "data_compare",
        "dp_rsi",
        "final_test",
    ]
    assert plans[-1]["action"] == "wait"
    assert records[-1].last["stop"] is True

    # 行动阶段的执行过程写进了本轮的 action.yaml（面板展开行动就能看到）
    act = read(loop / "iter-0002" / "action.yaml")
    assert act["result"]["flow"] == "data_rsi"
    assert any("generate" in e["text"] for e in act["events"])

    [a] = ApprovalStore(meta_env.approvals_dir).all()
    assert a.kind is Kind.FINAL_TEST
    for words in ("合成数据", "val 上已经定下的结果", "批准：", "拒绝：", "建议："):
        assert words in a.explain

    # 测试夹具里的临时审批队列，扮演人
    ApprovalStore(meta_env.approvals_dir).decide(a.id, approve=True)
    records = run_loop(profile.profile(), iterations=5, run_dir=tmp_path / "loop2")
    plans = [r.outputs[Stage.DECISION]["plan"] for r in records]
    assert plans[0]["action"] == "resume"
    assert "都完成了" in plans[-1]["why"]
    ws = workspace.load(meta_env, "t")
    final = ws.latest_output("dp_headline")
    assert set(final) >= {"baseline", "incumbent"}
    assert ApprovalStore(meta_env.approvals_dir).get(a.id).state is State.DONE
    assert _dirty(meta_env) == []


@pytest.mark.slow
def test_failed_flow_stops_loop_without_drafting_issue(meta_env, gh, tmp_path, monkeypatch):
    # 合成 val 只有 1 个 step：用 dpolicy 默认的 by: step 协议必然失败（真实的契约问题）
    monkeypatch.setitem(flows.TEMPLATES["synthetic"]["flows"], "dp_rsi", {"features": "from_rows"})
    records = run_loop(profile.profile(), iterations=20, run_dir=tmp_path / "loop")
    actions = [r.outputs[Stage.DECISION]["plan"]["action"] for r in records]
    assert "draft_issue" not in actions
    assert not [x for x in ApprovalStore(meta_env.approvals_dir).all() if x.kind is Kind.ISSUE]
    assert gh.created == []
    assert records[-1].last["stop"] is True
    why = records[-1].outputs[Stage.DECISION]["plan"]["why"]
    assert "dp_rsi 失败" in why
    assert "work reset" in why


@pytest.mark.slow
def test_agent_judge_writes_recipes_and_records_usage(cfg_with_env, fake_claude):
    ws = workspace.open_or_create(
        cfg_with_env, "agentic", "synthetic", {"judge": "agent", "seed": 0}
    )
    store = ApprovalStore(cfg_with_env.approvals_dir)
    for flow in ("prepare_data", "train_base", "data_rsi"):
        kind, params = flows.resolve("synthetic", flow, {})
        res = ws.run_step(ws.add(flow, kind, params).name, store, {})
        assert res["state"] == "done", res
    recipes = sorted(p.name for p in (ws.dir / "data-rsi" / "out" / "recipes").glob("*.yaml"))
    assert recipes == ["agent_hard.yaml"]
    [usage] = [json.loads(x) for x in (ws.dir / "usage.jsonl").read_text().splitlines()]
    assert usage["cost_usd"] == 0.12
    assert usage["direction"] == "data"
