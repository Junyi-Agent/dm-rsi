"""ADR 0001 P3：meta loop（xtra-rsi profile dm）和 agent 判断步骤。"""

from __future__ import annotations

import json
import sys

import pytest
from xtra_rsi import Stage, StageContext, run_loop

from dm_rsi import pipeline, profile
from dm_rsi.approvals import ApprovalStore, Kind
from dm_rsi.templates import synthetic

# 假 claude：从 --add-dir 的最后一个目录（writes）写一个配方 / 提案，打印 claude -p 的 JSON
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
    monkeypatch.setattr(profile, "_gh", lambda: gh)
    return cfg_with_env


def _ctx(stage, upstream):
    return StageContext(stage=stage, iteration=0, upstream=upstream)


def test_decision_prefers_advance_and_skips_issued_failures():
    up = {
        "findings": [
            {
                "kind": "step_failed",
                "run": "r",
                "step": "dp-rsi",
                "owner": "decision-policy",
                "suggest": "draft_issue",
            },
            {"kind": "runnable", "run": "r2", "step": "x", "suggest": "advance"},
        ],
        "lessons": [],
    }
    assert profile.decision(_ctx(Stage.DECISION, up))["plan"]["action"] == "advance"
    up["findings"] = up["findings"][:1]
    assert profile.decision(_ctx(Stage.DECISION, up))["plan"]["action"] == "draft_issue"
    up["lessons"] = ["issue:r/dp-rsi"]
    assert profile.decision(_ctx(Stage.DECISION, up))["plan"]["action"] == "wait"


def test_action_refuses_unknown_action():
    out = profile.action(_ctx(Stage.ACTION, {"plan": {"action": "rm -rf"}}))
    assert "拒绝" in out["result"]["error"]


@pytest.mark.slow
def test_meta_loop_runs_synthetic_pipeline_from_zero_and_stops_at_gate(meta_env, tmp_path):
    records = run_loop(profile.profile(), iterations=10, run_dir=tmp_path / "loop")
    runs = pipeline.list_runs(meta_env)
    assert len(runs) == 1
    run = runs[0]
    assert run.step_state("dp-rsi")["state"] == "done"
    assert run.step_state("final-test")["state"] == "blocked"
    [a] = ApprovalStore(meta_env.approvals_dir).all()
    assert a.kind is Kind.FINAL_TEST
    actions = [r.outputs[Stage.ACTION]["action"] for r in records]
    assert actions[0] == "new_run"
    assert "advance" in actions
    assert records[-1].last["stop"] is True
    assert (tmp_path / "loop" / "status.yaml").is_file()


@pytest.mark.slow
def test_failed_step_becomes_issue_draft_once(meta_env, gh, tmp_path):
    steps, params = synthetic()
    # 合成 val 只有 1 个 step：用 dpolicy 默认的 by: step 协议必然失败（真实的契约问题）
    steps = [
        pipeline.StepDef(s.name, s.kind, s.needs, {"features": "from_rows"})
        if s.name == "dp-rsi"
        else s
        for s in steps
    ]
    pipeline.create(meta_env, "broken", "synthetic", steps, params)
    records = run_loop(profile.profile(), iterations=10, run_dir=tmp_path / "loop")
    actions = [r.outputs[Stage.ACTION]["action"] for r in records]
    assert actions.count("draft_issue") == 1
    [a] = [x for x in ApprovalStore(meta_env.approvals_dir).all() if x.kind is Kind.ISSUE]
    assert str(a.detail["issue_title"]).startswith("[decision-policy]")
    assert gh.created == []  # 只进审批队列
    assert records[-1].last["stop"] is True


@pytest.mark.slow
def test_agent_judge_writes_recipes_and_records_usage(cfg_with_env, fake_claude):
    steps, params = synthetic()
    steps = steps[:3]  # make-data → score-base → data-rsi
    run = pipeline.create(cfg_with_env, "agentic", "synthetic", steps, {**params, "judge": "agent"})
    res = run.advance(ApprovalStore(cfg_with_env.approvals_dir), {})
    assert res["state"] == "done", res
    recipes = sorted(p.name for p in (run.dir / "data-rsi" / "out" / "recipes").glob("*.yaml"))
    assert recipes == ["agent_hard.yaml"]
    [usage] = [json.loads(x) for x in (run.dir / "usage.jsonl").read_text().splitlines()]
    assert usage["cost_usd"] == 0.12
    assert usage["direction"] == "data"
