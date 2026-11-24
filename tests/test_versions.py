"""ADR 0001（2026-11-24 第二次修订）：借鉴 Borui 的版本登记、升级规则、停止条件和交接时重算。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from xtra_rsi import Stage, StageContext

from dm_rsi import profile, subenv, versions, workspace
from dm_rsi.steps import contracts

if TYPE_CHECKING:
    from pathlib import Path


def _dp_run(root: Path, name: str, objectives: list[float]) -> dict[str, object]:
    """假的 dpolicy 输出：基线 + 依次被接受的候选。"""
    d = root / name
    (d / "r0").mkdir(parents=True, exist_ok=True)
    recs = [{"source": "baseline", "name": "baseline", "hash": "b", "objective": objectives[0]}]
    recs += [
        {"source": "proposal", "name": f"p{i}", "hash": f"h{i}", "objective": o, "accepted": True}
        for i, o in enumerate(objectives[1:])
    ]
    (d / "r0" / "archive.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return {"dp_model": name, "dp_dir": str(d), "dp_configs": [str(d / "dpolicy.r0.yaml")]}


def test_promotion_rule_and_patience(cfg, tmp_path):
    ws = workspace.open_or_create(
        cfg, "v", "synthetic", {"stop": {"min_delta": 0.01, "max_rounds": 10}}
    )
    reg = lambda step, objs: versions.register(ws, step, _dp_run(tmp_path, step, objs))  # noqa: E731

    e = reg("001-dp_rsi", [0.40, 0.50])
    assert (e["decision"], e["version"], e["objective"]) == ("promote", "V1", 0.50)
    assert reg("001-dp_rsi", [0.99])["version"] == "V1"  # 同一次执行只登记一次

    e = reg("002-dp_rsi", [0.505])  # 高了，但没超过 min_delta
    assert e["decision"] == "keep"
    assert "保留 V1" in e["reason"]
    assert versions.stop_reason(ws) is None

    e = reg("003-dp_rsi", [0.45, 0.60])
    assert (e["decision"], e["version"]) == ("promote", "V2")
    assert versions.best(ws)["dp_model"] == "003-dp_rsi"

    reg("004-dp_rsi", [0.55])
    assert versions.stop_reason(ws) is None
    reg("005-dp_rsi", [0.58])
    assert "patience" in versions.stop_reason(ws)
    s = versions.summary(ws)
    assert s["best"]["version"] == "V2"
    assert s["rounds"] == 5
    assert s["streak"] == 2


def test_max_rounds(cfg, tmp_path):
    ws = workspace.open_or_create(cfg, "m", "synthetic", {"stop": {"max_rounds": 2}})
    versions.register(ws, "a", _dp_run(tmp_path, "a", [0.1]))
    versions.register(ws, "b", _dp_run(tmp_path, "b", [0.2]))
    assert "max_rounds" in versions.stop_reason(ws)


def _ctx(stage, upstream):
    return StageContext(stage=stage, iteration=0, upstream=upstream)


def test_stop_condition_sends_loop_to_final_test_and_action_refuses_others(
    cfg, gh, tmp_path, monkeypatch
):
    monkeypatch.setenv("DMRSI_HOME", str(cfg.home))
    monkeypatch.setenv("DMRSI_WORKSPACE", "s")
    monkeypatch.setattr(profile, "_gh", lambda: gh)
    ws = workspace.open_or_create(cfg, "s", "synthetic", {"stop": {"max_rounds": 1}})
    versions.register(ws, "001-dp_rsi", _dp_run(tmp_path, "x", [0.3]))

    # 推理：产物看起来还该跑 data_rsi，但停止条件命中 → 指向 final_test
    up = {"artifacts": {"rows_dir": "/r", "base_model": "m", "models": {"base": "m"}}}
    up["versions"] = versions.summary(ws)
    up["template"] = "synthetic"
    out = profile.reasoning(_ctx(Stage.REASONING, up))
    [nxt] = [f for f in out["findings"] if f["kind"] == "next_flow"]
    assert nxt["flow"] == "final_test"
    assert "停止条件" in nxt["evidence"]

    # 行动：LLM 决策要跑别的流程也被拒
    plan = {"action": "flow", "flow": "data_rsi", "args": {}}
    res = profile.action(_ctx(Stage.ACTION, {"plan": plan}))["result"]
    assert "停止条件已命中" in res["error"]


def test_check_reported_names_field_and_both_values():
    mine = {"base": {"rows": 10, "pos": 2, "auprc": 0.5, "auroc": 0.7, "brier": 0.1}}
    contracts.check_reported(
        {"base": {"rows": 10, "pos": 2, "auprc": 0.5 + 1e-9, "auroc": 0.7, "f1": 0.3}},
        mine,
        "x",
    )
    with pytest.raises(contracts.ContractError, match=r"base\.auprc：报告 0\.6，dm-rsi 重算 0\.5"):
        contracts.check_reported({"base": {"auprc": 0.6}}, mine, "x")
    with pytest.raises(contracts.ContractError, match="没有这个模型"):
        contracts.check_reported({}, mine, "x")


def test_recompute_reads_score_files(cfg_with_env, tmp_path):
    d = tmp_path / "scores"
    d.mkdir()
    subenv.run_code(
        cfg_with_env,
        None,
        "import sys, pandas as pd\n"
        "pd.DataFrame({'id': list('abcd'), 'y': [0, 1, 0, 1], 'p': [0.1, 0.9, 0.6, 0.4]})"
        ".to_parquet(sys.argv[1] + '/m_val.parquet', index=False)",
        [str(d)],
    )
    got = contracts.recompute(cfg_with_env, d, {"base": "m"}, "val")
    assert got["base"]["rows"] == 4
    assert got["base"]["pos"] == 2
    assert got["base"]["auroc"] == pytest.approx(0.75)
    assert got["base"]["brier"] == pytest.approx((0.01 + 0.01 + 0.36 + 0.36) / 4)
