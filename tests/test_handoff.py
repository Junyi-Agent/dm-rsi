"""ADR 0001（2026-11-24 第三次修订）：sub-RSI → dm-rsi 的交接集中在 steps/handoff.py。"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from dm_rsi.steps import handoff
from dm_rsi.steps.contracts import ContractError

SRC = Path(__file__).resolve().parents[1] / "src" / "dm_rsi"

# sub-RSI 的内部文件名：只能在 steps/handoff.py 里出现在代码里（说明文字、给 agent 的提示词不算）
INTERNAL = {
    "archive.jsonl",
    "incumbent.yaml",
    "compare.json",
    "build_report.json",
    "final_test.json",
    "consensus.yaml",
    "trials",
}


def test_only_handoff_reads_sub_rsi_internal_files():
    offenders = []
    for p in SRC.rglob("*.py"):
        if p.name == "handoff.py":
            continue
        offenders.extend(
            f"{p.relative_to(SRC)}:{node.lineno} {node.value!r}"
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
            if isinstance(node, ast.Constant) and node.value in INTERNAL
        )
    assert not offenders, "经 steps/handoff.py 读 sub-RSI 的文件：\n" + "\n".join(offenders)


def test_need_lists_every_problem():
    with pytest.raises(ContractError) as e:
        handoff.need({"a": "x", "b": True}, "f", a=int, b=int, c=str, d=(str, type(None)))
    msg = str(e.value)
    assert "a 应该是整数" in msg
    assert "b 应该是整数，实际是 bool" in msg  # bool 不算整数
    assert "缺字段 c" in msg
    assert "缺字段 d" not in msg  # 可选字段缺了没关系


def _trsi_run(tmp_path: Path) -> tuple[dict[str, object], Path]:
    out = tmp_path / "trsi"
    home = out / "r1"
    (out / "trials").mkdir(parents=True)
    home.mkdir()
    events = [
        {"event": "incumbent", "candidate": "c0", "why": "start"},
        {"event": "use", "trial": "c0-s0"},
    ]
    (home / "archive.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (out / "trials" / "c0-s0.json").write_text(
        json.dumps(
            {
                "status": "done",
                "seed": 0,
                "run_dir": "/runs/a",
                "result": {"checkpoint": "ckpt/800", "metrics": {"auprc": 0.8}},
            }
        )
    )
    (home / "incumbent.yaml").write_text(
        yaml.safe_dump(
            {"candidate_id": "c0", "candidate": {"base": "elephant-1", "recipe": "/r/a.yaml"}}
        )
    )
    return {"out_dir": str(out), "run_name": "r1"}, home


def test_trsi_compat_and_official_handoff(tmp_path):
    run, home = _trsi_run(tmp_path)
    sch = handoff.TrsiRecords(run, tmp_path / "run.yaml").scheme()
    assert sch.source == "compat"
    assert sch.is_start
    assert sch.checkpoints["0"]["checkpoint"] == "ckpt/800"

    official = {
        "handoff": 1,
        "kind": "training",
        "scheme": {
            "candidate_id": "c9",
            "base": "mammoth-1",
            "recipe": "/r/b.yaml",
            "overrides": {"onejev.lr": {"value": 2e-4, "reason": "曲线还在变好"}},
            "is_start": False,
            "note": "换基座",
        },
        "checkpoints": {"0": {"run_dir": "/runs/b", "checkpoint": "ckpt/400", "metrics": None}},
    }
    (home / "handoff.yaml").write_text(yaml.safe_dump(official, allow_unicode=True))
    sch = handoff.TrsiRecords(run, tmp_path / "run.yaml").scheme()
    assert sch.source == "handoff.yaml"
    assert sch.candidate_id == "c9"
    assert sch.to_scheme()["overrides"] == {"onejev.lr": {"value": 2e-4, "reason": "曲线还在变好"}}

    official["handoff"] = 2
    (home / "handoff.yaml").write_text(yaml.safe_dump(official, allow_unicode=True))
    with pytest.raises(ContractError, match="handoff: 1"):
        handoff.TrsiRecords(run, tmp_path / "run.yaml").scheme()


def test_trsi_event_missing_field_is_a_contract_error(tmp_path):
    run, home = _trsi_run(tmp_path)
    with (home / "archive.jsonl").open("a") as f:
        f.write(json.dumps({"event": "launch", "trial": "c0-s1"}) + "\n")  # 少了 gpus
    with pytest.raises(ContractError, match=r"第 3 个事件（launch）：缺字段 gpus"):
        handoff.TrsiRecords(run, tmp_path / "run.yaml")


def test_dpolicy_incumbent_compat_and_official(tmp_path):
    r0 = tmp_path / "r0"
    r0.mkdir()
    recs = [
        {"source": "baseline", "name": "baseline", "hash": "b", "objective": 0.5},
        {"source": "proposal", "name": "p1", "hash": "h1", "objective": 0.55, "accepted": True,
         "vs_incumbent": {"diff": 0.05}},
        {"source": "proposal", "name": "p2", "hash": "h2", "objective": 0.4},
    ]  # fmt: skip
    (r0 / "archive.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    inc = handoff.dpolicy_incumbent(r0)
    assert inc is not None
    assert (inc.name, inc.objective, inc.accepted, inc.evaluated) == ("p1", 0.55, 1, 2)
    assert inc.baseline_objective == 0.5
    # 诊断要取 incumbent 的，不是最后评估的那个（p2 没被接受）
    (r0 / "eval" / "h1").mkdir(parents=True)
    (r0 / "eval" / "h1" / "diagnosis.md").write_text("incumbent 的诊断")
    assert handoff.dpolicy_diagnosis(r0) == r0 / "eval" / "h1" / "diagnosis.md"

    (r0 / "handoff.yaml").write_text(
        yaml.safe_dump(
            {
                "handoff": 1,
                "kind": "decision-policy",
                "incumbent": {"name": "p7", "hash": "h7", "objective": 0.6},
                "baseline": {"objective": 0.5},
                "evaluated": 7,
                "accepted": 2,
            }
        )
    )
    inc = handoff.dpolicy_incumbent(r0)
    assert inc is not None
    assert (inc.name, inc.source, inc.is_baseline) == ("p7", "handoff.yaml", False)


def test_dgen_compare_checks_primary_diff_has_accept_flag(tmp_path):
    out = tmp_path
    p = handoff.compare_path(out, "t")
    p.parent.mkdir(parents=True)
    res = {
        "n": 10,
        "n_pos": 2,
        "table": [{"run": "base", "auprc": 0.4}, {"run": "a", "auprc": 0.5}],
        "diffs": [{"run": "a", "metric": "auprc", "diff": 0.1, "lo": 0.01, "hi": 0.2}],
    }
    p.write_text(json.dumps({"primary": "auprc", "result": res}))
    with pytest.raises(ContractError, match="passes_accept"):
        handoff.dgen_compare(out, "t")
    res["diffs"][0]["passes_accept"] = True
    p.write_text(json.dumps({"primary": "auprc", "result": res}))
    c = handoff.dgen_compare(out, "t")
    assert c.table["a"]["auprc"] == 0.5
    assert c.diffs[0].passes_accept
