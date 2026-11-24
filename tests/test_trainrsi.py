"""ADR 0001（2026-11-24 修订，方案 A）：Training RSI 选训练方案，之后的训练按它来。

用 trsi 自带的假 ojh（``tests/fake_ojh.py``：``run`` 立即写出一个完成的 harness run 目录），不碰 GPU。
"""

from __future__ import annotations

import sys

import pytest

from dm_rsi import flows, steps, workspace, yamlfile
from dm_rsi.approvals import ApprovalStore, Kind, State
from dm_rsi.workspace import Outcome, StepState, step_kind

_ = steps  # 注册步骤种类

FAKE_CLAUDE = r"""#!__PY__
import json, sys
from pathlib import Path
args = sys.argv[1:]
dirs = [args[i + 1] for i, a in enumerate(args) if a == "--add-dir"]
pkg, writes = Path(dirs[0]), Path(dirs[-1])
p = writes / "lr-2e-4.yaml"
if not p.exists():  # 第二次被问时不再提候选
    p.write_text(f'''base: elephant-1
recipe: {pkg}/configs/recipes/elephant_lora_1g.yaml
overrides:
  onejev.lr: {{value: 2.0e-4, reason: "曲线到最后还在变好"}}
note: 训练不够，学习率加倍
prediction: {{auprc: 0.9, p_accept: 0.2}}
''')
print(json.dumps({"result": "诊断：……", "total_cost_usd": 0.05, "is_error": False, "usage": {}}))
"""


@step_kind("test.rows")
def _rows(ctx):
    return Outcome(
        StepState.DONE,
        outputs={
            "rows_dir": "/rows",
            "val_rows": "/rows/val.jsonl",
            "train": "/rows/train_sub40k.jsonl",
            "val_sub": "/rows/val_sub4000.jsonl",
        },
    )


@pytest.fixture
def fake_ojh(cfg_with_env, tmp_path, monkeypatch):
    pkg = cfg_with_env.direction("training").package_dir(cfg_with_env.root)
    ojh = tmp_path / "ojh"
    ojh.write_text(f'#!/bin/sh\nexec {sys.executable} {pkg}/tests/fake_ojh.py "$@"\n')
    ojh.chmod(0o755)
    monkeypatch.setenv("FAKE_OJH_OUT", str(tmp_path / "ojh-out"))
    monkeypatch.setenv("FAKE_OJH_CALLS", str(tmp_path / "calls.txt"))
    return ojh


def _ws(cfg, ojh, judge="default"):
    ws = workspace.open_or_create(cfg, "ell", "elliptic", {"judge": judge, "ojh_bin": str(ojh)})
    store = ApprovalStore(cfg.approvals_dir)
    ws.run_step(ws.add("prepare_data", "test.rows", {}).name, store, {})
    kind, params = flows.resolve("elliptic", "train_rsi", {"train_gpu_ids": [0, 1]})
    return ws.add("train_rsi", kind, params).name, store


def _step(cfg, name, store):
    return workspace.load(cfg, "ell").run_step(name, store, {})


def _approve_pending(store):
    pending = [a for a in store.all() if a.state is State.PENDING]
    for a in pending:
        assert a.kind is Kind.TRAIN
        store.decide(a.id, approve=True)
    return pending


def test_every_trsi_launch_needs_approval_then_scheme_drives_training(
    cfg_with_env, fake_ojh, tmp_path
):
    cfg = cfg_with_env
    name, store = _ws(cfg, fake_ojh)

    res = _step(cfg, name, store)
    assert res["state"] == "blocked"
    [a] = store.all()
    assert "起点" in a.explain
    assert "批准：" in a.explain
    assert "预算：" in a.explain
    assert not (tmp_path / "calls.txt").exists()  # 没批准不训练

    # 一次只问一个：起点的 3 个种子各批一次
    for _ in range(3):
        assert len(_approve_pending(store)) == 1
        res = _step(cfg, name, store)
    assert len((tmp_path / "calls.txt").read_text().splitlines()) == 3
    assert res["state"] == "running"

    res = _step(cfg, name, store)  # collect 读到结果；default 模式不提候选，定为起点
    assert res["state"] == "done", res
    out = workspace.load(cfg, "ell").step_state(name)["outputs"]
    sch = out["train_scheme"]
    assert sch["base"] == "elephant-1"
    assert sch["is_start"] is True
    assert sch["recipe"].endswith("configs/recipes/elephant_lora_1g.yaml")
    assert set(out["train_checkpoints"]) == {"0", "1", "2"}
    assert all(a.state is State.DONE for a in store.all())

    # trsi 的环境：完整训练集，ojh 前 source dm-rsi 生成的环境变量文件
    exp = yamlfile.read(cfg.home / "work" / "ell" / name / "exp.yaml")
    assert exp["data"]["train"] == "/rows/train.jsonl"
    assert "export DMRSI_DATA=" in (cfg.home / "dmrsi-env.sh").read_text()

    # 之后的训练按选出的方案：1 卡配方，没有 fields.text
    ws = workspace.load(cfg, "ell")
    kind, params = flows.resolve("elliptic", "train_base", {})
    tb = ws.add("train_base", kind, params).name
    res = ws.run_step(tb, store, {})
    assert res["state"] == "blocked"
    texp = yamlfile.read(cfg.home / "ojh" / "work" / "ell-base" / "exp.yaml")
    assert texp["recipe"] == sch["recipe"]
    assert texp["gpus"] == 1
    assert "text" not in texp["fields"]
    [train] = [a for a in store.all() if a.state is State.PENDING]
    assert "Training RSI 方案" in train.title


def test_agent_candidate_passes_probe_gets_seeds_and_is_accepted(
    cfg_with_env, fake_ojh, tmp_path, monkeypatch
):
    claude = tmp_path / "claude"
    claude.write_text(FAKE_CLAUDE.replace("__PY__", sys.executable))
    claude.chmod(0o755)
    monkeypatch.setenv("DMRSI_CLAUDE", str(claude))
    # 假 ojh 里候选明显更好（区分度 3 vs 1）：探针过门槛 → 补种子 → 接受
    monkeypatch.setenv("FAKE_OJH_QUALITY", '{"elephant-1|onejev.lr=0.0002": 3.0}')
    cfg = cfg_with_env
    name, store = _ws(cfg, fake_ojh, judge="agent")
    states = []
    for _ in range(12):
        _approve_pending(store)
        res = _step(cfg, name, store)
        states.append(res["state"])
        if res["state"] == "done":
            break
    assert states[-1] == "done", states
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert len(calls) == 6  # 起点 3 个种子 + 候选的探针 + 补的 2 个种子
    assert sum(a.kind is Kind.TRAIN for a in store.all()) == 6  # 每次训练一个审批
    out = workspace.load(cfg, "ell").step_state(name)["outputs"]
    sch = out["train_scheme"]
    assert sch["is_start"] is False
    assert sch["overrides"]["onejev.lr"]["value"] == 2.0e-4
    report = (cfg.home / "trsi" / "elliptic-txhist" / f"ell-{name}" / "report.md").read_text()
    assert "accepted" in report
    assert (
        len(list((cfg.home / "work" / "ell" / name).glob("agent-*.json"))) == 2
    )  # 第二次不再提候选 → 结束

    # 之后的训练带上方案的 overrides（连同原因）
    ws = workspace.load(cfg, "ell")
    kind, params = flows.resolve("elliptic", "train_base", {})
    ws.run_step(ws.add("train_base", kind, params).name, store, {})
    texp = yamlfile.read(cfg.home / "ojh" / "work" / "ell-base" / "exp.yaml")
    assert texp["overrides"] == {"onejev.lr": {"value": 2.0e-4, "reason": "曲线到最后还在变好"}}


def test_suggest_puts_train_rsi_before_training_only_for_elliptic():
    assert flows.suggest({"rows_dir": "/r"}, "elliptic")[0] == "train_rsi"
    assert flows.suggest({"rows_dir": "/r"}, "synthetic")[0] == "train_base"
    assert flows.suggest({"rows_dir": "/r", "train_scheme": {}}, "elliptic")[0] == "train_base"
