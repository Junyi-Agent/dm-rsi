"""ADR 0001 P2：Elliptic 训练步骤的调度逻辑（假 ojh、内存 GPU 后端，不碰真实 GPU）。"""

from __future__ import annotations

import sys
from pathlib import Path

from dm_rsi import pipeline, steps, yamlfile
from dm_rsi.approvals import ApprovalStore, Kind, State
from dm_rsi.backends.base import JobRef, JobState
from dm_rsi.pipeline import Outcome, StepDef, StepState, step_kind

_ = steps  # 注册步骤种类

FAKE_OJH = r"""#!/usr/bin/env python3
import os, sys, yaml
from pathlib import Path
cmd, exp = sys.argv[1], Path(sys.argv[2])
if cmd == "prepare":
    print("prepared")
elif cmd == "resolve":
    run = Path(os.environ["FAKE_OUT"]) / "run1"
    run.mkdir(parents=True, exist_ok=True)
    lock = run / "run.lock.yaml"
    lock.write_text(yaml.safe_dump({"run_dir": str(run)}))
    print(lock)
"""


class FakeGpu:
    name = "gpu"

    def __init__(self):
        self.submitted = []
        self.state = JobState.RUNNING

    def submit(self, spec):
        self.submitted.append(spec)
        return JobRef("gpu", f"j{len(self.submitted)}")

    def status(self, job):
        return self.state

    def fetch(self, job, dest):
        pass

    def cancel(self, job):
        pass


@step_kind("test.inputs")
def _inputs(ctx):
    return Outcome(
        StepState.DONE,
        outputs={"train": "/rows/train_sub40k.jsonl", "val_sub": "/rows/val_sub4000.jsonl"},
    )


def test_train_needs_approval_then_runs_in_foreground_on_gpu_backend(cfg, tmp_path, monkeypatch):
    ojh = tmp_path / "ojh"
    ojh.write_text(FAKE_OJH.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    ojh.chmod(0o755)
    monkeypatch.setenv("FAKE_OUT", str(tmp_path / "out"))
    steps = [
        StepDef("inputs", "test.inputs"),
        StepDef("train-base", "elliptic.train", ("inputs",), {"model": "BASE"}),
    ]
    run = pipeline.create(
        cfg, "ell", "t", steps, {"gpu_backend": "gpu", "ojh_bin": str(ojh), "ojh_python": "/py"}
    )
    store = ApprovalStore(cfg.approvals_dir)
    gpu = FakeGpu()

    res = run.advance(store, {"gpu": gpu})
    assert res["state"] == "blocked"
    [a] = store.all()
    assert a.kind is Kind.TRAIN
    assert gpu.submitted == []  # 没批准不提交

    store.decide(a.id, approve=True)
    res = pipeline.load(cfg, "ell").advance(store, {"gpu": gpu})
    assert res["state"] == "running"
    [spec] = gpu.submitted
    assert spec.gpus == 2
    assert spec.argv[1:3] == ["-m", "harness.runner"]  # 前台 runner，不是 ojh run（tmux）
    assert store.get(a.id).state is State.DONE

    exp = yamlfile.read(cfg.home / "ojh" / "work" / "ell-base" / "exp.yaml")
    assert exp["data"]["train"] == "/rows/train_sub40k.jsonl"
    machine = yamlfile.read(cfg.home / "ojh" / "machine.dmrsi.yaml")
    assert machine["gpu_source"] == "preassigned"

    # 轮询：仍在跑，不重复登记审批、不重复提交
    res = pipeline.load(cfg, "ell").advance(store, {"gpu": gpu})
    assert res["state"] == "running"
    assert len(store.all()) == 1
    assert len(gpu.submitted) == 1

    # 作业结束但 runner 没写 DONE：失败
    gpu.state = JobState.DONE
    res = pipeline.load(cfg, "ell").advance(store, {"gpu": gpu})
    assert res["state"] == "failed"

    # 写了 DONE：再推进一次（失败的步骤会重试，作业引用还在）就成功
    Path(tmp_path / "out" / "run1" / "DONE").write_text("exit_code: 0\n")
    res = pipeline.load(cfg, "ell").advance(store, {"gpu": gpu})
    assert res["state"] == "done"
    assert len(gpu.submitted) == 1
    out = pipeline.load(cfg, "ell").step_state("train-base")["outputs"]
    assert out["run_dirs"] == {"BASE": str(tmp_path / "out" / "run1")}
