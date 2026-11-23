from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from dm_rsi import backends
from dm_rsi.backends import BackendError, JobSpec, JobState, LocalBackend, TaskmgrBackend


def _wait(b, ref, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        st = b.status(ref)
        if st in (JobState.DONE, JobState.FAILED):
            return st
        time.sleep(0.05)
    return b.status(ref)


def test_local_run_and_submit(tmp_path):
    b = LocalBackend()
    spec = JobSpec("hello", [sys.executable, "-c", "print('hi')"], tmp_path, tmp_path / "logs")
    assert b.run(spec).stdout.strip() == "hi"
    ref = b.submit(
        JobSpec(
            "bg", [sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path, tmp_path / "logs"
        )
    )
    assert _wait(b, ref) is JobState.FAILED
    ref = b.submit(JobSpec("bg2", ["true"], tmp_path, tmp_path / "logs"))
    assert _wait(b, ref) is JobState.DONE


def test_local_refuses_gpu(tmp_path):
    with pytest.raises(BackendError, match="GPU"):
        LocalBackend().run(JobSpec("g", ["true"], tmp_path, tmp_path, gpus=1))


FAKE_TASK = r"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
log = Path(os.environ["FAKE_TASK_LOG"])
args = [a for a in sys.argv[1:] if a != "--plain"]
with log.open("a") as f:
    f.write(json.dumps(args) + "\n")
cmd = args[0]
if cmd == "pack":
    out = Path(args[args.index("-o") + 1]); out.mkdir(parents=True, exist_ok=True)
    (out / "task.json").write_text("{}"); print(out)
elif cmd == "submit":
    print(json.dumps({"host": "h", "jobid": "j123", "task_id": "DM-0001", "state": "queued"}))
elif cmd == "status":
    print(json.dumps({"host": "h", "jobid": args[1], "state": os.environ.get("FAKE_STATE", "running")}))
elif cmd in ("fetch", "cancel"):
    print("ok")
"""


@pytest.fixture
def fake_task(tmp_path, monkeypatch):
    p = tmp_path / "task"
    p.write_text(FAKE_TASK)
    p.chmod(0o755)
    log = tmp_path / "task.log"
    monkeypatch.setenv("FAKE_TASK_LOG", str(log))
    return p, log


def test_taskmgr_pack_submit_status(tmp_path, fake_task, monkeypatch):
    cli, log = fake_task
    b = TaskmgrBackend(
        "gpu",
        host="xtrah200",
        max_gpus=2,
        min_free_gb=40,
        state_dir=tmp_path / "st",
        task_cli=str(cli),
    )
    spec = JobSpec(
        "train-base",
        ["ojh", "run", "exp.yaml"],
        Path("/data/w"),
        tmp_path / "logs",
        env={"A": "1"},
        gpus=2,
        timeout=3600,
    )
    ref = b.submit(spec)
    assert ref.id == "j123"
    assert ref.meta["host"] == "xtrah200"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    pack, submit = calls[0], calls[1]
    assert pack[0] == "pack"
    assert json.loads(pack[pack.index("--gpus") + 1]) == {"count": 2, "min_free_gb": 40}
    assert pack[pack.index("--entry") + 1] == "bash run.sh"
    assert "A=1" in pack
    assert pack[pack.index("--task-id") + 1] == "DM-0001"
    assert submit[:3] == ["submit", "--host", "xtrah200"]
    run_sh = (tmp_path / "st" / "src" / "train-base" / "run.sh").read_text()
    assert "cd /data/w" in run_sh
    assert "exec ojh run exp.yaml" in run_sh
    assert b.status(ref) is JobState.RUNNING
    monkeypatch.setenv("FAKE_STATE", "succeeded")
    assert b.status(ref) is JobState.DONE
    monkeypatch.setenv("FAKE_STATE", "timed_out")
    assert b.status(ref) is JobState.FAILED


def test_taskmgr_enforces_max_gpus(tmp_path, fake_task):
    cli, _ = fake_task
    b = TaskmgrBackend(
        "gpu", host="h", max_gpus=2, min_free_gb=40, state_dir=tmp_path, task_cli=str(cli)
    )
    with pytest.raises(BackendError, match="上限"):
        b.submit(JobSpec("big", ["x"], tmp_path, tmp_path, gpus=4))


def test_make_local_and_unknown(tmp_path):
    assert isinstance(backends.make("l", {"kind": "local"}, state_dir=tmp_path), LocalBackend)
    with pytest.raises(BackendError, match="未知"):
        backends.make("x", {"kind": "slurm"}, state_dir=tmp_path)
