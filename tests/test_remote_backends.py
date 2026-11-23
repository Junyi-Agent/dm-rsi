"""ADR 0001 P4：ssh / pbs 后端。用一个在本机执行命令的假 ssh 跑通 ssh 后端；pbs 用假 qsub / qstat。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from dm_rsi import backends
from dm_rsi.backends import (
    BackendError,
    JobSpec,
    JobState,
    PbsBackend,
    SshBackend,
    requires_approval,
)

FAKE_SSH = r"""#!/usr/bin/env bash
# fake ssh: ssh HOST CMD -> run CMD locally (PATH may include fake qsub/qstat)
shift
exec bash -c "$1"
"""


@pytest.fixture
def fake_ssh(tmp_path):
    p = tmp_path / "ssh"
    p.write_text(FAKE_SSH)
    p.chmod(0o755)
    return str(p)


def test_ssh_backend_runs_and_fetches(tmp_path, fake_ssh):
    b = SshBackend(
        "school", host="a100", remote_dir=str(tmp_path / "remote"), max_gpus=2, ssh_cli=fake_ssh
    )
    assert requires_approval(b)
    ref = b.submit(
        JobSpec(
            "j1",
            [sys.executable, "-c", "print('remote hi')"],
            tmp_path,
            tmp_path / "logs",
            env={"X": "1"},
        )
    )
    end = time.time() + 10
    while b.status(ref) is JobState.RUNNING and time.time() < end:
        time.sleep(0.05)
    assert b.status(ref) is JobState.DONE
    b.fetch(ref, tmp_path / "got")
    assert (tmp_path / "got" / "out").read_text().strip() == "remote hi"
    ref2 = b.submit(JobSpec("j2", ["false"], tmp_path, tmp_path / "logs"))
    end = time.time() + 10
    while b.status(ref2) is JobState.RUNNING and time.time() < end:
        time.sleep(0.05)
    assert b.status(ref2) is JobState.FAILED


def test_pbs_script_and_states(tmp_path, fake_ssh, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "qsub").write_text("#!/usr/bin/env bash\necho 1234.pbs01\n")
    state = tmp_path / "state.json"
    (bindir / "qstat").write_text(f"#!/usr/bin/env bash\ncat {state}\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{Path('/usr/bin')}:{Path('/bin')}")
    b = PbsBackend(
        "borealis",
        host="login",
        remote_dir=str(tmp_path / "r"),
        project="P1",
        max_gpus=2,
        ssh_cli=fake_ssh,
    )
    spec = JobSpec("train-base", ["ojh", "x"], Path("/scratch/w"), tmp_path, gpus=2, timeout=4200)
    text = b.script(spec)
    assert "#PBS -P P1" in text
    assert "#PBS -l select=1:ngpus=2" in text
    assert "#PBS -l walltime=01:10:00" in text
    ref = b.submit(spec)
    assert ref.id == "1234.pbs01"
    state.write_text(json.dumps({"Jobs": {"1234.pbs01": {"job_state": "Q"}}}))
    assert b.status(ref) is JobState.QUEUED
    state.write_text(json.dumps({"Jobs": {"1234.pbs01": {"job_state": "F", "Exit_status": 0}}}))
    assert b.status(ref) is JobState.DONE
    state.write_text(json.dumps({"Jobs": {"1234.pbs01": {"job_state": "F", "Exit_status": 271}}}))
    assert b.status(ref) is JobState.FAILED
    with pytest.raises(BackendError, match="上限"):
        b.script(JobSpec("big", ["x"], tmp_path, tmp_path, gpus=8))


def test_make_remote_backends_validate_config(tmp_path):
    with pytest.raises(BackendError, match="remote_dir"):
        backends.make("school", {"kind": "ssh", "host": "a100"}, state_dir=tmp_path)
    with pytest.raises(BackendError, match="project"):
        backends.make(
            "b", {"kind": "pbs", "host": "login", "remote_dir": "/scratch/x"}, state_dir=tmp_path
        )
    b = backends.make(
        "b",
        {"kind": "pbs", "host": "login", "remote_dir": "/s", "project": "P"},
        state_dir=tmp_path,
    )
    assert isinstance(b, PbsBackend)
