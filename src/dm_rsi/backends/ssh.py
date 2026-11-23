"""远程后端（ADR 0001 P4）：``ssh``（学校 A100 服务器这类直接登录的机器）和 ``pbs``（Borealis 这类集群）。

两者都只通过 ``ssh <host> <命令>`` 访问远端（``ssh_cli`` 可替换，测试用一个在本机执行命令的假 ssh）。
提交到远端要经过 ``remote_submit`` 审批：后端的 ``requires_approval`` 为真，由步骤在提交前登记审批
（:func:`dm_rsi.steps.training.submit_gpu`）。

约定：

- 远端路径（``cwd``、产物）由流程配置给出，必须是远端大盘上的绝对路径；后端只负责运行和取日志。
- ``ssh``：``nohup`` 后台运行，退出码写 ``<remote_dir>/<name>.rc``；不分配 GPU（远端机器上用
  ``CUDA_VISIBLE_DEVICES`` 由配置的 ``gpus`` 指定），``max_gpus`` 照样限制。
- ``pbs``：生成作业脚本（``#PBS -P/-l select=1:ngpus=N/-l walltime``）上传后 ``qsub``；
  状态用 ``qstat -x -f -F json``（F = 结束，看 ``Exit_status``）。
"""

from __future__ import annotations

import json
import shlex
import subprocess
from typing import TYPE_CHECKING, cast

from dm_rsi.backends.base import BackendError, JobRef, JobSpec, JobState

if TYPE_CHECKING:
    from pathlib import Path


class _Remote:
    """``ssh <host> <cmd>`` 和 ``scp``。"""

    def __init__(self, host: str, ssh_cli: str, remote_dir: str) -> None:
        self.host: str = host
        self.ssh_cli: str = ssh_cli
        self.remote_dir: str = remote_dir

    def sh(self, cmd: str, *, stdin: str | None = None) -> str:
        proc = subprocess.run(
            [self.ssh_cli, self.host, cmd],
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if proc.returncode != 0:
            msg = f"ssh {self.host} 失败（{proc.returncode}）：{proc.stderr.strip()[-500:]}"
            raise BackendError(msg)
        return proc.stdout

    def put(self, text: str, path: str) -> None:
        self.sh(
            f"mkdir -p {shlex.quote(path.rsplit('/', 1)[0])} && cat > {shlex.quote(path)}",
            stdin=text,
        )

    def get_dir(self, path: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        out = self.sh(f"cd {shlex.quote(path)} && tar cf - . | base64")
        import base64  # noqa: PLC0415 -- 只有取日志时用
        import io  # noqa: PLC0415
        import tarfile  # noqa: PLC0415

        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(out)), mode="r:") as tf:
            tf.extractall(dest, filter="data")


def _env_prefix(spec: JobSpec) -> str:
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in spec.env.items())


class SshBackend:
    """直接登录的远程机器。"""

    requires_approval = True

    def __init__(
        self, name: str, *, host: str, remote_dir: str, max_gpus: int, ssh_cli: str = "ssh"
    ) -> None:
        """``remote_dir``：远端放作业脚本和日志的目录。"""
        self.name: str = name
        self.max_gpus: int = max_gpus
        self.r: _Remote = _Remote(host, ssh_cli, remote_dir)

    def submit(self, spec: JobSpec) -> JobRef:
        """``nohup`` 后台运行。"""
        if spec.gpus > self.max_gpus:
            msg = f"{spec.name} 要 {spec.gpus} 张卡，超过后端 {self.name} 的上限 {self.max_gpus}"
            raise BackendError(msg)
        d = f"{self.r.remote_dir}/{spec.name}"
        script = (
            "#!/usr/bin/env bash\n"
            f"cd {shlex.quote(str(spec.cwd))}\n"
            f"{_env_prefix(spec)} {shlex.join(spec.argv)} > {shlex.quote(d)}/out 2> {shlex.quote(d)}/err\n"
            f"echo $? > {shlex.quote(d)}/rc\n"
        )
        self.r.put(script, f"{d}/job.sh")
        q = shlex.quote(d)
        pid = self.r.sh(
            f"rm -f {q}/rc && nohup bash {q}/job.sh > /dev/null 2>&1 < /dev/null & echo $!"
        ).strip()
        return JobRef(backend=self.name, id=pid, meta={"dir": d})

    def status(self, job: JobRef) -> JobState:
        """Rc 文件 → 结束；进程在 → running；否则 unknown。"""
        d = shlex.quote(str(job.meta["dir"]))
        out = self.r.sh(
            f"if [ -f {d}/rc ]; then cat {d}/rc; elif kill -0 {int(job.id)} 2>/dev/null; then echo running; else echo unknown; fi"
        ).strip()
        if out == "running":
            return JobState.RUNNING
        if out == "unknown":
            return JobState.UNKNOWN
        return JobState.DONE if out == "0" else JobState.FAILED

    def fetch(self, job: JobRef, dest: Path) -> None:
        """取回作业目录（脚本、stdout、stderr、rc）。"""
        self.r.get_dir(str(job.meta["dir"]), dest)

    def cancel(self, job: JobRef) -> None:
        """杀进程。"""
        self.r.sh(f"kill {int(job.id)} 2>/dev/null || true")


_PBS_STATES = {
    "Q": JobState.QUEUED,
    "H": JobState.QUEUED,
    "W": JobState.QUEUED,
    "R": JobState.RUNNING,
    "E": JobState.RUNNING,
}


class PbsBackend:
    """PBS 集群（经登录节点 ``qsub``）。"""

    requires_approval = True

    def __init__(  # noqa: PLR0913 -- 配置项，keyword-only
        self,
        name: str,
        *,
        host: str,
        remote_dir: str,
        project: str,
        max_gpus: int,
        walltime: str = "02:00:00",
        ssh_cli: str = "ssh",
    ) -> None:
        """``project``：``#PBS -P``；``walltime``：默认上限，作业的 ``timeout`` 优先。"""
        self.name: str = name
        self.project: str = project
        self.max_gpus: int = max_gpus
        self.walltime: str = walltime
        self.r: _Remote = _Remote(host, ssh_cli, remote_dir)

    def script(self, spec: JobSpec) -> str:
        """PBS 作业脚本。"""
        if spec.gpus > self.max_gpus:
            msg = f"{spec.name} 要 {spec.gpus} 张卡，超过后端 {self.name} 的上限 {self.max_gpus}"
            raise BackendError(msg)
        d = f"{self.r.remote_dir}/{spec.name}"
        wall = self.walltime
        if spec.timeout:
            h, rem = divmod(spec.timeout, 3600)
            wall = f"{h:02d}:{rem // 60:02d}:{rem % 60:02d}"
        select = f"select=1:ngpus={spec.gpus}" if spec.gpus else "select=1:ncpus=4"
        return (
            "#!/bin/bash\n"
            f"#PBS -P {self.project}\n"
            f"#PBS -N {spec.name[:15]}\n"
            f"#PBS -l {select}\n"
            f"#PBS -l walltime={wall}\n"
            f"#PBS -o {d}/out\n#PBS -e {d}/err\n"
            "set -eo pipefail\n"
            f"cd {shlex.quote(str(spec.cwd))}\n"
            f"{_env_prefix(spec)} {shlex.join(spec.argv)}\n"
        )

    def submit(self, spec: JobSpec) -> JobRef:
        """上传脚本并 ``qsub``。"""
        d = f"{self.r.remote_dir}/{spec.name}"
        self.r.put(self.script(spec), f"{d}/job.pbs")
        jobid = self.r.sh(f"qsub {shlex.quote(d)}/job.pbs").strip()
        if not jobid:
            msg = "qsub 没有返回作业号"
            raise BackendError(msg)
        return JobRef(backend=self.name, id=jobid, meta={"dir": d})

    def status(self, job: JobRef) -> JobState:
        """``qstat -x -f -F json``。"""
        obj = cast(
            "dict[str, object]", json.loads(self.r.sh(f"qstat -x -f -F json {shlex.quote(job.id)}"))
        )
        jobs = cast("dict[str, dict[str, object]]", obj.get("Jobs") or {})
        info = next(iter(jobs.values()), None)
        if info is None:
            return JobState.UNKNOWN
        st = str(info.get("job_state"))
        if st == "F":
            return JobState.DONE if str(info.get("Exit_status")) == "0" else JobState.FAILED
        return _PBS_STATES.get(st, JobState.UNKNOWN)

    def fetch(self, job: JobRef, dest: Path) -> None:
        """取回作业目录。"""
        self.r.get_dir(str(job.meta["dir"]), dest)

    def cancel(self, job: JobRef) -> None:
        """``qdel``。"""
        self.r.sh(f"qdel {shlex.quote(job.id)}")
