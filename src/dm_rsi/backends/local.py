"""本机后端：CPU 步骤。``run`` 同步执行；``submit`` 后台执行，退出码写进 ``<log_dir>/<name>.rc``。"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

from dm_rsi.backends.base import BackendError, JobRef, JobSpec, JobState

# 子进程退出后把退出码写到 rc 文件；wrapper 本身不经过用户输入，argv 通过 "$@" 原样传递
_WRAPPER = 'rc="$1"; shift; "$@"; echo $? > "$rc"'


class LocalBackend:
    """本机 subprocess。"""

    def __init__(self, name: str = "local") -> None:
        """后端名写进 JobRef。"""
        self.name: str = name

    @staticmethod
    def _env(spec: JobSpec) -> dict[str, str]:
        return {**os.environ, **spec.env}

    def run(self, spec: JobSpec) -> subprocess.CompletedProcess[str]:
        """同步执行，stdout/stderr 同时写日志文件并返回。"""
        if spec.gpus:
            msg = f"{spec.name}: 本机后端不分配 GPU，GPU 步骤用 taskmgr 后端"
            raise BackendError(msg)
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            spec.argv,
            cwd=spec.cwd,
            env=self._env(spec),
            capture_output=True,
            text=True,
            timeout=spec.timeout,
            check=False,
        )
        (spec.log_dir / f"{spec.name}.out").write_text(proc.stdout, encoding="utf-8")
        (spec.log_dir / f"{spec.name}.err").write_text(proc.stderr, encoding="utf-8")
        return proc

    def submit(self, spec: JobSpec) -> JobRef:
        """后台执行。"""
        if spec.gpus:
            msg = f"{spec.name}: 本机后端不分配 GPU，GPU 步骤用 taskmgr 后端"
            raise BackendError(msg)
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        rc = spec.log_dir / f"{spec.name}.rc"
        rc.unlink(missing_ok=True)
        with (
            (spec.log_dir / f"{spec.name}.out").open("w") as out,
            (spec.log_dir / f"{spec.name}.err").open("w") as err,
        ):
            p = subprocess.Popen(
                ["bash", "-c", _WRAPPER, "dm-rsi-job", str(rc), *spec.argv],
                cwd=spec.cwd,
                env=self._env(spec),
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        return JobRef(
            backend=self.name, id=str(p.pid), meta={"rc": str(rc), "log_dir": str(spec.log_dir)}
        )

    def status(self, job: JobRef) -> JobState:
        """Rc 文件存在 → 结束；进程还在 → running；都不是 → unknown（例如机器重启）。"""
        rc = job.meta.get("rc")
        if isinstance(rc, str) and os.path.exists(rc):  # noqa: PTH110 -- 和 rc 字符串一起用更直观
            code = open(rc, encoding="utf-8").read().strip()  # noqa: PTH123, SIM115
            return JobState.DONE if code == "0" else JobState.FAILED
        try:
            os.kill(int(job.id), 0)
        except ProcessLookupError:
            return JobState.UNKNOWN
        except PermissionError:
            return JobState.RUNNING
        return JobState.RUNNING

    def fetch(self, job: JobRef, dest: Path) -> None:
        """日志已经在本机，复制一份到 ``dest``。"""
        src = job.meta.get("log_dir")
        if isinstance(src, str):
            dest.mkdir(parents=True, exist_ok=True)
            for f in Path(src).iterdir():
                shutil.copy2(f, dest / f.name)

    def cancel(self, job: JobRef) -> None:
        """杀掉整个进程组。"""
        try:
            os.killpg(int(job.id), signal.SIGTERM)
        except ProcessLookupError:
            return
