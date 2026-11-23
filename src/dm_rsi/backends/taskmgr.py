"""GPU 后端：用 gpu-taskmgr 的 ``task`` CLI（https://github.com/Junyi-Agent/gpu-taskmgr）。

每个作业打成一个任务包：``run.sh`` 进入 ``cwd`` 执行命令，产物直接写服务器上的绝对路径
（``$DMRSI_HOME`` 在服务器的大盘上），任务包本身只带脚本。GPU 用条件选择
``{"count": n, "min_free_gb": m}``，由服务器在启动时挑空闲卡；``n`` 不超过 ``max_gpus``
（共享服务器每人 ≤ 2 卡）。
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from typing import TYPE_CHECKING, cast

from dm_rsi.backends.base import BackendError, JobRef, JobSpec, JobState

if TYPE_CHECKING:
    from pathlib import Path

_STATES = {
    "queued": JobState.QUEUED,
    "precompiling": JobState.RUNNING,
    "starting": JobState.RUNNING,
    "running": JobState.RUNNING,
    "stopping": JobState.RUNNING,
    "succeeded": JobState.DONE,
    "failed": JobState.FAILED,
    "invalid_result": JobState.FAILED,
    "cancelled": JobState.FAILED,
    "timed_out": JobState.FAILED,
    "unknown": JobState.UNKNOWN,
}


class TaskmgrBackend:
    """``task pack`` + ``task submit`` / ``status`` / ``fetch`` / ``cancel``。"""

    def __init__(  # noqa: PLR0913 -- 配置项，keyword-only
        self,
        name: str,
        *,
        host: str,
        max_gpus: int,
        min_free_gb: int,
        state_dir: Path,
        task_cli: str = "task",
    ) -> None:
        """``state_dir`` 放任务包和 task_id 计数器。"""
        self.name: str = name
        self.host: str = host
        self.max_gpus: int = max_gpus
        self.min_free_gb: int = min_free_gb
        self.state_dir: Path = state_dir
        self.task_cli: str = task_cli

    def _task(self, *argv: str) -> str:
        proc = subprocess.run(
            [self.task_cli, *argv, "--plain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if proc.returncode != 0:
            msg = f"task {' '.join(argv[:2])} 失败（{proc.returncode}）：{proc.stderr.strip() or proc.stdout.strip()}"
            raise BackendError(msg)
        return proc.stdout

    def _next_task_id(self) -> str:
        counter = self.state_dir / "taskmgr-counter"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        n = int(counter.read_text()) + 1 if counter.is_file() else 1
        counter.write_text(str(n))
        return f"DM-{n % 10000:04d}"

    def package(self, spec: JobSpec) -> Path:
        """写任务包源目录并用 ``task pack`` 校验打包，返回包目录。"""
        if spec.gpus > self.max_gpus:
            msg = f"{spec.name} 要 {spec.gpus} 张卡，超过后端 {self.name} 的上限 {self.max_gpus}"
            raise BackendError(msg)
        src = self.state_dir / "src" / spec.name
        src.mkdir(parents=True, exist_ok=True)
        (src / "run.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"cd {shlex.quote(str(spec.cwd))}\n"
            f"exec {shlex.join(spec.argv)}\n",
            encoding="utf-8",
        )
        (src / "run.sh").chmod(0o755)
        (src / "TASK.md").write_text(
            f"# {spec.name}\n\ndm-rsi 提交的作业。命令：\n\n```\n{shlex.join(spec.argv)}\n```\n\n工作目录：`{spec.cwd}`\n",
            encoding="utf-8",
        )
        gpus = (
            json.dumps({"count": spec.gpus, "min_free_gb": self.min_free_gb})
            if spec.gpus
            else "none"
        )
        out = self.state_dir / "pkg" / spec.name
        argv = [
            "pack",
            f"{src}/",
            "--task-id",
            self._next_task_id(),
            "--entry",
            "bash run.sh",
            "--gpus",
            gpus,
        ]
        argv += ["-o", str(out), "--force"]
        if spec.timeout:
            argv += ["--timeout", str(spec.timeout)]
        for k, v in spec.env.items():
            argv += ["--env", f"{k}={v}"]
        self._task(*argv)
        return out

    def submit(self, spec: JobSpec) -> JobRef:
        """打包并提交。submission key 由命令内容决定，重复提交同一个作业会被服务器去重。"""
        pkg = self.package(spec)
        key = (
            "dmrsi-"
            + hashlib.sha256(
                json.dumps([spec.name, spec.argv, str(spec.cwd)]).encode()
            ).hexdigest()[:16]
        )
        receipt = cast(
            "dict[str, object]",
            json.loads(
                self._task(
                    "submit", "--host", self.host, "--submission-key", key, str(pkg), "--detail"
                )
            ),
        )
        if "error" in receipt:
            msg = f"task submit 失败：{receipt['error']}"
            raise BackendError(msg)
        return JobRef(
            backend=self.name,
            id=str(receipt["jobid"]),
            meta={"host": self.host, "task_id": str(receipt.get("task_id")), "submission_key": key},
        )

    def status(self, job: JobRef) -> JobState:
        """``task status --detail`` 的 state。"""
        obj = cast(
            "dict[str, object]",
            json.loads(self._task("status", job.id, "--host", self.host, "--detail")),
        )
        return _STATES.get(str(obj.get("state")), JobState.UNKNOWN)

    def fetch(self, job: JobRef, dest: Path) -> None:
        """下载作业的日志和 result.json。"""
        dest.mkdir(parents=True, exist_ok=True)
        self._task("fetch", job.id, "--host", self.host, "--output", str(dest))

    def cancel(self, job: JobRef) -> None:
        """取消。"""
        self._task("cancel", job.id, "--host", self.host)
