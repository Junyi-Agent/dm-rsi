"""sub-RSI 的运行环境和命令。

- 一个共享 venv：``$DMRSI_HOME/envs/subrsi``，依赖取各 sub-RSI ``pyproject.toml`` 的 ``dependencies`` 并集。
  不 ``pip install -e`` sub-RSI 本身（会往 submodule 里写 egg-info），而是用 ``PYTHONPATH`` 指向包目录。
- 命令一律在 submodule 之外的目录里运行，``PYTHONDONTWRITEBYTECODE=1``，submodule 保持干净。
- 共享机器上 sklearn / numpy 默认会占满所有核，限制线程数（流程 README 的约定）。
"""

from __future__ import annotations

import hashlib
import subprocess
import tomllib
from typing import TYPE_CHECKING

from dm_rsi.backends.base import JobSpec
from dm_rsi.backends.local import LocalBackend

if TYPE_CHECKING:
    from pathlib import Path

    from dm_rsi.config import Config, Direction

THREADS = "8"
PYTHON_VERSION = "3.12"


class SubRsiError(RuntimeError):
    """sub-RSI 命令失败。"""


def _deps(cfg: Config) -> list[str]:
    deps: set[str] = set()
    for d in cfg.directions.values():
        pkg = d.package_dir(cfg.root)
        if pkg is None or not (pkg / "pyproject.toml").is_file():
            continue
        data = tomllib.loads((pkg / "pyproject.toml").read_text(encoding="utf-8"))
        deps.update(data.get("project", {}).get("dependencies", []))
    return sorted(deps)


def env_dir(cfg: Config) -> Path:
    """共享 venv 的位置。"""
    return cfg.home / "envs" / "subrsi"


def python(cfg: Config) -> Path:
    """共享 venv 的解释器。"""
    return env_dir(cfg) / "bin" / "python"


def ensure_env(cfg: Config) -> Path:
    """依赖变了（或 venv 不存在）时重建，返回解释器路径。"""
    deps = _deps(cfg)
    digest = hashlib.sha256("\n".join(deps).encode()).hexdigest()[:16]
    stamp = env_dir(cfg) / ".dmrsi-deps"
    if python(cfg).is_file() and stamp.is_file() and stamp.read_text().strip() == digest:
        return python(cfg)
    env_dir(cfg).parent.mkdir(parents=True, exist_ok=True)
    if not python(cfg).is_file():
        subprocess.run(
            ["uv", "venv", "-q", str(env_dir(cfg)), "--python", PYTHON_VERSION], check=True
        )
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(python(cfg)), *deps], check=True)
    stamp.write_text(digest + "\n")
    return python(cfg)


def base_env(pythonpath: Path) -> dict[str, str]:
    """sub-RSI 进程的环境变量。"""
    return {
        "PYTHONPATH": str(pythonpath),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": THREADS,
        "OPENBLAS_NUM_THREADS": THREADS,
        "MKL_NUM_THREADS": THREADS,
    }


def run_module(  # noqa: PLR0913 -- 都是命令的组成部分，keyword-only
    cfg: Config,
    direction: Direction,
    *,
    module: str,
    args: list[str],
    cwd: Path,
    log_dir: Path,
    name: str,
    ok_codes: tuple[int, ...] = (0,),
) -> str:
    """``python -m <module> <args>``，返回 stdout；退出码不在 ``ok_codes`` 时报错（带 stderr 结尾）。"""
    pkg = direction.package_dir(cfg.root)
    if pkg is None:
        msg = f"方向 {direction.name} 没有 sub-RSI 包"
        raise SubRsiError(msg)
    spec = JobSpec(
        name=name,
        argv=[str(python(cfg)), "-m", module, *args],
        cwd=cwd,
        log_dir=log_dir,
        env=base_env(pkg),
    )
    cwd.mkdir(parents=True, exist_ok=True)
    proc = LocalBackend().run(spec)
    if proc.returncode not in ok_codes:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        msg = f"{module} {' '.join(args[:1])} 失败（{proc.returncode}），日志 {log_dir}/{name}.err：\n{tail}"
        raise SubRsiError(msg)
    return proc.stdout
