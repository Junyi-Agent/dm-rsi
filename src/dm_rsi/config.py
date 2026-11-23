"""项目配置 ``dm-rsi.yaml``：各方向、GitHub 仓库、执行后端、``$DMRSI_HOME``。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dm_rsi import yamlfile

if TYPE_CHECKING:
    from xtra_rsi.payload import Payload, Value

CONFIG_NAME = "dm-rsi.yaml"


class ConfigError(ValueError):
    """配置缺字段或不合法。"""


@dataclass(frozen=True)
class Direction:
    """一个方向（Data、Training、Decision Policy）。没有 submodule 表示还没有 sub-RSI。"""

    name: str
    owner: str
    issue_prefix: str
    github_login: str | None = None
    submodule: str | None = None
    branch: str | None = None
    package: str | None = None
    module: str | None = None

    def package_dir(self, root: Path) -> Path | None:
        """sub-RSI 包在本仓库里的路径（只读）。"""
        if self.submodule is None or self.package is None:
            return None
        return root / self.submodule / self.package


@dataclass(frozen=True)
class Config:
    """``dm-rsi.yaml`` 加上运行时决定的路径。"""

    root: Path
    home: Path
    self_repo: str
    upstream_repo: str
    directions: dict[str, Direction]
    backends: dict[str, Payload] = field(default_factory=dict[str, "Payload"])

    def direction(self, name: str) -> Direction:
        """按名字取方向，不存在时报错并列出可选值。"""
        try:
            return self.directions[name]
        except KeyError:
            msg = f"未知方向 {name!r}，可选：{', '.join(self.directions)}"
            raise ConfigError(msg) from None

    @property
    def approvals_dir(self) -> Path:
        """审批队列。"""
        return self.home / "approvals"

    @property
    def runs_dir(self) -> Path:
        """流程 run。"""
        return self.home / "runs"


def find_root(start: Path | None = None) -> Path:
    """从 ``start`` 往上找含 ``dm-rsi.yaml`` 的目录。"""
    p = (start or Path.cwd()).resolve()
    for d in (p, *p.parents):
        if (d / CONFIG_NAME).is_file():
            return d
    msg = f"在 {p} 及其上级目录里找不到 {CONFIG_NAME}"
    raise ConfigError(msg)


def _str(obj: Payload, key: str, where: str) -> str:
    v = obj.get(key)
    if not isinstance(v, str) or not v:
        msg = f"{where}.{key} 必须是非空字符串"
        raise ConfigError(msg)
    return v


def _opt(obj: Payload, key: str) -> str | None:
    v = obj.get(key)
    return v if isinstance(v, str) and v else None


def _map(v: Value, where: str) -> Payload:
    if not isinstance(v, dict):
        msg = f"{where} 必须是映射"
        raise ConfigError(msg)
    return v


def load(root: Path | None = None, *, home: Path | None = None) -> Config:
    """读配置。``home`` 的优先级：参数 > 环境变量 ``DMRSI_HOME`` > 配置文件。"""
    root = find_root(root)
    raw = yamlfile.read(root / CONFIG_NAME)
    gh = _map(raw.get("github"), "github")
    dirs: dict[str, Direction] = {}
    for name, v in _map(raw.get("directions"), "directions").items():
        d = _map(v, f"directions.{name}")
        where = f"directions.{name}"
        dirs[name] = Direction(
            name=name,
            owner=_str(d, "owner", where),
            issue_prefix=_str(d, "issue_prefix", where),
            github_login=_opt(d, "github_login"),
            submodule=_opt(d, "submodule"),
            branch=_opt(d, "branch"),
            package=_opt(d, "package"),
            module=_opt(d, "module"),
        )
    env_home = os.environ.get("DMRSI_HOME")
    resolved = home or (Path(env_home) if env_home else Path(_str(raw, "home", "dm-rsi.yaml")))
    backends = {
        k: _map(v, f"backends.{k}") for k, v in _map(raw.get("backends", {}), "backends").items()
    }
    return Config(
        root=root,
        home=resolved,
        self_repo=_str(gh, "self", "github"),
        upstream_repo=_str(gh, "upstream", "github"),
        directions=dirs,
        backends=backends,
    )
