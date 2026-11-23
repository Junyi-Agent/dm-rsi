from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dm_rsi import config
from dm_rsi.github import Issue

ROOT = Path(__file__).resolve().parents[1]
REAL_HOME_ENV = Path("/ssd1/junyi/jev/dmrsi/envs/subrsi")


class FakeGitHub:
    """内存里的 GitHub：记录建了哪些 issue。"""

    def __init__(self) -> None:
        self.issues: dict[str, list[Issue]] = {}
        self.created: list[tuple[str, str, str]] = []

    def add(
        self, repo: str, title: str, body: str, *, state: str = "OPEN", author: str = "someone"
    ) -> Issue:
        n = len(self.issues.get(repo, [])) + 1
        issue = Issue(
            repo,
            n,
            title,
            state,
            f"https://github.com/{repo}/issues/{n}",
            body,
            author,
            "2026-11-23",
            0,
        )
        self.issues.setdefault(repo, []).append(issue)
        return issue

    def list_issues(
        self, repo: str, *, state: str = "all", search: str | None = None
    ) -> list[Issue]:
        out = self.issues.get(repo, [])
        if state != "all":
            out = [i for i in out if i.state.lower() == state]
        if search and "dm-rsi:fp" in search:
            out = [i for i in out if "dm-rsi:fp" in i.body]
        return list(out)

    def create_issue(self, repo: str, title: str, body: str) -> str:
        self.created.append((repo, title, body))
        return self.add(repo, title, body).url


@pytest.fixture
def gh() -> FakeGitHub:
    return FakeGitHub()


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> config.Config:
    monkeypatch.delenv("DMRSI_AGENT", raising=False)
    return config.load(ROOT, home=tmp_path / "home")


@pytest.fixture
def cfg_with_env(cfg: config.Config) -> config.Config:
    """复用本机已建好的 sub-RSI venv（建一次要几分钟）；没有时跳过。"""
    if not (REAL_HOME_ENV / "bin" / "python").is_file():
        pytest.skip("本机没有 sub-RSI venv（先跑一次 dm-rsi run advance 建好）")
    (cfg.home / "envs").mkdir(parents=True, exist_ok=True)
    (cfg.home / "envs" / "subrsi").symlink_to(REAL_HOME_ENV)
    return cfg


@pytest.fixture
def which_task() -> str | None:
    return shutil.which("task")
