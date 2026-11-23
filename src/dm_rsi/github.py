"""GitHub 访问。只有两个动作：列 issue、建 issue。生产用 ``gh`` CLI，测试换成假实现。"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Protocol, cast


class GitHubError(RuntimeError):
    """gh 调用失败。"""


@dataclass(frozen=True)
class Issue:
    """issue 的摘要。"""

    repo: str
    number: int
    title: str
    state: str
    """OPEN / CLOSED（gh 的写法）。"""
    url: str
    body: str
    author: str
    updated_at: str
    comments: int


class GitHub(Protocol):
    """dm-rsi 用到的 GitHub 操作。"""

    def list_issues(
        self, repo: str, *, state: str = "all", search: str | None = None
    ) -> list[Issue]:
        """列 issue。``state``：open / closed / all。"""
        ...

    def create_issue(self, repo: str, title: str, body: str) -> str:
        """建 issue，返回 URL。"""
        ...


_FIELDS = "number,title,state,url,body,author,updatedAt,comments"


class GhCli:
    """用本机已登录的 ``gh``。"""

    def __init__(self, limit: int = 100) -> None:
        """``limit``：每次最多列多少个 issue。"""
        self.limit: int = limit

    def _run(self, argv: list[str]) -> str:
        proc = subprocess.run(
            ["gh", *argv], capture_output=True, text=True, check=False, timeout=60
        )
        if proc.returncode != 0:
            msg = f"gh {' '.join(argv[:3])} 失败（{proc.returncode}）：{proc.stderr.strip()}"
            raise GitHubError(msg)
        return proc.stdout

    def list_issues(
        self, repo: str, *, state: str = "all", search: str | None = None
    ) -> list[Issue]:
        """列 issue。"""
        argv = [
            "issue",
            "list",
            "-R",
            repo,
            "--state",
            state,
            "-L",
            str(self.limit),
            "--json",
            _FIELDS,
        ]
        if search:
            argv += ["--search", search]
        rows = cast("list[dict[str, object]]", json.loads(self._run(argv) or "[]"))
        out: list[Issue] = []
        for r in rows:
            author = cast("dict[str, object] | None", r.get("author")) or {}
            comments = cast("list[object] | None", r.get("comments")) or []
            out.append(
                Issue(
                    repo=repo,
                    number=int(cast("int", r["number"])),
                    title=str(r["title"]),
                    state=str(r["state"]),
                    url=str(r["url"]),
                    body=str(r.get("body") or ""),
                    author=str(author.get("login") or ""),
                    updated_at=str(r.get("updatedAt") or ""),
                    comments=len(comments),
                )
            )
        return out

    def create_issue(self, repo: str, title: str, body: str) -> str:
        """建 issue。"""
        return self._run(["issue", "create", "-R", repo, "--title", title, "--body", body]).strip()
