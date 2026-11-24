from __future__ import annotations

import pytest

from dm_rsi import cli, issues
from dm_rsi.approvals import AGENT_ENV, ApprovalError, ApprovalStore, Kind, State
from dm_rsi.config import ConfigError


def test_request_dedupes_by_fingerprint(cfg):
    store = ApprovalStore(cfg.approvals_dir)
    a = store.request(Kind.TRAIN, "train x", "fp1", {"gpus": 2})
    b = store.request(Kind.TRAIN, "train x again", "fp1", {"gpus": 2})
    assert a.id == b.id == "a0001"
    c = store.request(Kind.TRAIN, "train y", "fp2", {})
    assert c.id == "a0002"


def test_rejected_is_not_reopened_with_same_fingerprint(cfg):
    store = ApprovalStore(cfg.approvals_dir)
    a = store.request(Kind.TRAIN, "t", "fp", {})
    store.decide(a.id, approve=False, reason="不值得")
    again = store.request(Kind.TRAIN, "t", "fp", {})
    assert again.id == a.id
    assert again.state is State.REJECTED


def test_done_allows_new_request(cfg):
    store = ApprovalStore(cfg.approvals_dir)
    a = store.request(Kind.TRAIN, "t", "fp", {})
    store.decide(a.id, approve=True)
    store.mark(a.id, ok=True, result={})
    assert store.request(Kind.TRAIN, "t", "fp", {}).id == "a0002"


def test_reject_needs_reason(cfg):
    store = ApprovalStore(cfg.approvals_dir)
    a = store.request(Kind.TRAIN, "t", "fp", {})
    with pytest.raises(ApprovalError, match="原因"):
        store.decide(a.id, approve=False)


def test_agent_cannot_decide(cfg, monkeypatch):
    store = ApprovalStore(cfg.approvals_dir)
    a = store.request(Kind.FINAL_TEST, "t", "fp", {})
    monkeypatch.setenv(AGENT_ENV, "1")
    with pytest.raises(ApprovalError, match="人"):
        store.decide(a.id, approve=True)


def test_mark_requires_approval(cfg):
    store = ApprovalStore(cfg.approvals_dir)
    a = store.request(Kind.TRAIN, "t", "fp", {})
    with pytest.raises(ApprovalError, match="未经批准"):
        store.mark(a.id, ok=True, result={})


def test_draft_goes_to_queue_with_prefix_and_marker(cfg, gh):
    store = ApprovalStore(cfg.approvals_dir)
    res = issues.draft(
        cfg, gh, store, direction="data", key="k1", title="池分数缺 step 列", body="正文"
    )
    assert res.approval is not None
    assert res.approval.kind is Kind.ISSUE
    d = res.approval.detail
    assert d["repo"] == "Jireh-Chen/DM-RSI-Framework"
    assert str(d["issue_title"]).startswith("[data] ")
    assert issues.marker(issues.fingerprint("data", "k1")) in str(d["issue_body"])
    assert "@BingqiaoL" in str(d["issue_body"])
    assert gh.created == []  # 没批准不发
    assert "公开仓库" in res.approval.explain
    assert "池分数缺 step 列" in res.approval.explain


def test_draft_skips_when_open_issue_has_same_fingerprint(cfg, gh):
    fp = issues.fingerprint("decision-policy", "k")
    existing = gh.add(cfg.upstream_repo, "[decision-policy] x", f"body\n{issues.marker(fp)}")
    res = issues.draft(
        cfg,
        gh,
        ApprovalStore(cfg.approvals_dir),
        direction="decision-policy",
        key="k",
        title="x",
        body="b",
    )
    assert res.existing == existing
    assert res.approval is None


def test_closed_issue_does_not_block_new_draft(cfg, gh):
    fp = issues.fingerprint("data", "k")
    gh.add(cfg.upstream_repo, "[data] x", f"b\n{issues.marker(fp)}", state="CLOSED")
    res = issues.draft(
        cfg, gh, ApprovalStore(cfg.approvals_dir), direction="data", key="k", title="x", body="b"
    )
    assert res.approval is not None


def test_unknown_direction(cfg, gh):
    with pytest.raises(ConfigError, match="未知方向"):
        issues.draft(
            cfg,
            gh,
            ApprovalStore(cfg.approvals_dir),
            direction="nope",
            key="k",
            title="x",
            body="b",
        )


def test_cli_approve_sends_issue(cfg, gh, capsys):
    store = ApprovalStore(cfg.approvals_dir)
    res = issues.draft(cfg, gh, store, direction="training", key="k", title="t", body="b")
    assert res.approval is not None
    rc = cli.main(["--home", str(cfg.home), "approve", res.approval.id], gh=gh)
    assert rc == 0
    assert len(gh.created) == 1
    repo, title, body = gh.created[0]
    assert repo == "CD22104/decision-model-training"  # Training 有自己的仓库
    assert title == "[training] t"
    assert "@CD22104" in body
    assert store.get(res.approval.id).state is State.DONE
    assert "url" in (store.get(res.approval.id).result or {})


def test_send_failure_is_recorded(cfg):
    class Broken:
        def list_issues(self, *_a, **_k):
            return []

        def create_issue(self, *_a):
            msg = "boom"
            raise RuntimeError(msg)

    store = ApprovalStore(cfg.approvals_dir)
    res = issues.draft(cfg, Broken(), store, direction="data", key="k", title="t", body="b")
    assert res.approval is not None
    store.decide(res.approval.id, approve=True)
    a = issues.send(Broken(), store, res.approval.id)
    assert a.state is State.FAILED


def test_inbox_and_outbound(cfg, gh):
    gh.add(cfg.self_repo, "dp 需要 pool 分数", "请提供", author="yile")
    gh.add(cfg.upstream_repo, "[data] y", f"b\n{issues.marker('0' * 12)}")
    gh.add(cfg.upstream_repo, "别人的 issue", "无标记")
    assert [i.title for i in issues.inbox(cfg, gh)] == ["dp 需要 pool 分数"]
    assert [i.title for i in issues.outbound(cfg, gh)] == ["[data] y"]


def test_issue_repo_per_direction(cfg, gh):
    assert cfg.issue_repo("data") == cfg.upstream_repo  # fork 关了 issues，发到 Yile 的上游
    assert cfg.issue_repo("decision-policy") == cfg.upstream_repo
    assert cfg.issue_repo("training") == "CD22104/decision-model-training"
    fp = issues.fingerprint("training", "k")
    gh.add("CD22104/decision-model-training", "[training] x", f"b\n{issues.marker(fp)}")
    store = ApprovalStore(cfg.approvals_dir)
    res = issues.draft(cfg, gh, store, direction="training", key="k", title="x", body="b")
    assert res.existing is not None  # 去重看的是 Training 自己的仓库
    assert [i.repo for i in issues.outbound(cfg, gh)] == ["CD22104/decision-model-training"]
