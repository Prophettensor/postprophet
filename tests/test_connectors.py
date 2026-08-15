"""Connector tests: git, rss, agent_output, webhook, github (mocked network)."""

import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from postprophet.connectors.git import GitConnector
from postprophet.connectors.rss import RssConnector
from postprophet.connectors.agent_output import AgentOutputConnector
from postprophet.connectors.webhook import WebhookConnector
from postprophet.connectors.github import GitHubConnector
from postprophet.context import Seed


# ---------- git ----------

def test_git_connector_emits_ship_seeds(tmp_path):
    # build a tiny git repo with one commit
    os.makedirs(tmp_path / ".git")
    from subprocess import run
    for cmd in [
        ["git", "init", "-q"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--allow-empty",
         "-m", "ship the framework"],
    ]:
        run(cmd, cwd=str(tmp_path), capture_output=True)
    c = GitConnector([str(tmp_path)])
    seeds = c.collect()
    assert len(seeds) >= 1
    assert seeds[0]["kind"] == "ship"
    assert "ship the framework" in seeds[0]["title"]


# ---------- agent_output ----------

def test_agent_output_reads_json_and_text(tmp_path):
    os.makedirs(tmp_path / "out", exist_ok=True)
    with open(tmp_path / "out" / "finding.json", "w") as f:
        json.dump({"title": "research found X", "detail": "a grounded finding",
                   "tags": ["crypto"]}, f)
    with open(tmp_path / "out" / "note.md", "w") as f:
        f.write("A plain text signal\nmore context here")
    c = AgentOutputConnector(str(tmp_path / "out"), kind="finding")
    seeds = c.collect()
    kinds = [s["kind"] for s in seeds]
    assert kinds.count("finding") == 2
    titles = " ".join(s["title"] for s in seeds)
    assert "research found X" in titles
    assert "A plain text signal" in titles


# ---------- rss ----------

def test_rss_parses_atom(monkeypatch):
    atom = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Market</title>
  <entry>
    <title>Big competitor ships v2</title>
    <link href="https://example.com/1"/>
    <updated>2026-08-15T00:00:00Z</updated>
    <summary>A major release</summary>
  </entry>
</feed>"""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30: _FakeResp(atom.encode()),
    )
    c = RssConnector(["https://example.com/feed"], since_hours=48)
    seeds = c.collect()
    assert len(seeds) == 1
    assert seeds[0]["kind"] == "signal"
    assert "ships v2" in seeds[0]["title"]


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def read(self, n=-1):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------- webhook ----------

def test_webhook_receives_seed(tmp_path):
    w = WebhookConnector(port=0)  # port 0 -> OS-assigned
    # rebind to a real free port
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    w.port = port
    w.start()
    try:
        import urllib.request
        body = json.dumps({"source": "hook", "kind": "event",
                           "title": "injected live"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{w.path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=10).read()
        seeds = w.collect()
        assert len(seeds) == 1
        assert seeds[0]["title"] == "injected live"
    finally:
        w.stop()


# ---------- github (mocked) ----------

def test_github_emits_merged_pr_and_release(monkeypatch):
    fake = {
        "/pulls?": [{
            "title": "Add auth", "number": 42, "merged_at": "2026-08-14T10:00:00Z",
            "body": "closes the auth gap",
        }],
        "/releases?": [{
            "tag_name": "v1.2.0", "name": "v1.2.0",
            "published_at": "2026-08-13T10:00:00Z",
        }],
    }

    def _fake_get(self, path):
        if "pulls" in path:
            return fake["/pulls?"]
        if "releases" in path:
            return fake["/releases?"]
        return []

    monkeypatch.setattr(GitHubConnector, "_get", _fake_get)
    c = GitHubConnector("some/repo", since_days=7)
    seeds = c.collect()
    kinds = [s["kind"] for s in seeds]
    assert "ship" in kinds and "milestone" in kinds
    assert any("Add auth" in s["title"] for s in seeds)
    assert any("v1.2.0" in s["title"] for s in seeds)


def test_github_skips_old_merged_pr(monkeypatch):
    def _fake_get(self, path):
        return [{"title": "old", "number": 1,
                 "merged_at": "2026-01-01T00:00:00Z", "body": ""}]

    monkeypatch.setattr(GitHubConnector, "_get", _fake_get)
    c = GitHubConnector("some/repo", since_days=7)
    seeds = c.collect()
    assert seeds == []


# ---------- pipeline integration with new connectors ----------

def test_pipeline_harvest_with_rss_and_agent(tmp_path, monkeypatch):
    from postprophet.pipeline import Pipeline
    from postprophet.config import InstanceConfig, LoopConfig, ConnectorConfig, VisionConfig

    # agent_output file
    os.makedirs(tmp_path / "out", exist_ok=True)
    with open(tmp_path / "out" / "f.json", "w") as f:
        json.dump({"title": "grounded finding", "detail": "d", "tags": []}, f)

    atom = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>M</title>
<entry><title>market shift</title><link href="https://e.com"/>
<updated>2026-08-15T00:00:00Z</updated><summary>s</summary></entry></feed>"""
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=30: _FakeResp(atom.encode()))

    cfg = InstanceConfig(
        name="t", loop=LoopConfig(),
        vision=VisionConfig(positioning="postprophet content agents",
                            problem="marketing for builders"),
        connectors=[
            ConnectorConfig(name="a", type="agent_output",
                            options={"path": str(tmp_path / "out")}),
            ConnectorConfig(name="r", type="rss",
                            options={"feeds": ["https://example.com/feed"]}),
        ],
    )
    pipe = Pipeline(cfg, str(tmp_path / "data"))
    seeds = pipe.harvest()
    titles = [s.title for s in seeds]
    assert any("grounded finding" in t for t in titles)
    assert any("market shift" in t for t in titles)
    ideas = pipe.ideas(seeds)
    assert len(ideas) >= 1
