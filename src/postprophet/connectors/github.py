"""GitHub connector — emits 'ship' seeds from a repo's PRs, issues, and releases.

For builders shipping on GitHub (the norm). Reads recent activity via the GitHub
REST API using a token from the environment (GH_TOKEN / GITHUB_TOKEN) or passed
as an option. Zero LLM.

Options:
    repo        owner/repo to watch (required)
    since_days  look back this many days (default 7)
    kinds       which activity to emit: pr | release | issue (default all three)
    token       optional PAT; falls back to env GH_TOKEN / GITHUB_TOKEN

Emitted seed kinds:
    merged PR   -> kind "ship", angle-friendly (announcement / build-in-public)
    release     -> kind "milestone"
    opened PR   -> kind "signal" (work in progress worth noting)
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone

from ..context import Seed

DEFAULT_KINDS = ["pr", "release", "issue"]
GITHUB_API = "https://api.github.com"


class GitHubConnector:
    def __init__(self, repo, since_days: int = 7, kinds=None, token=None,
                 limit_per_kind: int = 5):
        self.repo = repo.lstrip("/")
        self.since_days = since_days
        self.kinds = kinds or DEFAULT_KINDS
        self.token = token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        self.limit_per_kind = limit_per_kind

    def _since_iso(self):
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.since_days)
        return cutoff.isoformat()

    def _get(self, path):
        req = urllib.request.Request(f"{GITHUB_API}{path}")
        if self.token:
            req.add_header("Authorization", f"token {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception:
            return []

    def _prs(self):
        since = self._since_iso()
        # merged PRs are the strongest "ship" signal
        merged = self._get(
            f"/repos/{self.repo}/pulls?state=closed&sort=updated&direction=desc"
            f"&per_page={self.limit_per_kind}"
        )
        out = []
        for pr in merged or []:
            merged_at = pr.get("merged_at")
            if not merged_at:
                continue
            if merged_at < since:
                continue
            out.append({
                "kind": "ship",
                "title": f"[{self.repo}] merged: {pr['title']} (#{pr['number']})",
                "detail": f"PR #{pr['number']} merged into {self.repo} — {pr.get('body','')[:200]}",
                "date": merged_at[:10],
                "tags": ["ship", self.repo],
            })
        return out

    def _releases(self):
        since = self._since_iso()
        rels = self._get(f"/repos/{self.repo}/releases?per_page={self.limit_per_kind}")
        out = []
        for r in rels or []:
            pub = r.get("published_at")
            if not pub or pub < since:
                continue
            out.append({
                "kind": "milestone",
                "title": f"[{self.repo}] release {r.get('tag_name')}",
                "detail": f"Release {r.get('tag_name')} of {self.repo} — {r.get('name') or ''}",
                "date": pub[:10],
                "tags": ["milestone", self.repo],
            })
        return out

    def collect(self):
        seeds = []
        if "pr" in self.kinds:
            seeds.extend(self._prs())
        if "release" in self.kinds:
            seeds.extend(self._releases())
        for item in seeds:
            item.setdefault("source", f"github:{self.repo}")
        return seeds
