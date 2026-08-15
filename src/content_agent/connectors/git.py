"""Git connector — emits 'ship' seeds from recent commits in one or more repos.

The foundational connector: every builder ships code. Reads recent commit
subjects (and optional body snippets) as discrete ship events. Pure git + shell,
zero LLM. Deterministic.

Options:
    repos        list of repo paths (or single path string)
    since_days   look back this many days for recent commits (default 14)
    limit_per_repo  max commits to emit per repo (default 5)
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone

from ..context import Seed


class GitConnector:
    def __init__(self, repos, since_days: int = 14, limit_per_repo: int = 5):
        if isinstance(repos, str):
            repos = [repos]
        self.repos = repos
        self.since_days = since_days
        self.limit_per_repo = limit_per_repo

    def _git(self, repo, args):
        try:
            out = subprocess.run(
                ["git", "-C", repo] + args, capture_output=True, text=True, timeout=30
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except (subprocess.TimeoutExpired, subprocess.FileNotFoundError):
            return ""

    def _recent_commits(self, repo):
        log = self._git(repo, ["log", f"-n{self.limit_per_repo}", "--format=%ad|%s|%h", "--date=short"])
        commits = []
        for line in log.splitlines():
            if not line:
                continue
            date, subject, short = line.split("|", 2)
            commits.append({"date": date, "subject": subject, "short": short})
        return commits

    def collect(self):
        seeds = []
        for repo in self.repos:
            repo = os.path.expanduser(repo)
            if not os.path.isdir(os.path.join(repo, ".git")):
                continue
            repo_name = os.path.basename(os.path.normpath(repo))
            for c in self._recent_commits(repo):
                seeds.append(Seed(
                    source=f"git:{repo_name}",
                    kind="ship",
                    title=f"[{repo_name}] {c['subject']}",
                    detail=f"commit {c['short']} on {c['date']} in {repo_name}",
                    date=c["date"],
                    tags=["ship", repo_name],
                ).to_dict())
        return seeds
