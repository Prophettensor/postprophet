"""
Tracker — reads real engagement for posted drafts and records outcomes for the
learner. Designed to run as a no_agent cron (zero LLM tokens).

Reads the account's recent timeline via xurl (X API), maps each post back to the
strategy it was generated with (data/posts.jsonl), records reward metrics, and
triggers the learner update. Silent when there's nothing new (watchdog pattern)
so a no_agent cron can run it every tick without spamming.

Auth is read from the environment/credentials store (xurl), never from config.
If not authed, exits silently — nothing to report yet.
"""

from __future__ import annotations

import json
import os
import subprocess

from .reward import reward_per_impression

METRIC_FIELDS = [
    "reply_count", "retweet_count", "like_count",
    "quote_count", "bookmark_count", "impression_count",
]


class Tracker:
    def __init__(self, data_dir: str, home: str = "/opt/data/home",
                 xurl: str = "xurl"):
        self.data_dir = data_dir
        self.home = home
        self.xurl = xurl
        self.posts_path = os.path.join(data_dir, "posts.jsonl")
        self.outcomes_path = os.path.join(data_dir, "outcomes.jsonl")
        self.resolved_path = os.path.join(data_dir, "resolved_ids.txt")

    def _run_xurl(self, args, timeout=60):
        env = dict(os.environ)
        env["HOME"] = self.home
        try:
            out = subprocess.run(
                [self.xurl] + args, capture_output=True, text=True,
                timeout=timeout, env=env,
            )
            if out.returncode != 0:
                return None
            return json.loads(out.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            return None

    def _auth_ok(self):
        return bool(self._run_xurl(["auth", "status"]))

    def _read_posts(self):
        posts = {}
        if os.path.exists(self.posts_path):
            with open(self.posts_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("tweet_id") and rec.get("posted"):
                            posts[rec["tweet_id"]] = rec
                    except json.JSONDecodeError:
                        continue
        return posts

    def _read_resolved(self):
        if os.path.exists(self.resolved_path):
            with open(self.resolved_path) as f:
                return set(f.read().split())
        return set()

    def _append_resolved(self, tid):
        with open(self.resolved_path, "a") as f:
            f.write(tid + "\n")

    @staticmethod
    def _metrics_of(tweet):
        pm = tweet.get("public_metrics") or {}
        return {k: pm.get(k, 0) for k in METRIC_FIELDS}

    def resolve(self) -> int:
        """Fetch the account timeline, match to our posted drafts, record any not
        yet resolved. Returns number newly recorded (0 = nothing new)."""
        if not self._auth_ok():
            return 0
        posts = self._read_posts()
        if not posts:
            return 0
        resolved = self._read_resolved()
        timeline = self._run_xurl(["timeline", "-n", "100"])
        if not timeline:
            return 0

        tweets = timeline.get("data", timeline.get("tweets", []))
        if isinstance(tweets, dict):
            tweets = [tweets]

        new = 0
        for tweet in tweets:
            tid = tweet.get("id")
            if not tid or tid in resolved or tid not in posts:
                continue
            metrics = self._metrics_of(tweet)
            record = {
                "tweet_id": tid,
                "strategy": posts[tid].get("strategy", "unknown"),
                "network_lever": posts[tid].get("network_lever", "out_of_network"),
                "text": posts[tid].get("text", ""),
                "posted_at": posts[tid].get("posted_at"),
                "metrics": metrics,
                "reward": reward_per_impression(metrics),
            }
            with open(self.outcomes_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._append_resolved(tid)
            new += 1
        return new
