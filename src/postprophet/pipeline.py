"""
Pipeline — the framework's public API. Coordinates the closed loop:

    harvest()    pull seeds from configured connectors  (zero LLM)
    ideas()      surface + rank ideas against vision     (zero LLM)
    plan()       pair strategies with ideas (bandit)     (zero LLM)
    draft()      [LLM step] turn a plan into post text   (in-session only)
    publish()    post (coach = human, auto = agent)       (needs auth)
    measure()    read real outcomes + retrain bandit      (no_agent cron)
    learn()      alias for measure()

A Pipeline is bound to ONE builder instance (one config). Use it directly or via
the CLI entrypoint. Connectors are built from the instance config on demand.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .config import load_config, InstanceConfig
from .context import normalize, Seed
from .ideas import surface_ideas, format_idea_brief
from .learner import Learner
from .generator import choose_generation_plan, build_draft_prompt
from .tracker import Tracker
from .connectors import build_connector


class Pipeline:
    def __init__(self, config: InstanceConfig, data_dir: str):
        self.config = config
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.learner = Learner(data_dir)
        self.tracker = Tracker(data_dir)

    # ---- lifecycle ----
    @classmethod
    def from_config(cls, config_path: str, data_dir: str) -> "Pipeline":
        cfg = load_config(config_path)
        return cls(cfg, data_dir)

    def _connectors(self):
        conns = []
        for cfg in self.config.connectors:
            c = build_connector(cfg)
            if c is not None:
                conns.append(c)
        return conns

    # ---- the loop ----
    def harvest(self) -> list[Seed]:
        """Pull and validate seeds from all configured connectors."""
        seeds = []
        for c in self._connectors():
            try:
                out = c.collect()
            except Exception:
                continue  # a broken connector shouldn't kill the round
            seeds.extend(normalize(out))
        # dedupe by title
        seen, uniq = set(), []
        for s in seeds:
            if s.title in seen:
                continue
            seen.add(s.title)
            uniq.append(s)
        return uniq

    def ideas(self, seeds=None, limit: int = 10) -> list[dict]:
        seeds = seeds if seeds is not None else self.harvest()
        return surface_ideas(seeds, self.config.vision, limit=limit)

    def plan(self, ideas=None, n=None) -> list[dict]:
        ideas = ideas if ideas is not None else self.ideas()
        n = n or self.config.loop.candidates_per_round
        return choose_generation_plan(
            self.learner, ideas, n=n,
            strategies_filter=self.config.loop.strategies or None,
        )

    def draft_prompts(self, plan=None) -> list[dict]:
        """Build the drafting instructions for a plan. Returns [{strategy, lever,
        idea, prompt}]. The agent drafts post text from these (in-session)."""
        plan = plan if plan is not None else self.plan()
        out = []
        for item in plan:
            prompt = build_draft_prompt(item, self.config.vision, self.config.voice)
            out.append({**item, "prompt": prompt})
        return out

    # ---- publish + measure ----
    def record_post(self, strategy, lever, text, tweet_id=None, posted=False,
                    posted_at=None):
        """Record a draft so the tracker can attribute real outcomes. Called by
        the drafting agent (and by the auto publisher) after a post is live."""
        rec = {
            "tweet_id": tweet_id,
            "strategy": strategy,
            "network_lever": lever,
            "text": text,
            "posted_at": posted_at or datetime.now(timezone.utc).isoformat(),
            "posted": posted,
        }
        with open(os.path.join(self.data_dir, "posts.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def measure(self) -> int:
        """Read real outcomes + retrain bandit. Safe as a no_agent cron. Returns
        number of newly-recorded outcomes (0 = nothing new)."""
        new = self.tracker.resolve()
        self.learner.update()
        return new

    def state_summary(self) -> str:
        return self.learner.summary()

    def round_brief(self, n=None) -> str:
        """Render a full drafting round for the agent: learning state + ranked
        ideas + plan. The agent drafts from this, in-session."""
        seeds = self.harvest()
        ideas = self.ideas(seeds)
        plan = self.plan(ideas, n)
        return "\n\n".join([
            f"BUILDER: {self.config.name}",
            "=== LEARNING STATE ===\n" + self.learner.summary(),
            "=== GROUNDED IDEAS ===\n" + format_idea_brief(ideas),
            "=== GENERATION PLAN ===\n" + self._format_plan(plan),
        ])

    def _format_plan(self, plan):
        lines = []
        for i, item in enumerate(plan, 1):
            seed = (item.get("idea") or {}).get("seed", {})
            lines.append(
                f"{i}. [{item['strategy']} / {item['lever']}] ({item['why']})"
            )
            lines.append(f"   FACT: {seed.get('title', '(no seed)')}")
            lines.append(f"   angle: {(item.get('idea') or {}).get('primary_angle')}")
        return "\n".join(lines)


def run_round(config_path: str, data_dir: str, n: int = None) -> str:
    """One-shot: build a Pipeline from config, produce the drafting brief."""
    pipe = Pipeline.from_config(config_path, data_dir)
    return pipe.round_brief(n)


def run_measure(config_path: str, data_dir: str) -> int:
    """One-shot measure: read outcomes + retrain. For the no_agent cron."""
    pipe = Pipeline.from_config(config_path, data_dir)
    return pipe.measure()
