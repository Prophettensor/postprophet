"""
Pipeline — the framework's public API. Coordinates the closed loop:

    harvest()    pull seeds from configured connectors  (zero LLM)
    ideas()      surface + rank ideas against vision     (zero LLM)
    plan()       pair strategies with ideas (bandit)     (zero LLM)
    draft()      [LLM step] turn a plan into post text   (in-session only)
    publish()    post — manual (human), approve (human approve then auto), or auto (needs auth)
    measure()    read real outcomes + retrain bandit      (no_agent cron)
    learn()      alias for measure()

A Pipeline is bound to ONE builder instance (one config). Use it directly or via
the CLI entrypoint. Connectors are built from the instance config on demand.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .config import (load_config, InstanceConfig, ConnectorConfig, VisionConfig,
                     _merge_vision, _merge_voice)
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
        # In "just connect your agent" mode, derive vision/voice from agent memory
        # and the connected repos instead of requiring them hand-authored.
        if config.loop.derive:
            self._apply_derived_context()

    def _apply_derived_context(self):
        from . import derive
        repo_paths = self._repo_paths()
        vision, voic, memory = derive.derive(repo_paths=repo_paths)
        # merge: derived fills gaps, explicit config values win
        self.config.vision = _merge_vision(self.config.vision, vision)
        self.config.voice = _merge_voice(self.config.voice, voic)

    def _repo_paths(self):
        paths = []
        for c in self.config.connectors:
            opts = c.options or {}
            if c.type == "git" and opts.get("repos"):
                r = opts["repos"]
                paths.extend(r if isinstance(r, list) else [r])
        return paths

    # ---- lifecycle ----
    @classmethod
    def from_config(cls, config_path: str, data_dir: str) -> "Pipeline":
        cfg = load_config(config_path)
        return cls(cfg, data_dir)

    def for_project(self, name: str) -> "Pipeline":
        """Return a Pipeline scoped to ONE project. This is how a multi-project
        agent markets one project (or one at a time) from the same config.

        The scoped pipeline:
          - only runs the connectors for THIS project's repos/agents/feeds
          - uses the project's own vision (positioning/problem/market), which is
            derived from the project's repos if not overridden in config
          - keeps the global voice (voice is per-person, not per-project)
        """
        proj = next((p for p in self.config.projects if p.name == name), None)
        if proj is None:
            raise KeyError(f"no project named '{name}' in config; "
                           f"available: {[p.name for p in self.config.projects]}")

        # Build a scoped connector list for this project.
        scoped_connectors = []
        if proj.repos:
            scoped_connectors.append(ConnectorConfig(
                name=f"{name}-git", type="git", options={"repos": proj.repos}))
        for i, out in enumerate(proj.agent_outputs):
            scoped_connectors.append(ConnectorConfig(
                name=f"{name}-agents{i}", type="agent_output",
                options={"path": out, "kind": "finding"}))
        for i, feed in enumerate(proj.feeds):
            scoped_connectors.append(ConnectorConfig(
                name=f"{name}-feed{i}", type="rss", options={"feeds": [feed]}))
        # The agent's own memory always feeds every project (voice + background).
        scoped_connectors.append(ConnectorConfig(
            name=f"{name}-memory", type="agent_memory", options={}))

        # Project vision: override where set in config, else derive from project repos.
        vision = VisionConfig(
            positioning=proj.positioning,
            problem=proj.problem,
            market=proj.market,
            audience=self.config.vision.audience,
            competitors=self.config.vision.competitors,
            avoid=self.config.vision.avoid,
        )
        if not (vision.positioning or vision.problem) and proj.repos:
            from . import derive
            d, _, _ = derive.derive(repo_paths=proj.repos)
            vision = VisionConfig(
                positioning=vision.positioning or d.positioning,
                problem=vision.problem or d.problem,
                market=vision.market or d.market,
                audience=self.config.vision.audience,
                competitors=self.config.vision.competitors,
                avoid=self.config.vision.avoid,
            )

        scoped_cfg = InstanceConfig(
            name=f"{self.config.name}:{name}",
            connectors=scoped_connectors,
            vision=vision,
            voice=self.config.voice,       # voice stays global (per-person)
            account=self.config.account,
            loop=self.config.loop,
        )
        return Pipeline(scoped_cfg, self.data_dir)

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
        idea, prompt}]. The agent drafts post text from these (in-session).

        The agent's own memory is included as the VOICE/VISION source (when the
        instance has agent_memory connectors), so the agent writes in its own
        voice without any voice/vision config being required."""
        plan = plan if plan is not None else self.plan()
        memory = self._agent_memory_text()
        out = []
        for item in plan:
            prompt = build_draft_prompt(item, self.config.vision, self.config.voice,
                                        agent_memory=memory)
            out.append({**item, "prompt": prompt})
        return out

    def _agent_memory_text(self) -> str:
        """Read the agent's memory files via an agent_memory connector, returned
        as raw text for grounding the drafting prompt's voice/vision."""
        for cfg in self.config.connectors:
            if cfg.type == "agent_memory":
                c = build_connector(cfg)
                if c is None:
                    continue
                try:
                    seeds = c.collect()
                except Exception:
                    continue
                # agent_memory seeds carry the raw line in 'detail'
                return "\n".join(s.get("detail", "") for s in seeds)
        return ""

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
        if not seeds:
            return self._empty_round_guide()
        ideas = self.ideas(seeds)
        plan = self.plan(ideas, n)
        return "\n\n".join([
            f"BUILDER: {self.config.name}",
            "=== LEARNING STATE ===\n" + self.learner.summary(),
            "=== GROUNDED IDEAS ===\n" + format_idea_brief(ideas),
            "=== GENERATION PLAN ===\n" + self._format_plan(plan),
        ])

    def _empty_round_guide(self) -> str:
        """Help message shown when no connectors produced any seeds yet. Guides a
        new user to connect their stack before drafting — instead of silently
        producing ungrounded (no-seed) candidates."""
        conn_names = [c.name for c in self.config.connectors]
        lines = [
            f"BUILDER: {self.config.name}",
            "",
            "No content ideas yet — PostProphet found no seeds from your connectors.",
            "",
            "It drafts from REAL facts about what you're building, so it needs at "
            "least one source connected. Check your config:",
            "",
            f"  configured connectors: {', '.join(conn_names) or 'none'}\n",
            "Fix any of these:",
            "  - git:   point 'repos' at a repo with recent commits (not an empty dir)",
            "  - github: set 'repo' to owner/repo and GH_TOKEN/GITHUB_TOKEN in env",
            "  - agent_output: point 'path' at a dir where your agents write files",
            "  - rss:   add feed URLs for your market / competitors",
            "",
            "Once a connector returns at least one seed, run 'round' again and you'll",
            "get ranked, grounded content ideas to draft from.",
        ]
        return "\n".join(lines)

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
