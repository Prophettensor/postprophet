"""
Builder-instance config schema and loader.

The framework is shared; a builder instance is ONE config file. This schema
defines that file. Everything the pipeline needs is here:

    connectors   which context sources to pull from (name -> connector config)
    vision       positioning, audience, competitors, problem  (why it matters)
    account      platform + handle for the publishing/measurement side
    voice        content constraints (topics to cover or avoid, tone)
    loop         publish mode (manual/approve/auto), strategy mix, cadence

The config is declarative so a builder can wire up their stack without touching
code. Connector configs are passed to the matching connector's init; unknown
connectors are ignored with a warning so a partial config still runs.

Loaded via load_config(path). Returns a plain namespace-like dict. No secrets in
here — auth (e.g. xurl) is read from the environment/credentials store, not the
config file, so a builder can commit their config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class ConnectorConfig:
    name: str
    type: str
    enabled: bool = True
    options: dict = field(default_factory=dict)


@dataclass
class VisionConfig:
    positioning: str = ""
    audience: List[str] = field(default_factory=list)
    competitors: List[str] = field(default_factory=list)
    problem: str = ""
    market: str = ""
    # optional guardrails: topics the content must never claim, claims to avoid
    avoid: List[str] = field(default_factory=list)


@dataclass
class VoiceConfig:
    tone: str = "direct"
    do_not: List[str] = field(default_factory=list)  # e.g. ["emojis", "hashtags"]
    max_chars: int = 280


@dataclass
class AccountConfig:
    platform: str = "x"
    handle: str = ""
    # publish_mode: "manual" (agent drafts, human posts by hand),
    #               "approve" (agent drafts, human approves, then auto-posts),
    #               "auto" (agent drafts and posts fully autonomously)
    publish_mode: str = "manual"

    def __post_init__(self):
        valid = {"manual", "approve", "auto"}
        if self.publish_mode not in valid:
            raise ValueError(f"publish_mode must be one of {sorted(valid)}, "
                             f"got '{self.publish_mode}'")


@dataclass
class LoopConfig:
    candidates_per_round: int = 5
    strategies: List[str] = field(default_factory=list)  # empty = all
    # how often the tracker (no_agent cron) resolves outcomes
    measure_interval_hours: int = 12
    # if True, derive vision/voice from agent memory + connected repos instead of
    # requiring them in the config (the "just connect your agent" mode)
    derive: bool = False


@dataclass
class ProjectConfig:
    """A named marketing scope for a multi-project agent.

    One agent may build many things. A Project is one of those things you want to
    market. It carries the repos/agents that feed it and an optional per-project
    vision override (vision is per-project; voice stays global to the agent).

    If the config declares projects, rounds are scoped with --project NAME and
    only that project's connectors/vision are used. Without projects, the config
    behaves as a single-project instance (backward compatible).
    """
    name: str
    # repos + agent sources that feed THIS project's marketing
    repos: List[str] = field(default_factory=list)
    agent_outputs: List[str] = field(default_factory=list)
    feeds: List[str] = field(default_factory=list)
    # optional per-project vision override (derived from its repos if empty)
    positioning: str = ""
    problem: str = ""
    market: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectConfig":
        return cls(
            name=str(d.get("name", "")),
            repos=list(d.get("repos", [])),
            agent_outputs=list(d.get("agent_outputs", [])),
            feeds=list(d.get("feeds", [])),
            positioning=str(d.get("positioning", "")),
            problem=str(d.get("problem", "")),
            market=str(d.get("market", "")),
        )


@dataclass
class InstanceConfig:
    name: str
    connectors: List[ConnectorConfig] = field(default_factory=list)
    vision: VisionConfig = field(default_factory=VisionConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    projects: List[ProjectConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceConfig":
        return cls(
            name=str(d.get("name", "builder")),
            connectors=[
                ConnectorConfig(**c) for c in d.get("connectors", [])
            ],
            vision=VisionConfig(**d.get("vision", {})),
            voice=VoiceConfig(**d.get("voice", {})),
            account=AccountConfig(**d.get("account", {})),
            loop=LoopConfig(**d.get("loop", {})),
            projects=[
                ProjectConfig.from_dict(p) for p in d.get("projects", [])
            ],
        )


def load_config(path: str) -> InstanceConfig:
    """Load a builder-instance config from YAML. Raises FileNotFoundError."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return InstanceConfig.from_dict(raw)


def _merge_vision(base: VisionConfig, derived: VisionConfig) -> VisionConfig:
    """Merge derived vision into the base; explicit base values win, derived fill
    gaps. Used by derive mode so a hand-authored config is never overwritten."""
    return VisionConfig(
        positioning=base.positioning or derived.positioning,
        problem=base.problem or derived.problem,
        market=base.market or derived.market,
        audience=base.audience or derived.audience,
        competitors=base.competitors or derived.competitors,
        avoid=base.avoid or derived.avoid,
    )


def _merge_voice(base: VoiceConfig, derived: VoiceConfig) -> VoiceConfig:
    """Merge derived voice into the base; explicit base values win, derived fill
    gaps (e.g. memory saying 'no emojis' is only added if base doesn't already
    forbid it)."""
    return VoiceConfig(
        tone=base.tone if base.tone != "direct" else (derived.tone or base.tone),
        do_not=sorted(set(base.do_not) | set(derived.do_not)),
        max_chars=base.max_chars or derived.max_chars,
    )


def write_example_config(path: str):
    """Emit a documented example config a builder can copy and edit."""
    example = {
        "name": "my-build",
        "connectors": [
            {"name": "git", "type": "git", "options": {"repos": ["."]}},
            {"name": "research", "type": "agent_output",
             "options": {"path": "agent-outputs/", "exts": [".json", ".md"]}},
        ],
        "vision": {
            "positioning": "We build X for Y so that Z.",
            "audience": ["builder personas"],
            "competitors": ["names"],
            "problem": "the problem we solve",
            "market": "one-line market description",
            "avoid": ["claims we must never make"],
        },
        "voice": {
            "tone": "direct",
            "do_not": ["emojis", "hashtags"],
            "max_chars": 280,
        },
        "account": {
            "platform": "x",
            "handle": "your_handle",
            "publish_mode": "manual",
        },
        "loop": {
            "candidates_per_round": 5,
            "strategies": [],
            "measure_interval_hours": 12,
            "derive": True,
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(example, f, sort_keys=False)
    return path
