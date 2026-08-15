"""
Context-input contract — the portability layer.

The product connects to ANY agent stack through a single structured "seed"
contract. A seed is one discrete fact from a builder's world: something shipped,
a finding, a customer signal, an ecosystem event. Every connector (git, agent
outputs, RSS, webhook, custom) emits seeds. The idea layer consumes seeds.

The contract is deliberately small and plain so a builder's own thin adapter can
emit it without depending on this package. A seed is a dict with:

    source   str   which connector/agent produced it (e.g. "git", "research")
    kind     str   what it is: ship | finding | signal | event | milestone
    title    str   one-line human summary (used as the idea anchor)
    detail   str   fuller context (commits, links, numbers) for grounding
    date     str   ISO date when it happened (optional)
    tags     list  free-form topical tags (optional)

The contract only carries FACTS. Interpretation (vision fit, market angle,
which action to target) happens UPSTREAM in the idea layer, never here. This
keeps connectors dumb and portable, and keeps fabrication impossible — the idea
layer is told these are ground truth to be framed, not invented.

A connector is any callable with signature:
    collect() -> list[dict]   (list of seed dicts matching the contract)

or a class with a collect() method. The product ships reference connectors in
postprophet.connectors; builders add their own for stack-specific sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

VALID_KINDS = {"ship", "finding", "signal", "event", "milestone"}


@dataclass
class Seed:
    """One discrete, verifiable fact from a builder's world."""
    source: str
    kind: str
    title: str
    detail: str = ""
    date: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"invalid seed kind: {self.kind} (use one of {VALID_KINDS})")
        if not self.title.strip():
            raise ValueError("seed title is required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Seed":
        return cls(
            source=str(d.get("source", "")),
            kind=str(d.get("kind", "event")),
            title=str(d.get("title", "")).strip(),
            detail=str(d.get("detail", "")),
            date=d.get("date"),
            tags=list(d.get("tags", [])),
        )


def validate_seed(d: dict) -> Seed:
    """Validate a raw dict (from a connector) against the contract. Raises on bad
    shape so a broken connector is caught at load, not silently mid-pipeline."""
    if not isinstance(d, dict):
        raise TypeError(f"seed must be a dict, got {type(d)}")
    return Seed.from_dict(d)


def normalize(connector_output: list) -> list[Seed]:
    """Coerce connector output (list of dicts or Seeds) to validated Seed objects."""
    out = []
    for item in connector_output:
        if isinstance(item, Seed):
            out.append(item)
        else:
            out.append(validate_seed(item))
    return out
