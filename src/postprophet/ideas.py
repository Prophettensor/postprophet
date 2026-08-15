"""
Idea layer — surfaces and ranks content ideas from the seed stream, grounded in
the builder's vision and market.

This is where "understands the vision and market you're within" lives. It does
NOT draft content and it does NOT invent facts. It turns raw seeds (what shipped,
findings, signals) into candidate IDEAS by framing them against:
    - the builder's positioning / problem / market (vision)
    - the content strategy playbook (which high-value X action to target)

Deterministic idea scoring (zero LLM): each idea is scored on how well its seed
aligns with the vision profile. This is a PRIOR for generation, not a prediction.

Idea shape:
    seed      the originating Seed (ground truth)
    angle     a framing label the drafting step will expand (e.g. "announcement",
              "contrarian take", "pattern lesson", "customer insight")
    strategy  recommended strategy (reply_starter / quote_worthy / ...) from the
              playbook, chosen by the bandit in the generation step
    fit       0..1 how well this seed fits the builder's vision/market
    why       short human reason for the fit score

Angles map a seed kind to a content approach:
    ship      -> announcement, milestone, build-in-public
    finding   -> pattern lesson, insight, thought-leadership
    signal    -> customer insight, market take, contrarian
    event     -> timely take, reaction
    milestone -> milestone, recap
"""

from __future__ import annotations

from typing import List

from .config import VisionConfig
from .context import Seed

# seed kind -> suggested content angles (the drafting step picks one + strategy)
ANGLE_BY_KIND = {
    "ship":      ["announcement", "build-in-public", "milestone"],
    "finding":   ["pattern lesson", "thought-leadership", "insight"],
    "signal":    ["customer insight", "market take", "contrarian"],
    "event":     ["timely take", "reaction"],
    "milestone": ["milestone", "recap"],
}

# Low-cost keyword heuristics for vision alignment. A builder can extend these in
# config. Matching on the seed title/detail + vision fields. This is a rough prior,
# deliberately simple (ELI5): does this fact mention the builder's domain/problem?
_FIT_WORDS = None  # built per-call from vision


def _tokenize(text: str) -> set:
    import re
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _fit_score(seed: Seed, vision: VisionConfig) -> tuple[float, str]:
    """0..1 vision fit. Simple, transparent, no LLM.

    Combines two signals:
      1. source-affinity: does the seed's source/repo relate to the vision domain?
         (e.g. a seed from 'postprophet' repo against a vision about content
         agents). Strongest signal a builder cares.
      2. keyword overlap: does the seed text literally mention vision terms?
         Softened so short commit titles aren't punished.
    """
    if not vision.positioning and not vision.problem:
        return 0.5, "no vision configured (neutral)"

    text = f"{seed.title} {seed.detail}"
    toks = _tokenize(text)

    # Vision keyword bag from positioning + problem + market + audience.
    bag = set()
    for field in (vision.positioning, vision.problem, vision.market):
        bag |= _tokenize(field)
    for a in vision.audience:
        bag |= _tokenize(a)

    reasons = []
    score = 0.0

    # Signal 1: source affinity — repo name / source label appearing in vision.
    source = seed.source.lower()
    source_toks = _tokenize(source) | set(source.split(":"))
    src_shared = source_toks & bag
    if src_shared:
        score += 0.45
        reasons.append(f"source '{seed.source}' matches vision ({len(src_shared)})")

    # Signal 2: keyword overlap, softened (overlap / bag size), capped.
    if bag:
        overlap = len(toks & bag)
        kw_score = min(1.0, overlap / max(1, len(bag))) * 0.55
        if overlap >= 1:
            score += kw_score
            reasons.append(f"{overlap} keyword(s) overlap vision")

    # Neutral prior so nothing is a hard zero — keeps exploration alive early.
    score = min(1.0, 0.15 + score)

    # Floor on explicit avoid topics (>=2 shared tokens or phrase present).
    for avoid in vision.avoid:
        avoid_toks = _tokenize(avoid)
        shared = avoid_toks & toks
        phrase_present = avoid.lower() in text.lower()
        if avoid_toks and (phrase_present or len(shared) >= 2):
            score *= 0.2
            return score, f"matches an avoid topic: '{avoid}'"

    if score >= 0.4:
        return round(score, 2), "; ".join(reasons) or "reasonable default fit"
    return round(score, 2), "; ".join(reasons) or "low direct overlap"


def surface_ideas(seeds: List[Seed], vision: VisionConfig,
                  limit: int = 10) -> List[dict]:
    """Frame a stream of seeds into ranked content ideas. Deterministic.

    Returns ideas sorted by fit desc. Each idea is a dict the drafting step can
    act on, carrying its ground-truth seed and a recommended angle.
    """
    ideas = []
    for seed in seeds:
        # internal seeds (agent memory = voice/context, not content) never surface
        if "internal" in seed.tags:
            continue
        fit, why = _fit_score(seed, vision)
        angles = ANGLE_BY_KIND.get(seed.kind, ["general"])
        ideas.append({
            "seed": seed.to_dict(),
            "angles": angles,
            "primary_angle": angles[0],
            "fit": round(fit, 2),
            "why": why,
        })
    ideas.sort(key=lambda i: -i["fit"])
    return ideas[:limit]


def format_idea_brief(ideas: List[dict], limit: int = 10) -> str:
    """Render ranked ideas for the drafting LLM (ground truth + vision framing)."""
    lines = ["RANKED CONTENT IDEAS (grounded in real facts — frame these, do not invent):"]
    for i, idea in enumerate(ideas, 1):
        seed = idea["seed"]
        lines.append(
            f"{i}. fit={idea['fit']:.2f} | {seed['kind']} | {idea['primary_angle']} "
            f"\n   FACT: {seed['title']}"
        )
        if seed.get("detail"):
            lines.append(f"   DETAIL: {seed['detail'][:200]}")
        if idea["why"]:
            lines.append(f"   why: {idea['why']}")
        lines.append(f"   angle options: {', '.join(idea['angles'])}")
    return "\n".join(lines)
