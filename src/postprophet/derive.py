"""
Derivation layer — auto-extracts vision, positioning, problem, and voice from
the agent's own memory and the connected repos' READMEs, instead of asking the
user to hand-author them.

The insight: a builder's agent already KNOWS the vision, market, and voice. It
lives in agent memory (MEMORY.md / USER.md), in the READMEs of what they build,
and in their plans. PostProphet should read that, not re-ask.

This is deterministic extraction (zero LLM) — it pulls candidate signals:
    - vision:   README first lines, repo descriptions, memory lines about the
                product/market/problem
    - voice:    explicit preferences already in memory (emojis, tone, fluff)
                and the config's voice block (which acts as overrides)

The drafting LLM then uses these as grounded context. Where extraction finds
nothing meaningful, it falls back to neutral defaults so nothing breaks.
"""

from __future__ import annotations

import os
import re

from .config import VisionConfig, VoiceConfig

# ---- voice hints we can recognize in memory/preferences ----
_EMOJI_HINT = re.compile(r"\b(no )?emoji", re.I)
_FLUFF_HINT = re.compile(r"\bno fluff\b|\bno marketing fluff\b|\bdry\b|\bspec\b", re.I)
_CORNY_HINT = re.compile(r"\bno corny\b|\bno corny copy\b|\bno house\b", re.I)
_HASHTAG_HINT = re.compile(r"\bno hashtags?\b|\bhashtags?\b", re.I)


def read_memory(memory_paths=None) -> str:
    """Concatenate agent memory files (MEMORY.md, USER.md, etc.) into one string.
    memory_paths: list of paths; if None, look in common agent-home locations."""
    if memory_paths is None:
        candidates = []
        for home in (os.environ.get("HERMES_HOME", "~/.hermes"), "~/.hermes"):
            base = os.path.expanduser(home)
            candidates += [
                os.path.join(base, "memories", "MEMORY.md"),
                os.path.join(base, "memories", "USER.md"),
            ]
        memory_paths = candidates
    parts = []
    for p in memory_paths:
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            try:
                with open(p, errors="ignore") as f:
                    parts.append(f.read())
            except OSError:
                continue
    return "\n".join(parts)


def extract_voice(memory: str = "", voice: VoiceConfig | None = None) -> VoiceConfig:
    """Derive a VoiceConfig from memory preferences, falling back to the user's
    explicit voice block (which always wins where set)."""
    v = voice or VoiceConfig()

    do_not = set(v.do_not)
    if _EMOJI_HINT.search(memory):
        do_not.add("emojis")
    if _HASHTAG_HINT.search(memory):
        do_not.add("hashtags")

    tone = v.tone
    if tone == "direct" and (_FLUFF_HINT.search(memory) or _CORNY_HINT.search(memory)):
        tone = "direct, no fluff"  # signal the drafting LLM to stay dry

    return VoiceConfig(tone=tone, do_not=sorted(do_not), max_chars=v.max_chars)


def _readme_head(repo_path):
    for name in ("README.md", "README", "readme.md"):
        p = os.path.join(repo_path, name)
        if os.path.exists(p):
            try:
                with open(p, errors="ignore") as f:
                    lines = [l.strip() for l in f.read(1200).splitlines() if l.strip()]
                for l in lines:
                    if l and not l.startswith(("#", "!", "[", "<")) and len(l) > 3:
                        return l
            except OSError:
                continue
    return ""


def _memory_topic_lines(memory: str, keywords) -> str:
    """Pull memory lines mentioning vision-relevant keywords."""
    hits = []
    for line in memory.splitlines():
        ll = line.lower()
        if any(k in ll for k in keywords):
            hits.append(line.strip())
    return " ".join(hits)


def derive_vision(memory: str = "", repo_paths=None) -> VisionConfig:
    """Derive a VisionConfig from agent memory + connected repo READMEs.

    Deterministic extraction: positioning from README leads, market/problem from
    memory lines, avoid from explicit guardrails. If nothing is found, leaves
    fields empty (the pipeline falls back to neutral) rather than guessing.
    """
    positioning = ""
    problem = ""
    market = ""
    audience = []

    # positioning: strongest signal is the first real README line of a connected repo
    for rp in (repo_paths or []):
        head = _readme_head(rp)
        if head:
            positioning = head
            break

    # problem/market: look in memory for lines about what the builder is doing
    mem_topic = _memory_topic_lines(memory, ["building", "build", "market", "product",
                                             "competition", "problem", "subnet", "repo"])
    if mem_topic:
        # use the first substantive memory line as problem context if no README
        if not problem:
            problem = mem_topic[:200]
        if not market:
            market = ""

    # audience: agent memory often names who it works for
    for kw in ("builders", "agents", "developers", "traders", "miners"):
        if re.search(rf"\b{kw}\b", memory, re.I):
            audience.append(kw)

    return VisionConfig(
        positioning=positioning,
        problem=problem,
        market=market,
        audience=audience,
    )


def derive(memory_paths=None, repo_paths=None,
           voice: VoiceConfig | None = None) -> tuple[VisionConfig, VoiceConfig, str]:
    """One-call derivation: read memory, produce vision + voice, return them plus
    the raw memory (for grounding in the drafting brief)."""
    memory = read_memory(memory_paths)
    vision = derive_vision(memory, repo_paths)
    voic = extract_voice(memory, voice)
    return vision, voic, memory
