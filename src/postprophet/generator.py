"""
Generation planner — decides WHICH (strategy, lever) combos to generate and how
to pair them with surfaced ideas.

Deterministic (zero LLM). It reads the bandit state to exploit winners + explore
unknowns, and pairs each chosen strategy with the highest-fit surfaced idea that
matches. The actual drafting (turning idea+strategy into post text) is the LLM
step, done in-session by the agent — never by a cron.
"""

from __future__ import annotations

from .learner import Learner
from .strategies import STRATEGIES, NETWORK_LEVERS

# Map a strategy to compatible idea kinds (so a reply_starter isn't paired with a
# dry milestone, etc.). Loose — the drafting step refines.
STRATEGY_KIND_AFFINITY = {
    "reply_starter": ["signal", "event", "finding", "ship"],
    "quote_worthy":  ["finding", "ship", "event", "milestone"],
    "share_worthy":  ["finding", "milestone", "ship"],
    "dwell_holder":  ["finding", "ship", "signal"],
    "follow_bait":   ["ship", "finding", "milestone"],
}


def choose_generation_plan(learner: Learner, ideas, n: int = 5,
                           strategies_filter=None):
    """Pair bandit-chosen (strategy, lever) combos with surfaced ideas.

    ideas: list from ideas.surface_ideas (each has a 'seed' with a 'kind').
    Returns list of plan dicts: {strategy, lever, idea, why}.
    """
    scores = learner.strategy_scores()

    combos = []
    for sname, levers in scores.items():
        if strategies_filter and sname not in strategies_filter:
            continue
        for lever, stats in levers.items():
            combos.append({
                "strategy": sname, "lever": lever,
                "mean": stats["mean"], "n": stats["n"],
            })

    explored = [c for c in combos if c["n"] == 0]
    exploited = sorted([c for c in combos if c["n"] > 0], key=lambda c: -c["mean"])

    pool = []
    i = j = 0
    while len(pool) < n and (i < len(explored) or j < len(exploited)):
        if i < len(explored):
            c = explored[i]; i += 1
            c["why"] = "explore (no data yet)"
            pool.append(c)
        if len(pool) >= n:
            break
        if j < len(exploited):
            c = exploited[j]; j += 1
            c["why"] = f"exploit (mean={c['mean']:.5f}, n={c['n']})"
            pool.append(c)

    # Pair each chosen combo with the best-fitting surfaced idea.
    plan = []
    used_ideas = set()
    for item in pool[:n]:
        idea = _pick_idea_for(item["strategy"], ideas, used_ideas)
        plan.append({**item, "idea": idea})
    return plan


def _pick_idea_for(strategy, ideas, used_ideas):
    affinity = STRATEGY_KIND_AFFINITY.get(strategy, [])
    # best-fit idea of a compatible kind, not already used
    for idea in ideas:
        key = idea["seed"]["title"]
        if key in used_ideas:
            continue
        if idea["seed"]["kind"] in affinity:
            used_ideas.add(key)
            return idea
    # fall back to the highest-fit unused idea
    for idea in ideas:
        key = idea["seed"]["title"]
        if key in used_ideas:
            continue
        used_ideas.add(key)
        return idea
    return None


def build_draft_prompt(plan_item, vision, voice):
    """Assemble the drafting instruction for one (strategy, lever) x idea. The
    agent (LLM) produces the post text from this. Grounds in the seed facts."""
    sname, lever = plan_item["strategy"], plan_item["lever"]
    sdef = STRATEGIES[sname]
    lever_def = NETWORK_LEVERS[lever]
    idea = plan_item.get("idea")
    seed = idea["seed"] if idea else {}

    p = []
    p.append("You are a content strategist drafting ONE X post. GROUND EVERYTHING "
             "IN THE FACTS BELOW. Do NOT invent metrics, facts, numbers, or claims.")
    p.append(f"\nPRIMARY OBJECTIVE: elicit a {sdef['action'].upper()} from readers "
             f"(X weights this action at {sdef['x_weight']:.1f} in ranking).")
    p.append(f"\nTHE FACT (ground truth): {seed.get('title', '(no seed)')}")
    if seed.get("detail"):
        p.append(f"CONTEXT: {seed['detail']}")
    p.append(f"SUGGESTED ANGLE: {idea.get('primary_angle', 'general')} "
             f"(options: {', '.join(idea.get('angles', []))})")
    p.append(f"\nSTRATEGY LEVERS:")
    for i, l in enumerate(sdef["levers"], 1):
        p.append(f"  {i}. {l}")
    p.append(f"\nNETWORK TARGET: {lever_def['desc']}")
    p.append(f"  {lever_def['prompt_hint']}")
    p.append(f"\nVOICE: {voice.tone}. Do NOT: {', '.join(voice.do_not)}. "
             f"Max {voice.max_chars} chars.")
    p.append("\nVISION (why it matters to the audience): "
             f"{vision.positioning} — {vision.problem}")
    p.append("\nFORMAT: output ONLY the post text on its own line. No preamble.")
    return "\n".join(p)
