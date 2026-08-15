"""
Strategy playbook — the generation prior encoded from X's shipped ranking weights.

Each DRAFT is tagged with one STRATEGY that encodes WHICH high-value action it is
engineered to elicit, plus the network lever (mutual vs out-of-network). The
learner reweights strategies from real measured outcomes, so the playbook is a
PRIOR, not a rule.

Rationale from x-algorithm params: reply 5.0 (->20.0 mutual), quote 5.0,
share 2.0, follow_author 4.0, favorite 0.5. Out-of-network discount 0.75,
author-diversity decay 0.5. So the highest-leverage actions to engineer are
reply, quote, share, follow.
"""

STRATEGIES = {
    "reply_starter": {
        "action": "reply",
        "x_weight": 5.0,
        "levers": [
            "End on a direct question or an open position a reader can push against.",
            "State a take and invite the disagreement explicitly (controversial-but-true framing).",
            "Leave a deliberate gap / ambiguity that begs a clarifying reply.",
            "Reference a named party or group so they are drawn to respond.",
        ],
        "prompt_hint": "Optimize for REPLIES: make it effortless and low-cost to respond. Ask a real question or stake a debatable claim.",
    },
    "quote_worthy": {
        "action": "quote",
        "x_weight": 5.0,
        "levers": [
            "Make the post a self-contained, bold, quotable statement.",
            "Lead with the take in the first line (quote cards show the first ~2 lines).",
            "Be newsy / of-the-moment so people quote it to react publicly.",
            "State a claim strong enough that quoting + adding a take feels natural.",
        ],
        "prompt_hint": "Optimize for QUOTES: read as a standalone quotable statement people screenshot or quote with their own take. Lead with the punchline.",
    },
    "share_worthy": {
        "action": "share",
        "x_weight": 2.0,
        "levers": [
            "Make it reference-worthy: a list, a finding, a resource others forward.",
            "Identity-affirming framing (this is what people who X believe/do).",
            "Practical utility someone sends to a colleague or friend.",
            "A single, easily-repeated takeaway line.",
        ],
        "prompt_hint": "Optimize for SHARES: content someone forwards or saves. Make it a digestible reference-worthy nugget with one repeatable takeaway.",
    },
    "dwell_holder": {
        "action": "dwell",
        "x_weight": 3.0,
        "levers": [
            "Open a loop that only closes by reading to the end.",
            "Use specificity / concrete detail that rewards careful reading.",
            "Build tension: set up a question, defer the payoff.",
            "Thread-form if the idea needs room to unfold.",
        ],
        "prompt_hint": "Optimize for DWELL / read-time: hook and make them finish. Use a tension loop and concrete, specific detail. Do not make it instantly skimmable to the point it is forgettable.",
    },
    "follow_bait": {
        "action": "follow",
        "x_weight": 4.0,
        "levers": [
            "Signal a recurring beat / ongoing series a reader will want to follow.",
            "Establish a distinct voice or niche ('I cover X, here's the pattern').",
            "Show the value of following (what they get by staying).",
            "Out-of-network framing: assume the reader does not know you yet.",
        ],
        "prompt_hint": "Optimize for FOLLOWS / new followers: written for a stranger who does not know you. Establish a distinct voice and a reason to follow for more.",
    },
}

NETWORK_LEVERS = {
    "mutual": {
        "desc": "Engage accounts that already follow you (bidirectional follow reply boost +15).",
        "prompt_hint": "Target YOUR existing mutuals / engaged audience: reference ongoing threads, deepen an existing conversation. Replies from mutuals are the single highest-value action (weighted ~20 vs 5).",
    },
    "out_of_network": {
        "desc": "Write for people who do not follow you (out-of-network discount 0.75 = the way to grow).",
        "prompt_hint": "Target NEW / out-of-network readers: assume zero shared context. Make it self-contained and discoverable; this is how you escape your follower graph (virality).",
    },
}

DENSITY_RULE = {
    "desc": "Author-diversity decay 0.5 (floor 0.25) punishes posting volume. Each post must clear a high bar on its own.",
    "prompt_hint": "This draft must stand on its own — the algorithm discounts repeated same-author posting. Quality per post, not frequency.",
}


def strategy_list():
    return list(STRATEGIES.keys())


def strategy_prompt_hint(name):
    s = STRATEGIES.get(name)
    return s["prompt_hint"] if s else ""
