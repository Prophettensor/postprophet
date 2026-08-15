"""
Reward function — encoded from xai-org/x-algorithm's shipped ranking weights.

Source: home-mixer/params/param.rs (the For You feed ranking action weights X
open-sourced Aug 2026), plus the bidirectional-follow boost documented in
docs/BIDIRECTIONAL_BOOST_CHANGE.md.

IMPORTANT (from X's own code comments): the ranking weights scale PREDICTED
ACTION PROBABILITIES, not raw engagement counts. We do NOT reproduce that (we
have no per-viewer probability model). We only use the weights as an ORDINAL
prior on the relative value of each measurable public action, normalized so the
numbers are interpretable. This is a ranking prior for choosing among our own
drafts, NOT a reach predictor — anyone reading these ratios as "1 report cancels
468 likes" is misreading X's own model.

Measurable actions (public_metrics): reply, retweet, like, quote, bookmark,
impression.
"""

# Ordinal value weights per measurable action, derived from X's shipped params.
# quote and reply are the top (5.0); bookmark is a dwell/intent proxy.
ACTION_WEIGHTS = {
    "quote_count":     5.0,
    "reply_count":     5.0,
    "bookmark_count":  3.0,   # dwell / save-for-later intent proxy
    "retweet_count":   1.0,
    "like_count":      0.5,
}

# Structural levers from the repo (shape strategy, not part of the score itself).
MUTUAL_REPLY_BOOST = 15.0   # bidirectional follow reply weight (rolled 20->15 Jul 24 2026)
OUT_OF_NETWORK_DISCOUNT = 0.75
AUTHOR_DIVERSITY_DECAY = 0.5
AUTHOR_DIVERSITY_FLOOR = 0.25

# Negative weights (mass-action gaming suppression) — informational.
NEGATIVE_WEIGHTS = {
    "report": -234.0, "mute": -58.8, "not_interested": -43.2, "block": -31.2,
}


def raw_reward(metrics: dict) -> float:
    """Weighted sum of measurable engagement on a single post."""
    total = 0.0
    for key, w in ACTION_WEIGHTS.items():
        total += w * float(metrics.get(key, 0))
    return total


def reward_per_impression(metrics: dict) -> float:
    """Reward normalized by reach — a small post that converts beats a big post
    that doesn't. The learner's per-post signal."""
    r = raw_reward(metrics)
    imp = float(metrics.get("impression_count", 0) or 0)
    return r / imp if imp > 0 else 0.0


def engagement_rate(metrics: dict) -> float:
    """Total measurable engagement per impression (0..1 typically)."""
    imp = float(metrics.get("impression_count", 0) or 0)
    if imp <= 0:
        return 0.0
    total_eng = sum(float(metrics.get(k, 0)) for k in ACTION_WEIGHTS)
    return total_eng / imp


def conversational_score(metrics: dict) -> float:
    """Share of reward from high-value conversation actions (reply+quote+bookmark).
    High = started discussion; low = liked but forgotten."""
    total = raw_reward(metrics)
    if total <= 0:
        return 0.0
    conv = (ACTION_WEIGHTS["reply_count"] * metrics.get("reply_count", 0)
            + ACTION_WEIGHTS["quote_count"] * metrics.get("quote_count", 0)
            + ACTION_WEIGHTS["bookmark_count"] * metrics.get("bookmark_count", 0))
    return conv / total
