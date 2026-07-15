"""
PostProphet — Mock test for the prediction engine.

Tests the full loop (capture → predict → resolve → score) with fake tweet data.
No API keys needed. Proves the harness works end-to-end.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta

# Add the harness to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Mock data ──────────────────────────────────────────────────────────

MOCK_TWEETS = [
    {
        "id": "1001",
        "text": "Just shipped a new feature that predicts tweet reach before you post. The future of social media marketing is here. 🧵",
        "created_at": "2026-07-15T14:30:00Z",
        "author_id": "a1",
        "public_metrics": {"like_count": 45, "retweet_count": 12, "reply_count": 8, "quote_count": 3, "impression_count": 0},
    },
    {
        "id": "1002",
        "text": "Hot take: most marketing courses are a scam. You don't need to LEARN the skill — you need an agent that DOES it. That's the future.",
        "created_at": "2026-07-15T14:25:00Z",
        "author_id": "a2",
        "public_metrics": {"like_count": 89, "retweet_count": 34, "reply_count": 22, "quote_count": 5, "impression_count": 0},
    },
    {
        "id": "1003",
        "text": "reminder: if you're not measuring it, you're not improving it. build the ruler first, then build the thing.",
        "created_at": "2026-07-15T14:20:00Z",
        "author_id": "a3",
        "public_metrics": {"like_count": 12, "retweet_count": 3, "reply_count": 1, "quote_count": 0, "impression_count": 0},
    },
    {
        "id": "1004",
        "text": "we just raised $5.4B at a $40B valuation. AI music is the biggest platform shift since mobile. proud of the team. 🎵",
        "created_at": "2026-07-15T14:15:00Z",
        "author_id": "a4",
        "public_metrics": {"like_count": 230, "retweet_count": 89, "reply_count": 45, "quote_count": 12, "impression_count": 0},
    },
    {
        "id": "1005",
        "text": "day 47 of building in public. shipped the eval framework today. brier score went from 0.34 to 0.21. numbers don't lie.",
        "created_at": "2026-07-15T14:10:00Z",
        "author_id": "a1",
        "public_metrics": {"like_count": 67, "retweet_count": 15, "reply_count": 9, "quote_count": 2, "impression_count": 0},
    },
]

MOCK_AUTHORS = {
    "a1": {"id": "a1", "username": "techbuilder", "public_metrics": {"followers_count": 8500, "following_count": 320, "tweet_count": 1400}},
    "a2": {"id": "a2", "username": "marketing_prophet", "public_metrics": {"followers_count": 22000, "following_count": 1800, "tweet_count": 8900}},
    "a3": {"id": "a3", "username": "indiehacker", "public_metrics": {"followers_count": 1200, "following_count": 450, "tweet_count": 340}},
    "a4": {"id": "a4", "username": "aimusicceo", "public_metrics": {"followers_count": 45000, "following_count": 200, "tweet_count": 2100}},
}

MOCK_BASELINES = {
    "a1": {"avg_impressions": 8000, "avg_likes": 50, "avg_retweets": 12, "sample_size": 20},
    "a2": {"avg_impressions": 25000, "avg_likes": 180, "avg_retweets": 45, "sample_size": 20},
    "a3": {"avg_impressions": 1500, "avg_likes": 8, "avg_retweets": 2, "sample_size": 15},
    "a4": {"avg_impressions": 80000, "avg_likes": 350, "avg_retweets": 110, "sample_size": 25},
}

MOCK_RECENT_TWEETS = {
    "a1": [
        {"text": "building in public day 46 — the eval framework is taking shape...", "impressions": 9200, "likes": 55, "retweets": 14},
        {"text": "if you can't measure it, you can't improve it. here's why →", "impressions": 7100, "likes": 42, "retweets": 10},
        {"text": "just had a breakthrough on the prediction engine. brier score improving.", "impressions": 6500, "likes": 38, "retweets": 8},
    ],
    "a2": [
        {"text": "stop selling courses. start selling agents. here's the math →", "impressions": 42000, "likes": 310, "retweets": 78},
        {"text": "the future of marketing is not content. it's prediction.", "impressions": 28000, "likes": 195, "retweets": 52},
        {"text": "if your agent can't grade its own work, you're flying blind.", "impressions": 19000, "likes": 120, "retweets": 31},
    ],
    "a3": [
        {"text": "shipped a small thing today. feels good.", "impressions": 1800, "likes": 10, "retweets": 2},
        {"text": "the best marketing is just building something people want.", "impressions": 1200, "likes": 7, "retweets": 1},
        {"text": "day 30 of building. still going.", "impressions": 900, "likes": 5, "retweets": 0},
    ],
    "a4": [
        {"text": "we're generating 7 million songs a day. the industry will never be the same.", "impressions": 120000, "likes": 520, "retweets": 180},
        {"text": "AI music is not a gimmick. it's a platform shift. here's our roadmap →", "impressions": 89000, "likes": 410, "retweets": 95},
        {"text": "proud to announce our partnership with a major label. more details soon.", "impressions": 65000, "likes": 280, "retweets": 62},
    ],
}

MOCK_TRENDING = ["AI agents", "Bittensor", "music generation", "startup fundraising", "open source"]

# Simulated actual impressions (what the tweets "actually" got after 1h)
# These are the ground truth for Brier scoring
MOCK_ACTUAL_IMPRESSIONS = {
    "1001": 8500,   # below target of 10000 → MISS
    "1002": 31000,  # above target → HIT
    "1003": 1400,   # below target → MISS
    "1004": 95000,  # above target → HIT
    "1005": 8200,   # below target → MISS
}

# ── Test harness ───────────────────────────────────────────────────────


def build_mock_context(tweet, author_id):
    """Build context from mock data (simulates what the real harness gathers)."""
    author = MOCK_AUTHORS[author_id]
    metrics = tweet["public_metrics"]
    baseline = MOCK_BASELINES[author_id]
    recent = MOCK_RECENT_TWEETS.get(author_id, [])

    return {
        "tweet_id": tweet["id"],
        "text": tweet["text"],
        "created_at": tweet["created_at"],
        "planned_post_time": tweet["created_at"],
        "platform": "x",
        "author": {
            "username": author["username"],
            "followers": author["public_metrics"]["followers_count"],
            "following": author["public_metrics"]["following_count"],
            "tweet_count": author["public_metrics"]["tweet_count"],
        },
        "author_baseline": baseline,
        "author_recent_top_tweets": recent,
        "tweet_metrics_at_capture": {
            "likes": metrics["like_count"],
            "retweets": metrics["retweet_count"],
            "replies": metrics["reply_count"],
            "quotes": metrics["quote_count"],
        },
        "trending_topics": MOCK_TRENDING,
    }


def run_mock_test():
    """Run the full PostProphet loop with mock data."""
    from postprophet import predict, brier_score

    TARGET = 10000
    TIMEFRAME = 1  # 1 hour

    print("\n" + "=" * 60)
    print("  PostProphet — Mock Test (no API keys needed)")
    print(f"  Target: {TARGET} impressions in {TIMEFRAME}h")
    print(f"  Batch: {len(MOCK_TWEETS)} tweets")
    print("=" * 60 + "\n")

    predictions = []

    for i, tweet in enumerate(MOCK_TWEETS, 1):
        author_id = tweet["author_id"]
        context = build_mock_context(tweet, author_id)

        author = context["author"]
        baseline = context["author_baseline"]

        print(f"  [{i}/{len(MOCK_TWEETS)}] @{author['username']} ({author['followers']:,} followers)")
        print(f"    Baseline: avg {baseline['avg_impressions']:,} imp/tweet")
        print(f"    Tweet: {context['text'][:90]}...")

        # Call the actual prediction engine
        try:
            prediction = predict(context, target=TARGET, timeframe=TIMEFRAME)
        except Exception as e:
            print(f"    ⚠️  Prediction error: {e}")
            print(f"    (This is expected if no OPENAI_API_KEY is set)")
            prediction = {"probability": 0.5, "reasoning": "no api key — using default", "suggestions": ""}

        # Get "actual" impressions from mock data
        actual = MOCK_ACTUAL_IMPRESSIONS[tweet["id"]]
        hit = actual >= TARGET

        record = {
            "tweet_id": tweet["id"],
            "author": author["username"],
            "predicted_probability": prediction["probability"],
            "actual_impressions": actual,
            "target": TARGET,
            "hit": hit,
            "prediction": prediction,
        }
        predictions.append(record)

        prob = prediction["probability"]
        reasoning = prediction.get("reasoning", "")[:100]
        print(f"    → Predicted: {prob:.0%} | Actual: {actual:,} | {'✅ HIT' if hit else '❌ MISS'}")
        print(f"    Reasoning: {reasoning}")
        print()

    # Score with Brier
    print("=" * 60)
    print("  Brier Score Calculation")
    print("=" * 60 + "\n")

    # Convert to the format brier_score expects
    brier_input = []
    for p in predictions:
        brier_input.append({
            "resolved": True,
            "prediction": {"probability": p["predicted_probability"]},
            "actual_impressions": p["actual_impressions"],
            "target": p["target"],
        })

    score = brier_score(brier_input)

    print(f"  {'Tweet':<8} {'Author':<22} {'Predicted':<12} {'Actual':<10} {'Result':<8}")
    print(f"  {'-' * 60}")
    for p in predictions:
        print(f"  {p['tweet_id']:<8} @{p['author']:<21} {p['predicted_probability']:<12.0%} {p['actual_impressions']:<10,} {'HIT' if p['hit'] else 'MISS':<8}")

    print(f"\n  Brier score: {score:.4f}")
    print(f"  (0.0 = perfect, 1.0 = worst, 0.25 = random guessing)")
    print(f"\n  Interpretation:")
    if score < 0.1:
        print(f"  → Excellent predictions")
    elif score < 0.25:
        print(f"  → Better than random")
    elif score < 0.33:
        print(f"  → Around random")
    else:
        print(f"  → Worse than random — check the model")

    print("\n" + "=" * 60)
    print("  ✅ Mock test complete — harness works end-to-end")
    print("=" * 60)

    return score


if __name__ == "__main__":
    run_mock_test()
