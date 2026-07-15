"""
SocialQuant — Prediction engine for social media reach.

A harness that captures real tweets, gathers context, asks an LLM to predict
impression probability, waits for resolution, and scores with Brier score.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta

import httpx
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────

X_BEARER_TOKEN = os.environ.get("X_BEARER_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("SOCIALQUANT_MODEL", "gpt-4o-mini")
TIMEFRAME_HOURS = int(os.environ.get("SOCIALQUANT_TIMEFRAME", "4"))
TARGET_IMPRESSIONS = int(os.environ.get("SOCIALQUANT_TARGET", "10000"))
BATCH_SIZE = int(os.environ.get("SOCIALQUANT_BATCH", "20"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.jsonl")
SCORES_FILE = os.path.join(DATA_DIR, "scores.jsonl")

# ── X API helpers ───────────────────────────────────────────────────────


def x_headers():
    return {"Authorization": f"Bearer {X_BEARER_TOKEN}"}


def search_recent_tweets(query="is:lang:en -is:retweet", max_results=50):
    """Search recent tweets using X API v2."""
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": min(max_results, 100),
        "tweet.fields": "public_metrics,created_at,author_id,lang",
        "expansions": "author_id",
        "user.fields": "public_metrics,username",
    }
    resp = httpx.get(url, headers=x_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_tweet_impressions(tweet_id):
    """Get current impression count for a tweet."""
    url = f"https://api.twitter.com/2/tweets/{tweet_id}"
    params = {"tweet.fields": "public_metrics"}
    resp = httpx.get(url, headers=x_headers(), params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {}).get("public_metrics", {}).get("impression_count", 0)


def get_trending_topics():
    """Get trending topics (WOEID 1 = worldwide)."""
    url = "https://api.twitter.com/1.1/trends/place.json?id=1"
    resp = httpx.get(url, headers=x_headers(), timeout=15)
    if resp.status_code != 200:
        return []
    data = resp.json()
    trends = data[0].get("trends", []) if data else []
    return [t["name"] for t in trends[:10]]


# ── Context gathering ────────────────────────────────────────────────────


def gather_context(tweet_data, includes):
    """Build the context object the LLM will use for prediction."""
    authors = {u["id"]: u for u in includes.get("users", [])}
    author = authors.get(tweet_data.get("author_id", ""), {})
    author_metrics = author.get("public_metrics", {})
    tweet_metrics = tweet_data.get("public_metrics", {})

    return {
        "tweet_id": tweet_data["id"],
        "text": tweet_data.get("text", ""),
        "created_at": tweet_data.get("created_at", ""),
        "author": {
            "username": author.get("username", "unknown"),
            "followers": author_metrics.get("followers_count", 0),
            "following": author_metrics.get("following_count", 0),
            "tweet_count": author_metrics.get("tweet_count", 0),
        },
        "tweet_metrics_at_capture": {
            "likes": tweet_metrics.get("like_count", 0),
            "retweets": tweet_metrics.get("retweet_count", 0),
            "replies": tweet_metrics.get("reply_count", 0),
            "quotes": tweet_metrics.get("quote_count", 0),
        },
        "trending_topics": [],
    }


# ── Prediction engine ───────────────────────────────────────────────────


PREDICTION_PROMPT = """You are a social media reach forecasting engine.

Given a tweet and its context, predict the probability (0.0 to 1.0) that this tweet will reach {target} impressions within {timeframe} hours of being posted.

Consider:
- Author's follower count and engagement baseline
- Early engagement signals (likes, retweets, replies at capture time)
- Tweet content (hook quality, topic relevance, media, length)
- What's currently trending and whether the tweet relates
- Time of day and posting patterns

Return JSON:
{{
  "probability": <float 0.0-1.0>,
  "reasoning": "<1-2 sentences explaining the prediction>",
  "suggestions": "<1 sentence on what would improve the prediction>"
}}

Tweet: {text}
Author: @{username} ({followers} followers)
Early metrics: {likes} likes, {retweets} retweets, {replies} replies
Posted at: {created_at}
Trending: {trending}
Timeframe: {timeframe}h
Target: {target} impressions"""


def predict(context: dict) -> dict:
    """Call LLM to predict impression probability."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = PREDICTION_PROMPT.format(
        target=TARGET_IMPRESSIONS,
        timeframe=TIMEFRAME_HOURS,
        text=context["text"],
        username=context["author"]["username"],
        followers=context["author"]["followers"],
        likes=context["tweet_metrics_at_capture"]["likes"],
        retweets=context["tweet_metrics_at_capture"]["retweets"],
        replies=context["tweet_metrics_at_capture"]["replies"],
        created_at=context["created_at"],
        trending=", ".join(context.get("trending_topics", [])) or "none available",
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a social media reach prediction engine. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    try:
        content = resp.choices[0].message.content
        result = json.loads(content) if content else {"probability": 0.5, "reasoning": "empty response", "suggestions": ""}
    except (json.JSONDecodeError, IndexError, TypeError):
        result = {"probability": 0.5, "reasoning": "parse error", "suggestions": ""}

    return result


# ── Eval: Brier score ───────────────────────────────────────────────────


def brier_score(predictions: list[dict]) -> float:
    """Calculate Brier score for a batch of resolved predictions.

    Brier score = mean((predicted_probability - actual_outcome)^2)
    Lower is better. 0 = perfect, 1 = worst.

    actual_outcome: 1.0 if impressions >= target, 0.0 if not.
    """
    if not predictions:
        return 1.0

    total = 0.0
    count = 0
    for p in predictions:
        if p.get("resolved") is not True:
            continue
        prob = p["prediction"]["probability"]
        actual = 1.0 if p["actual_impressions"] >= p["target"] else 0.0
        total += (prob - actual) ** 2
        count += 1

    return total / count if count > 0 else 1.0


# ── Harness: the main loop ──────────────────────────────────────────────


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_predictions() -> list[dict]:
    """Load all stored predictions."""
    if not os.path.exists(PREDICTIONS_FILE):
        return []
    with open(PREDICTIONS_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_prediction(prediction: dict):
    """Append a prediction to the JSONL store."""
    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(prediction) + "\n")


def save_score(score_data: dict):
    """Append a score record to the scores JSONL."""
    with open(SCORES_FILE, "a") as f:
        f.write(json.dumps(score_data) + "\n")


def capture_phase():
    """Phase 1: Capture real tweets and predict."""
    print("\n{'='*60}")
    print(f"  SocialQuant — Capture Phase")
    print(f"  Target: {TARGET_IMPRESSIONS} impressions in {TIMEFRAME_HOURS}h")
    print(f"  Batch: {BATCH_SIZE} tweets")
    print(f"{'='*60}\n")

    try:
        results = search_recent_tweets(max_results=BATCH_SIZE)
    except Exception as e:
        print(f"Error searching tweets: {e}")
        return []

    tweets = results.get("data", [])
    includes = results.get("includes", {})

    if not tweets:
        print("No tweets found.")
        return []

    # Try to get trending topics (optional, don't fail if it errors)
    try:
        trending = get_trending_topics()
    except Exception:
        trending = []

    predictions = []
    for i, tweet in enumerate(tweets, 1):
        context = gather_context(tweet, includes)
        context["trending_topics"] = trending

        print(f"  [{i}/{len(tweets)}] @{context['author']['username']}: {context['text'][:60]}...")

        try:
            prediction = predict(context)
        except Exception as e:
            print(f"    Prediction error: {e}")
            prediction = {"probability": 0.5, "reasoning": "error", "suggestions": ""}

        record = {
            "tweet_id": context["tweet_id"],
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "timeframe_hours": TIMEFRAME_HOURS,
            "target": TARGET_IMPRESSIONS,
            "context": context,
            "prediction": prediction,
            "resolve_after": (
                datetime.now(timezone.utc) + timedelta(hours=TIMEFRAME_HOURS)
            ).isoformat(),
            "resolved": False,
            "actual_impressions": None,
        }

        save_prediction(record)
        predictions.append(record)

        prob = prediction.get("probability", 0.5)
        reasoning = prediction.get("reasoning", "")[:80]
        print(f"    → {prob:.0%} | {reasoning}")

    print(f"\nCaptured {len(predictions)} predictions. Waiting {TIMEFRAME_HOURS}h for resolution.")
    return predictions


def resolve_phase():
    """Phase 2: Check predictions whose timeframe has elapsed, score them."""
    print("\n{'='*60}")
    print("  SocialQuant — Resolve Phase")
    print(f"{'='*60}\n")

    predictions = load_predictions()
    now = datetime.now(timezone.utc)

    resolved_batch = []
    unresolved = 0

    for p in predictions:
        if p.get("resolved"):
            continue

        resolve_after = datetime.fromisoformat(p["resolve_after"])
        if now < resolve_after:
            unresolved += 1
            continue

        # Fetch actual impressions
        try:
            actual = get_tweet_impressions(p["tweet_id"])
        except Exception as e:
            print(f"  Error fetching impressions for {p['tweet_id']}: {e}")
            continue

        p["actual_impressions"] = actual
        p["resolved"] = True

        hit = actual >= p["target"]
        prob = p["prediction"]["probability"]
        print(f"  @{p['context']['author']['username']}: predicted {prob:.0%}, got {actual:,} → {'HIT' if hit else 'MISS'}")

        resolved_batch.append(p)

    if not resolved_batch:
        print(f"  No predictions ready to resolve. {unresolved} still pending.")
        return

    # Calculate Brier score for this batch
    score = brier_score(resolved_batch)
    print(f"\n  Brier score: {score:.4f} (0=perfect, 1=worst)")

    # Save score record
    score_record = {
        "scored_at": now.isoformat(),
        "batch_size": len(resolved_batch),
        "brier_score": score,
        "model": OPENAI_MODEL,
        "timeframe_hours": TIMEFRAME_HOURS,
        "target": TARGET_IMPRESSIONS,
    }
    save_score(score_record)

    # Rewrite predictions file with resolved updates
    with open(PREDICTIONS_FILE, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"  Resolved {len(resolved_batch)} predictions. Score saved.")
    return score


def report_phase():
    """Phase 3: Show summary of all scores."""
    if not os.path.exists(SCORES_FILE):
        print("\nNo scores yet.")
        return

    print("\n{'='*60}")
    print("  SocialQuant — Score History")
    print(f"{'='*60}\n")

    with open(SCORES_FILE) as f:
        scores = [json.loads(line) for line in f if line.strip()]

    print(f"  {'Date':<24} {'Model':<20} {'Brier':<8} {'Batch':<6}")
    print(f"  {'-'*60}")
    for s in scores:
        date = s["scored_at"][:19]
        model = s["model"][:18]
        brier = s["brier_score"]
        batch = s["batch_size"]
        print(f"  {date:<24} {model:<20} {brier:<8.4f} {batch:<6}")

    # Show overall average
    avg = sum(s["brier_score"] for s in scores) / len(scores)
    print(f"\n  Overall average Brier: {avg:.4f}")
    print(f"  Total batches: {len(scores)}")
    print(f"  Total predictions: {sum(s['batch_size'] for s in scores)}")


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    import sys

    ensure_dirs()

    if len(sys.argv) < 2:
        print("""
SocialQuant — Prediction engine for social media reach

Usage:
  python socialquant.py capture     — Capture real tweets and predict
  python socialquant.py resolve    — Resolve pending predictions and score
  python socialquant.py report     — Show score history
  python socialquant.py run        — Capture then resolve if anything is ready

Environment:
  X_BEARER_TOKEN      — X API v2 bearer token
  OPENAI_API_KEY      — OpenAI API key
  SOCIALQUANT_MODEL   — LLM model (default: gpt-4o-mini)
  SOCIALQUANT_TIMEFRAME — Hours to wait before resolving (default: 4)
  SOCIALQUANT_TARGET    — Target impressions (default: 10000)
  SOCIALQUANT_BATCH     — Tweets per capture batch (default: 20)
""")
        return

    cmd = sys.argv[1]

    if cmd == "capture":
        capture_phase()
    elif cmd == "resolve":
        resolve_phase()
    elif cmd == "report":
        report_phase()
    elif cmd == "run":
        capture_phase()
        resolve_phase()
        report_phase()
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: capture, resolve, report, run")


if __name__ == "__main__":
    main()
