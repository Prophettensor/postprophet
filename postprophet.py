"""
PostProphet — Prediction harness for social media reach.

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
OPENAI_MODEL = os.environ.get("POSTPROPHET_MODEL", "gpt-4o-mini")
TIMEFRAME_HOURS = int(os.environ.get("POSTPROPHET_TIMEFRAME", "24"))
TARGET_IMPRESSIONS = int(os.environ.get("POSTPROPHET_TARGET", "10000"))
BATCH_SIZE = int(os.environ.get("POSTPROPHET_BATCH", "20"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.jsonl")
SCORES_FILE = os.path.join(DATA_DIR, "scores.jsonl")
AUTHOR_CACHE_FILE = os.path.join(DATA_DIR, "author_cache.json")

CACHE_TTL_HOURS = 6

# ── X API helpers ───────────────────────────────────────────────────────


def x_headers():
    return {"Authorization": f"Bearer {X_BEARER_TOKEN}"}


def search_recent_tweets(query="AI -is:retweet", max_results=50):
    """Search recent tweets using X API v2."""
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": max(10, min(max_results, 100)),
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


def get_author_latest_tweet(author_id, max_results=10):
    """Fetch an author's most recent original tweets (not reply/retweet).
    Returns up to max_results tweets, newest first."""
    url = "https://api.twitter.com/2/users/{}/tweets".format(author_id)
    params = {
        "max_results": max(10, min(max_results, 100)),
        "tweet.fields": "public_metrics,created_at,in_reply_to_user_id,referenced_tweets",
        "exclude": "retweets,replies",
    }
    resp = httpx.get(url, headers=x_headers(), params=params, timeout=20)
    if resp.status_code != 200:
        return None
    data = resp.json()
    tweets = data.get("data", [])
    return tweets if tweets else None


def pick_fresh_tweet(tweets, max_age_hours=12, author_id=None):
    """From a list of tweets, return the first one that's:
    1. Under max_age_hours old
    2. Not a self-reply (thread continuation)
    Returns None if none qualify."""
    if not tweets:
        return None
    now = datetime.now(timezone.utc)
    for tweet in tweets:
        # Check age
        created_str = tweet.get("created_at", "")
        if not created_str:
            continue
        try:
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            age_hours = (now - created).total_seconds() / 3600
            if age_hours > max_age_hours:
                continue
        except Exception:
            continue

        # Check self-reply (thread continuation)
        in_reply_to = tweet.get("in_reply_to_user_id")
        if in_reply_to and author_id and str(in_reply_to) == str(author_id):
            continue

        return tweet
    return None


def get_user_by_username(username):
    """Look up a user by username, return user data with public metrics."""
    url = f"https://api.twitter.com/2/users/by/username/{username}"
    params = {"user.fields": "public_metrics,username"}
    resp = httpx.get(url, headers=x_headers(), params=params, timeout=15)
    if resp.status_code != 200:
        return None
    return resp.json().get("data", {})


def load_tracked_accounts():
    """Load tracked accounts from accounts.txt."""
    accounts_file = os.path.join(os.path.dirname(__file__), "accounts.txt")
    if not os.path.exists(accounts_file):
        return []
    with open(accounts_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def capture_from_accounts():
    """Capture latest tweets from tracked accounts and predict with dynamic targets."""
    print("\n" + "=" * 60)
    print("  PostProphet — Account Tracking Capture")
    print("=" * 60 + "\n")

    handles = load_tracked_accounts()
    if not handles:
        print("  No accounts in accounts.txt. Add handles first.")
        return []

    print(f"  Tracking {len(handles)} accounts...\n")

    # Try to get trending topics (optional)
    try:
        trending = get_trending_topics()
    except Exception:
        trending = []

    predictions = []
    for i, handle in enumerate(handles, 1):
        print(f"  [{i}/{len(handles)}] @{handle}...")

        # Check cache first — skip X API calls if fresh
        cached = get_cached_author(handle)
        if cached:
            print(f"    (cached)")
            followers = cached["followers"]
            if followers == 0:
                print(f"    ⚠️  0 followers — skipping")
                continue

            baseline = cached["baseline"]
            median_imp = baseline.get("median_impressions", 0)
            if median_imp == 0:
                print(f"    ⚠️  No baseline data — skipping")
                continue

            dynamic_target = int(baseline.get("p75_impressions", median_imp * 1.2))
            print(f"    {followers:,} followers | median {median_imp:.0f} imp/tweet | target (p75): {dynamic_target:,}")

            user_id = cached["user_id"]

            # Still need to fetch latest tweet (not cached — changes every run)
            tweets = get_author_latest_tweet(user_id)
            if not tweets:
                print(f"    ⚠️  No recent tweets — skipping")
                continue

            latest = pick_fresh_tweet(tweets, max_age_hours=12, author_id=user_id)
            if not latest:
                print(f"    ⏭️  No tweets under 12h — skipping")
                continue

            existing = load_predictions()
            already_predicted = any(p["tweet_id"] == latest["id"] for p in existing)
            if already_predicted:
                print(f"    ⏭️  Already predicted this tweet — skipping")
                continue

            # Use cached best/worst samples
            recent_samples = cached["recent_samples"]

            tweet_metrics = latest.get("public_metrics", {})
            context = {
                "tweet_id": latest["id"],
                "text": latest.get("text", ""),
                "created_at": latest.get("created_at", ""),
                "planned_post_time": latest.get("created_at", ""),
                "platform": "x",
                "author": {
                    "username": handle,
                    "followers": followers,
                    "following": cached["following"],
                    "tweet_count": cached["tweet_count"],
                },
                "author_baseline": baseline,
                "author_recent_top_tweets": recent_samples,
                "tweet_metrics_at_capture": {
                    "likes": tweet_metrics.get("like_count", 0),
                    "retweets": tweet_metrics.get("retweet_count", 0),
                    "replies": tweet_metrics.get("reply_count", 0),
                    "quotes": tweet_metrics.get("quote_count", 0),
                },
                "trending_topics": trending,
                "has_x_data": True,
            }

            print(f"    Tweet: {context['text'][:80]}...")

            try:
                prediction = predict(context, target=dynamic_target, timeframe=TIMEFRAME_HOURS)
            except Exception as e:
                print(f"    ❌ Prediction error: {e}")
                continue

            record = {
                "tweet_id": context["tweet_id"],
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "timeframe_hours": TIMEFRAME_HOURS,
                "target": dynamic_target,
                "context": context,
                "prediction": prediction,
                "resolve_after": (
                    datetime.fromisoformat(context["created_at"].replace("Z", "+00:00")) + timedelta(hours=TIMEFRAME_HOURS)
                ).isoformat(),
                "resolved": False,
                "actual_impressions": None,
            }

            save_prediction(record)
            predictions.append(record)

            prob = prediction.get("probability", 0.5)
            reasoning = prediction.get("reasoning", "")[:100]
            print(f"    → {prob:.0%} | target {dynamic_target:,} | {reasoning}")
            print()
            continue

        # ── Cold path: fetch from X API ──────────────────────────────

        # Look up user
        user = get_user_by_username(handle)
        if not user:
            print(f"    ❌ User not found")
            continue

        user_id = user.get("id", "")
        author_metrics = user.get("public_metrics", {})
        followers = author_metrics.get("followers_count", 0)

        if followers == 0:
            print(f"    ⚠️  0 followers — skipping")
            continue

        # Fetch recent tweets for baseline
        try:
            recent = get_author_recent_tweets(user_id)
        except Exception:
            recent = []

        baseline = compute_author_baseline(recent)
        median_imp = baseline.get("median_impressions", 0)

        if median_imp == 0:
            print(f"    ⚠️  No baseline data — skipping")
            continue

        # Dynamic target: p75 (will this tweet beat their 75th percentile?)
        dynamic_target = int(baseline.get("p75_impressions", median_imp * 1.2))
        print(f"    {followers:,} followers | median {median_imp:.0f} imp/tweet | target (p75): {dynamic_target:,}")

        # Get their latest tweet
        tweets = get_author_latest_tweet(user_id)
        if not tweets:
            print(f"    ⚠️  No recent tweets — skipping")
            continue

        latest = pick_fresh_tweet(tweets, max_age_hours=12, author_id=user_id)
        if not latest:
            print(f"    ⏭️  No tweets under 12h — skipping")
            continue

        # Check if we already predicted this tweet
        existing = load_predictions()
        already_predicted = any(p["tweet_id"] == latest["id"] for p in existing)
        if already_predicted:
            print(f"    ⏭️  Already predicted this tweet — skipping")
            continue

        # Build reference samples: tweet closest to p75 (hit) and p25 (miss)
        hit_sample, miss_sample = get_reference_tweets(recent, baseline)
        recent_samples = [s for s in [hit_sample, miss_sample] if s]

        # Cache this author's data for future runs
        cache_author(handle, user, baseline, recent_samples)

        # Build context
        tweet_metrics = latest.get("public_metrics", {})
        context = {
            "tweet_id": latest["id"],
            "text": latest.get("text", ""),
            "created_at": latest.get("created_at", ""),
            "planned_post_time": latest.get("created_at", ""),
            "platform": "x",
            "author": {
                "username": handle,
                "followers": followers,
                "following": author_metrics.get("following_count", 0),
                "tweet_count": author_metrics.get("tweet_count", 0),
            },
            "author_baseline": baseline,
            "author_recent_top_tweets": recent_samples,
            "tweet_metrics_at_capture": {
                "likes": tweet_metrics.get("like_count", 0),
                "retweets": tweet_metrics.get("retweet_count", 0),
                "replies": tweet_metrics.get("reply_count", 0),
                "quotes": tweet_metrics.get("quote_count", 0),
            },
            "trending_topics": trending,
            "has_x_data": True,
        }

        print(f"    Tweet: {context['text'][:80]}...")

        # Predict with dynamic target
        try:
            prediction = predict(context, target=dynamic_target, timeframe=TIMEFRAME_HOURS)
        except Exception as e:
            print(f"    ❌ Prediction error: {e}")
            continue

        record = {
            "tweet_id": context["tweet_id"],
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "timeframe_hours": TIMEFRAME_HOURS,
            "target": dynamic_target,
            "context": context,
            "prediction": prediction,
            "resolve_after": (
                datetime.fromisoformat(context["created_at"].replace("Z", "+00:00")) + timedelta(hours=TIMEFRAME_HOURS)
            ).isoformat(),
            "resolved": False,
            "actual_impressions": None,
        }

        save_prediction(record)
        predictions.append(record)

        prob = prediction.get("probability", 0.5)
        reasoning = prediction.get("reasoning", "")[:100]
        print(f"    → {prob:.0%} | target {dynamic_target:,} | {reasoning}")
        print()

    print(f"Captured {len(predictions)} new predictions. Waiting {TIMEFRAME_HOURS}h for resolution.")
    return predictions


def get_trending_topics():
    """Get trending topics (WOEID 1 = worldwide)."""
    url = "https://api.twitter.com/1.1/trends/place.json?id=1"
    resp = httpx.get(url, headers=x_headers(), timeout=15)
    if resp.status_code != 200:
        return []
    data = resp.json()
    trends = data[0].get("trends", []) if data else []
    return [t["name"] for t in trends[:10]]


def get_author_recent_tweets(author_id, max_results=10):
    """Fetch an author's recent tweets to compute engagement baseline.
    Filters out self-replies (thread continuations) which get artificially
    low impressions — the API's exclude=replies does NOT filter self-replies."""
    url = "https://api.twitter.com/2/users/{}/tweets".format(author_id)
    params = {
        "max_results": max(10, min(max_results, 100)),
        "tweet.fields": "public_metrics,created_at,in_reply_to_user_id,referenced_tweets",
        "exclude": "retweets,replies",
    }
    resp = httpx.get(url, headers=x_headers(), params=params, timeout=20)
    if resp.status_code != 200:
        return []
    data = resp.json()
    tweets = data.get("data", [])
    # Filter out self-replies (thread tails)
    return [
        t for t in tweets
        if not (
            t.get("in_reply_to_user_id")
            and str(t.get("in_reply_to_user_id")) == str(author_id)
        )
    ]


def compute_author_baseline(recent_tweets: list) -> dict:
    """Compute engagement baseline from an author's recent tweets.
    Uses median + percentiles instead of mean — resistant to viral outliers."""
    if not recent_tweets:
        return {"median_impressions": 0, "p25_impressions": 0, "p75_impressions": 0, "sample_size": 0}

    impressions = sorted([t.get("public_metrics", {}).get("impression_count", 0) for t in recent_tweets])
    likes = sorted([t.get("public_metrics", {}).get("like_count", 0) for t in recent_tweets])
    retweets = sorted([t.get("public_metrics", {}).get("retweet_count", 0) for t in recent_tweets])

    n = len(impressions)

    def percentile(data, p):
        """Linear interpolation percentile."""
        if n == 1:
            return data[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return data[f] + (data[c] - data[f]) * (k - f)

    return {
        "median_impressions": percentile(impressions, 0.50),
        "p25_impressions": percentile(impressions, 0.25),
        "p75_impressions": percentile(impressions, 0.75),
        "median_likes": percentile(likes, 0.50),
        "median_retweets": percentile(retweets, 0.50),
        "sample_size": n,
    }


def get_reference_tweets(recent_tweets: list, baseline: dict) -> tuple:
    """Find the tweets closest to p75 (a 'hit') and p25 (a 'miss') for context.
    Returns (hit_sample, miss_sample)."""
    if not recent_tweets or not baseline.get("p75_impressions"):
        return None, None

    p75 = baseline["p75_impressions"]
    p25 = baseline["p25_impressions"]

    hit_tweet = min(recent_tweets, key=lambda t: abs(t.get("public_metrics", {}).get("impression_count", 0) - p75))
    miss_tweet = min(recent_tweets, key=lambda t: abs(t.get("public_metrics", {}).get("impression_count", 0) - p25))

    def format_sample(t, tier):
        return {
            "text": t.get("text", "")[:100],
            "impressions": t.get("public_metrics", {}).get("impression_count", 0),
            "likes": t.get("public_metrics", {}).get("like_count", 0),
            "retweets": t.get("public_metrics", {}).get("retweet_count", 0),
            "tier": tier,
        }

    return format_sample(hit_tweet, "hit"), format_sample(miss_tweet, "miss")


# ── Author baseline cache ──────────────────────────────────────────────


def load_author_cache() -> dict:
    """Load the author cache. Returns {handle: {cached_at, data...}}."""
    if not os.path.exists(AUTHOR_CACHE_FILE):
        return {}
    try:
        with open(AUTHOR_CACHE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_author_cache(cache: dict):
    """Persist the author cache to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(AUTHOR_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def get_cached_author(handle: str) -> dict | None:
    """Return cached author data if fresh (< CACHE_TTL_HOURS old), else None."""
    cache = load_author_cache()
    entry = cache.get(handle)
    if not entry:
        return None
    cached_at = datetime.fromisoformat(entry["cached_at"])
    age = datetime.now(timezone.utc) - cached_at
    if age > timedelta(hours=CACHE_TTL_HOURS):
        return None
    return entry


def cache_author(handle: str, user_data: dict, baseline: dict, recent_samples: list):
    """Store author data in the cache with a timestamp."""
    cache = load_author_cache()
    cache[handle] = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_data.get("id", ""),
        "followers": user_data.get("public_metrics", {}).get("followers_count", 0),
        "following": user_data.get("public_metrics", {}).get("following_count", 0),
        "tweet_count": user_data.get("public_metrics", {}).get("tweet_count", 0),
        "baseline": baseline,
        "recent_samples": recent_samples,
    }
    save_author_cache(cache)


# ── Context gathering ────────────────────────────────────────────────────


def gather_context(tweet_data, includes, fetch_baseline=True):
    """Build the context object the LLM will use for prediction."""
    authors = {u["id"]: u for u in includes.get("users", [])}
    author = authors.get(tweet_data.get("author_id", ""), {})
    author_metrics = author.get("public_metrics", {})
    tweet_metrics = tweet_data.get("public_metrics", {})
    author_id = tweet_data.get("author_id", "")
    handle = author.get("username", "")

    # Compute engagement baseline from recent tweets (with cache)
    baseline = {"median_impressions": 0, "p25_impressions": 0, "p75_impressions": 0, "sample_size": 0}
    recent_samples = []
    if fetch_baseline and handle:
        cached = get_cached_author(handle)
        if cached:
            baseline = cached["baseline"]
            recent_samples = cached["recent_samples"]
        elif author_id:
            try:
                recent = get_author_recent_tweets(author_id)
                baseline = compute_author_baseline(recent)
                hit_sample, miss_sample = get_reference_tweets(recent, baseline)
                recent_samples = [s for s in [hit_sample, miss_sample] if s]
                # Cache for future calls
                if author:
                    cache_author(handle, author, baseline, recent_samples)
            except Exception:
                pass

    return {
        "tweet_id": tweet_data["id"],
        "text": tweet_data.get("text", ""),
        "created_at": tweet_data.get("created_at", ""),
        "planned_post_time": tweet_data.get("created_at", ""),  # In eval mode, this is when it was actually posted. In product mode, this is when it WILL be posted.
        "author": {
            "username": author.get("username", "unknown"),
            "followers": author_metrics.get("followers_count", 0),
            "following": author_metrics.get("following_count", 0),
            "tweet_count": author_metrics.get("tweet_count", 0),
        },
        "author_baseline": baseline,
        "author_recent_top_tweets": recent_samples,
        "tweet_metrics_at_capture": {
            "likes": tweet_metrics.get("like_count", 0),
            "retweets": tweet_metrics.get("retweet_count", 0),
            "replies": tweet_metrics.get("reply_count", 0),
            "quotes": tweet_metrics.get("quote_count", 0),
        },
        "trending_topics": [],
    }


# ── Prediction harness ───────────────────────────────────────────────────


PREDICTION_PROMPT = """You are a social media reach forecasting harness.

Given a tweet and its context, predict the probability (0.0 to 1.0) that this tweet will reach {target} impressions within {timeframe} hours of being posted.

The target is this author's 75th percentile — meaning their top 25%% of tweets reach this level. You're predicting whether this tweet is a "good one" for this author, not whether it goes viral.

Consider:
- Author's follower count AND engagement baseline (their median impressions per tweet)
- The spread between their p25 (typical miss) and p75 (typical hit) — a wide gap means erratic performance
- How this tweet compares to their reference "hit" tweet (does it have similar hooks, topics, or formats?)
- How this tweet compares to their reference "miss" tweet (does it share patterns with their flops?)
- Tweet content (hook quality, topic relevance, media, length)
- What's currently trending and whether the tweet relates
- Time of day the tweet was posted and whether that's a high-engagement window
- Time elapsed since posting (older tweets have less room to grow)

{llm_only_note}

Return JSON:
{{
  "probability": <float 0.0-1.0>,
  "point_estimate": <integer, your best guess at total impressions after {timeframe}h>,
  "reasoning": "<2-3 sentences. Reference specific data: which of their past tweets is this most similar to? What makes it better or worse?>",
  "suggestions": "<1 sentence on what specific change would improve the prediction>"
}}

Tweet: {text}
Author: @{username} ({followers} followers, {following} following)
Author baseline: median {median_impressions:.0f} impressions/tweet, p25 {p25_impressions:.0f}, p75 {p75_impressions:.0f}, median {median_likes:.0f} likes/tweet, median {median_retweets:.0f} retweets/tweet (sample: {baseline_sample} tweets)
Author's reference "hit" tweet (closest to p75):
{hit_tweet}
Author's reference "miss" tweet (closest to p25):
{miss_tweet}
Posted at: {posted_at}
Time elapsed since posting: {elapsed}
Timeframe: {timeframe}h
Target: {target} impressions (their p75)
Trending: {trending}"""


def predict(context: dict, target: int = None, timeframe: int = None) -> dict:
    """Call LLM to predict impression probability.

    The LLM never sees this tweet's engagement metrics (likes, retweets, replies,
    impressions). It predicts purely from: author identity + baseline (from their
    OTHER tweets) + tweet content + posting time + elapsed time + trending topics.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    tgt = target or TARGET_IMPRESSIONS
    tf = timeframe or TIMEFRAME_HOURS

    # Format recent top tweets for the prompt (these are the author's OTHER tweets,
    # used to establish their baseline — not the tweet being predicted)
    baseline = context.get("author_baseline", {})

    # Calculate elapsed time since posting
    posted_at_str = context.get("planned_post_time", context.get("created_at", ""))
    elapsed_str = "unknown"
    try:
        if posted_at_str:
            posted_at = datetime.fromisoformat(posted_at_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            elapsed_delta = now - posted_at
            hours = elapsed_delta.total_seconds() / 3600
            if hours < 1:
                elapsed_str = f"{int(elapsed_delta.total_seconds() / 60)} minutes"
            elif hours < 24:
                elapsed_str = f"{hours:.1f} hours"
            else:
                elapsed_str = f"{hours / 24:.1f} days"
    except Exception:
        pass

    # If no X API data, tell the LLM to infer from the username
    has_x = context.get("has_x_data", True)
    if not has_x:
        llm_only_note = "NOTE: No X API data available. You must infer the author's approximate follower count, engagement baseline, and audience size from your knowledge of this X account. The follower count shows as unknown because no API was called — do not assume 0. Use your training data knowledge of this account. If you truly don't recognize the account, assume 500-5,000 followers and a modest engagement baseline."
        followers_str = "unknown (infer from username)"
        following_str = "unknown"
        median_imp = 0
        p25_imp = 0
        p75_imp = 0
        median_likes = 0
        median_retweets = 0
        baseline_sample = 0
        hit_str = "  (no data — infer from knowledge)"
        miss_str = "  (no data — infer from knowledge)"
    else:
        llm_only_note = ""
        followers_str = context["author"]["followers"]
        following_str = context["author"]["following"]
        median_imp = baseline.get("median_impressions", 0)
        p25_imp = baseline.get("p25_impressions", 0)
        p75_imp = baseline.get("p75_impressions", 0)
        median_likes = baseline.get("median_likes", 0)
        median_retweets = baseline.get("median_retweets", 0)
        baseline_sample = baseline.get("sample_size", 0)
        
        hit_lines = []
        miss_lines = []
        for t in context.get("author_recent_top_tweets", []):
            line = f"  - \"{t['text']}\" → {t['impressions']} impressions, {t['likes']} likes"
            if t.get("tier") == "hit":
                hit_lines.append(line)
            elif t.get("tier") == "miss":
                miss_lines.append(line)
        hit_str = "\n".join(hit_lines) if hit_lines else "  (no data)"
        miss_str = "\n".join(miss_lines) if miss_lines else "  (no data)"

    prompt = PREDICTION_PROMPT.format(
        target=tgt,
        timeframe=tf,
        text=context["text"],
        username=context["author"]["username"],
        followers=followers_str,
        following=following_str,
        median_impressions=median_imp,
        p25_impressions=p25_imp,
        p75_impressions=p75_imp,
        median_likes=median_likes,
        median_retweets=median_retweets,
        baseline_sample=baseline_sample,
        hit_tweet=hit_str,
        miss_tweet=miss_str,
        posted_at=posted_at_str,
        elapsed=elapsed_str,
        trending=", ".join(context.get("trending_topics", [])) or "none available",
        llm_only_note=llm_only_note,
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a social media reach prediction harness. You predict impression probability from content, author baseline, and timing only. You never see engagement metrics for the tweet being predicted. Output only valid JSON."},
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
    print("\n" + "=" * 60)
    print("  PostProphet — Capture Phase")
    print(f"  Target: {TARGET_IMPRESSIONS} impressions in {TIMEFRAME_HOURS}h")
    print(f"  Batch: {BATCH_SIZE} tweets")
    print("=" * 60 + "\n")

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
        print(f"  [{i}/{len(tweets)}] Gathering context for tweet {tweet['id']}...")

        context = gather_context(tweet, includes, fetch_baseline=True)
        context["trending_topics"] = trending

        baseline = context.get("author_baseline", {})
        print(f"    @{context['author']['username']} ({context['author']['followers']} followers, median {baseline.get('median_impressions', 0):.0f} imp/tweet)")
        print(f"    Tweet: {context['text'][:80]}...")

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
                datetime.fromisoformat(context["created_at"].replace("Z", "+00:00")) + timedelta(hours=TIMEFRAME_HOURS)
            ).isoformat(),
            "resolved": False,
            "actual_impressions": None,
        }

        save_prediction(record)
        predictions.append(record)

        prob = prediction.get("probability", 0.5)
        reasoning = prediction.get("reasoning", "")[:100]
        print(f"    → {prob:.0%} | {reasoning}")
        print()

    print(f"Captured {len(predictions)} predictions. Waiting {TIMEFRAME_HOURS}h for resolution.")
    return predictions


def resolve_phase():
    """Phase 2: Check predictions whose timeframe has elapsed, score them."""
    print("\n" + "=" * 60)
    print("  PostProphet — Resolve Phase")
    print("=" * 60 + "\n")

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
        username = p["context"]["author"]["username"]
        baseline = p["context"].get("author_baseline", {}).get("median_impressions", 0)
        print(f"  @{username}: predicted {prob:.0%}, got {actual:,} (median: {baseline:.0f}) → {'HIT' if hit else 'MISS'}")

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
    """Phase 3: Show summary of all scores + detailed predictions with reasoning."""
    print("\n" + "=" * 60)
    print("  PostProphet — Score History")
    print("=" * 60 + "\n")

    if not os.path.exists(SCORES_FILE):
        print("  No scores yet. Run 'track' then 'resolve' first.")
        return

    with open(SCORES_FILE) as f:
        scores = [json.loads(line) for line in f if line.strip()]

    print(f"  {'Date':<24} {'Model':<20} {'Brier':<8} {'Batch':<6}")
    print(f"  {'-' * 60}")
    for s in scores:
        date = s["scored_at"][:19]
        model = s["model"][:18]
        brier = s["brier_score"]
        batch = s["batch_size"]
        print(f"  {date:<24} {model:<20} {brier:<8.4f} {batch:<6}")

    # Overall stats
    all_briers = [s["brier_score"] for s in scores]
    avg = sum(all_briers) / len(all_briers)
    best = min(all_briers)
    worst = max(all_briers)
    total_preds = sum(s["batch_size"] for s in scores)
    print(f"\n  Overall average Brier: {avg:.4f}")
    print(f"  Best Brier:  {best:.4f}")
    print(f"  Worst Brier: {worst:.4f}")
    print(f"  Total batches: {len(scores)}")
    print(f"  Total predictions: {total_preds}")

    # Show detailed predictions from the most recent batch
    predictions = load_predictions()
    resolved = [p for p in predictions if p.get("resolved")]

    if resolved:
        print("\n" + "=" * 60)
        print("  Recent Predictions (with reasoning)")
        print("=" * 60 + "\n")

        # Show last 10 resolved predictions
        for p in resolved[-10:]:
            username = p["context"]["author"]["username"]
            followers = p["context"]["author"].get("followers", 0)
            baseline = p["context"].get("author_baseline", {})
            avg_imp = baseline.get("median_impressions", 0)
            prob = p["prediction"].get("probability", 0)
            point = p["prediction"].get("point_estimate", "—")
            actual = p.get("actual_impressions", 0)
            target = p.get("target", 0)
            hit = actual >= target
            reasoning = p["prediction"].get("reasoning", "")
            suggestions = p["prediction"].get("suggestions", "")
            tweet_text = p["context"].get("text", "")[:70]
            elapsed = ""
            created = p["context"].get("created_at", "")
            predicted = p.get("predicted_at", "")
            try:
                if created and predicted:
                    c = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    pr = datetime.fromisoformat(predicted.replace("Z", "+00:00"))
                    delta = pr - c
                    hrs = delta.total_seconds() / 3600
                    elapsed = f"{hrs:.1f}h old at prediction"
            except Exception:
                pass

            print(f"  @{username} ({followers:,} followers, avg {avg_imp:.0f} imp/tweet)")
            print(f"  Tweet: {tweet_text}...")
            print(f"  Target: {target:,} (1.2x baseline) | Predicted: {prob:.0%} | Estimate: {point:,} | Actual: {actual:,} | {'✅ HIT' if hit else '❌ MISS'}")
            if elapsed:
                print(f"  Tweet age at prediction: {elapsed}")
            print(f"  Reasoning: {reasoning}")
            if suggestions:
                print(f"  Suggestion: {suggestions}")
            print()


# ── Predict for an unpublished tweet (product mode) ─────────────────────


def predict_tweet(
    text: str,
    author_username: str,
    target: int = None,
    timeframe_hours: int = None,
    planned_post_time: str = None,
    platform: str = "x",
) -> dict:
    """Predict reach for an unpublished tweet (product mode).

    If X_BEARER_TOKEN is set, fetches real author data.
    If not, falls back to LLM-only mode — the model reasons from the
    username and tweet text alone (less accurate but works without X API).

    Args:
        text: The tweet text
        author_username: X handle (without @)
        target: Target impressions (default from config)
        timeframe_hours: Hours after posting to measure (default from config)
        planned_post_time: ISO datetime when tweet will be posted (default: now)
        platform: Platform identifier (default: "x" — only X supported)

    Returns:
        {probability, reasoning, suggestions, context_summary}
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    tgt = target or TARGET_IMPRESSIONS
    tf = timeframe_hours or TIMEFRAME_HOURS
    post_time = planned_post_time or datetime.now(timezone.utc).isoformat()

    # Try to fetch author info from X API
    # If no bearer token or API fails, fall back to LLM-only mode
    author_metrics = {"followers_count": 0, "following_count": 0, "tweet_count": 0}
    baseline = {"median_impressions": 0, "p25_impressions": 0, "p75_impressions": 0, "sample_size": 0}
    recent_samples = []
    trending = []
    has_x_data = False

    if X_BEARER_TOKEN:
        # Check cache first
        cached = get_cached_author(author_username)
        if cached:
            author_metrics = {
                "followers_count": cached["followers"],
                "following_count": cached["following"],
                "tweet_count": cached["tweet_count"],
            }
            baseline = cached["baseline"]
            recent_samples = cached["recent_samples"]
            has_x_data = True
        else:
            try:
                url = f"https://api.twitter.com/2/users/by/username/{author_username}"
                resp = httpx.get(url, headers=x_headers(), params={"user.fields": "public_metrics,username"}, timeout=15)
                resp.raise_for_status()
                user_data = resp.json().get("data", {})
                user_id = user_data.get("id", "")
                author_metrics = user_data.get("public_metrics", {})

                # Fetch author baseline
                if user_id:
                    try:
                        recent = get_author_recent_tweets(user_id)
                        baseline = compute_author_baseline(recent)
                        hit_sample, miss_sample = get_reference_tweets(recent, baseline)
                        recent_samples = [s for s in [hit_sample, miss_sample] if s]
                    except Exception:
                        pass

                # Cache for future calls
                cache_author(author_username, user_data, baseline, recent_samples)

            except Exception as e:
                print(f"  (X API unavailable: {e}. Using LLM-only mode.)")

        # Fetch trending (not cached — changes frequently)
        try:
            trending = get_trending_topics()
        except Exception:
            pass

    context = {
        "text": text,
        "planned_post_time": post_time,
        "created_at": post_time,
        "platform": platform,
        "author": {
            "username": author_username,
            "followers": author_metrics.get("followers_count", 0),
            "following": author_metrics.get("following_count", 0),
            "tweet_count": author_metrics.get("tweet_count", 0),
        },
        "author_baseline": baseline,
        "author_recent_top_tweets": recent_samples,
        "tweet_metrics_at_capture": {"likes": 0, "retweets": 0, "replies": 0},
        "trending_topics": trending,
        "has_x_data": has_x_data,
    }

    prediction = predict(context, target=tgt, timeframe=tf)

    # Include a context summary so the caller understands what was considered
    prediction["context_summary"] = {
        "who": f"@{author_username}" + (f" ({author_metrics.get('followers_count', 0):,} followers)" if has_x_data else " (no X API data — LLM inferred)"),
        "what": text[:100],
        "when": post_time,
        "where": platform,
        "target": f"{tgt} impressions in {tf}h after posting",
        "author_baseline": baseline,
        "trending_at_prediction_time": trending[:5] if trending else [],
        "x_api_used": has_x_data,
    }

    return prediction


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    import sys

    ensure_dirs()

    if len(sys.argv) < 2:
        print("""
PostProphet — Prediction harness for social media reach

Usage:
  python postprophet.py capture          — Capture real tweets via keyword search and predict
  python postprophet.py track            — Capture latest tweets from tracked accounts (accounts.txt)
  python postprophet.py resolve          — Resolve pending predictions and score
  python postprophet.py report           — Show score history
  python postprophet.py run              — Capture → resolve → report
  python postprophet.py predict <tweet>  — Predict reach for an unpublished tweet
                                           (requires --author, optional: --target, --timeframe, --post-time)

Environment:
  X_BEARER_TOKEN          — X API v2 bearer token
  OPENAI_API_KEY          — OpenAI API key
  POSTPROPHET_MODEL       — LLM model (default: gpt-4o-mini)
  POSTPROPHET_TIMEFRAME   — Hours to wait before resolving (default: 1)
  POSTPROPHET_TARGET      — Target impressions (default: 10000, ignored in track mode)
  POSTPROPHET_BATCH       — Tweets per capture batch (default: 20)
""")
        return

    cmd = sys.argv[1]

    if cmd == "capture":
        capture_phase()
    elif cmd == "track":
        capture_from_accounts()
    elif cmd == "resolve":
        resolve_phase()
    elif cmd == "report":
        report_phase()
    elif cmd == "run":
        capture_phase()
        resolve_phase()
        report_phase()
    elif cmd == "predict":
        # Product mode: predict for an unpublished tweet
        if len(sys.argv) < 3:
            print("Usage: python postprophet.py predict <tweet_text> --author <username> [--target N] [--timeframe H] [--post-time ISO]")
            return

        # Parse args
        tweet_text = ""
        author = ""
        target = None
        timeframe = None
        post_time = None

        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--author" and i + 1 < len(args):
                author = args[i + 1].lstrip("@")
                i += 2
            elif args[i] == "--target" and i + 1 < len(args):
                target = int(args[i + 1])
                i += 2
            elif args[i] == "--timeframe" and i + 1 < len(args):
                timeframe = int(args[i + 1])
                i += 2
            elif args[i] == "--post-time" and i + 1 < len(args):
                post_time = args[i + 1]
                i += 2
            else:
                if tweet_text:
                    tweet_text += " " + args[i]
                else:
                    tweet_text = args[i]
                i += 1

        if not author:
            print("Error: --author is required for predict mode")
            return

        print(f"\nPredicting reach for @{author}...")
        print(f"Tweet: {tweet_text[:100]}...")
        print()

        result = predict_tweet(
            text=tweet_text,
            author_username=author,
            target=target,
            timeframe_hours=timeframe,
            planned_post_time=post_time,
        )

        print(f"Probability: {result.get('probability', 0):.0%}")
        print(f"Reasoning: {result.get('reasoning', '')}")
        print(f"Suggestions: {result.get('suggestions', '')}")
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: capture, resolve, report, run, predict")


if __name__ == "__main__":
    main()
