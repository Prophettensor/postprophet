"""
PostProphet — Prediction harness for social media reach.

A harness that captures real tweets, gathers context, asks an LLM to predict
impression probability, waits for resolution, and scores with Brier score.
"""

import json
import os
import re
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
    Returns up to max_results tweets, newest first. Includes media + engagement context."""
    url = "https://api.twitter.com/2/users/{}/tweets".format(author_id)
    params = {
        "max_results": max(10, min(max_results, 100)),
        "tweet.fields": "public_metrics,created_at,in_reply_to_user_id,referenced_tweets,attachments,entities",
        "exclude": "retweets,replies",
        "expansions": "attachments.media_keys",
        "media.fields": "type,url,preview_image_url",
    }
    resp = httpx.get(url, headers=x_headers(), params=params, timeout=20)
    if resp.status_code != 200:
        return None
    data = resp.json()
    tweets = data.get("data", [])
    media_map = {m["media_key"]: m for m in data.get("includes", {}).get("media", [])}
    
    # Attach media info to each tweet
    for tweet in tweets:
        attachments = tweet.get("attachments", {})
        media_keys = attachments.get("media_keys", []) if attachments else []
        tweet["_media"] = [media_map[k] for k in media_keys if k in media_map]
    
    return tweets if tweets else None


def pick_fresh_tweet(tweets, max_age_hours=12, author_id=None):
    """From a list of tweets, return the first one that's:
    1. Under max_age_hours old
    2. Not a self-reply (thread continuation)
    3. Not a quote tweet (can't see quoted content yet)
    4. Not a thread reply (need full thread context)
    5. Not link-only with X-internal URL (can't evaluate)
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

        # Skip quote tweets — can't see quoted content yet (future feature)
        refs = tweet.get("referenced_tweets", [])
        if any(r.get("type") == "quoted" for r in refs):
            continue

        # Skip thread replies — need full thread context (future feature)
        if any(r.get("type") == "replied_to" for r in refs):
            continue

        # Skip link-only tweets with X-internal URLs
        clean = re.sub(r'https?://\S+', '', tweet.get("text", "")).strip()
        if len(clean) < 15:
            urls = tweet.get("entities", {}).get("urls", [])
            if urls and all(
                any(d in u.get("expanded_url", "") for d in ["x.com", "twitter.com"])
                for u in urls
            ):
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


def add_accounts(handles: list[str]):
    """Add accounts to accounts.txt and fetch/cache their data immediately.
    Skips handles already in the list."""
    accounts_file = os.path.join(os.path.dirname(__file__), "accounts.txt")
    existing = set(load_tracked_accounts())
    
    new_handles = []
    for handle in handles:
        handle = handle.strip().lstrip("@")
        if not handle or handle in existing:
            continue
        new_handles.append(handle)
        existing.add(handle)
    
    if not new_handles:
        print("  All accounts already tracked.")
        return
    
    # Append to accounts.txt
    with open(accounts_file, "a") as f:
        for handle in new_handles:
            f.write(handle + "\n")
    
    print(f"  Added {len(new_handles)} account(s): {', '.join(new_handles)}")
    
    # Fetch and cache each new account's data
    if not X_BEARER_TOKEN:
        print("  ⚠️  No X_BEARER_TOKEN set. Run 'track' later to fetch data.")
        return
    
    for handle in new_handles:
        print(f"  Fetching @{handle}...")
        user = get_user_by_username(handle)
        if not user:
            print(f"    ❌ User not found")
            continue
        user_id = user.get("id", "")
        followers = user.get("public_metrics", {}).get("followers_count", 0)
        if followers == 0:
            print(f"    ⚠️  0 followers")
            continue
        try:
            recent = get_author_recent_tweets(user_id)
            baseline = compute_author_baseline(recent)
            all_samples, hit_sample, miss_sample = get_reference_tweets(recent, baseline)
            recent_samples = [s for s in [hit_sample, miss_sample] if s]
            cache_author(handle, user, baseline, recent_samples)
            median_imp = baseline.get("median_impressions", 0)
            print(f"    ✓ {followers:,} followers | median {median_imp:.0f} imp/tweet")
        except Exception as e:
            print(f"    ❌ Error: {e}")


def capture_from_accounts():
    """Capture latest tweets from tracked accounts and predict with dynamic targets.
    Two passes: first fetch all author data + compute ecosystem baseline, then predict."""
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

    # ── Pass 1: Fetch all author data ─────────────────────────────
    print("  Phase 1: Fetching author data...\n")
    all_author_data = []
    author_contexts = []  # (handle, latest_tweet, context_dict, dynamic_target)
    existing = load_predictions()
    tracked_author_ids = set()  # author IDs from our accounts.txt — used to filter replies/quotes
    author_id_to_handle = {}  # map for displaying reply/quote authors

    for i, handle in enumerate(handles, 1):
        print(f"  [{i}/{len(handles)}] @{handle}...", end="")

        # Check cache first
        cached = get_cached_author(handle)
        if cached:
            followers = cached["followers"]
            if followers == 0:
                print(" ⚠️ 0 followers — skip")
                continue
            baseline = cached["baseline"]
            median_imp = baseline.get("median_impressions", 0)
            if median_imp == 0:
                print(" ⚠️ no baseline — skip")
                continue
            user_id = cached["user_id"]
            recent_samples = cached["recent_samples"]
            all_tweet_samples = []
            # For ecosystem, use cached recent_samples (best/worst only)
            recent_for_eco = []
        else:
            # Cold fetch
            user = get_user_by_username(handle)
            if not user:
                print(" ❌ not found")
                continue
            user_id = user.get("id", "")
            author_metrics = user.get("public_metrics", {})
            followers = author_metrics.get("followers_count", 0)
            if followers == 0:
                print(" ⚠️ 0 followers — skip")
                continue
            try:
                recent = get_author_recent_tweets(user_id)
            except Exception:
                recent = []
            baseline = compute_author_baseline(recent)
            median_imp = baseline.get("median_impressions", 0)
            if median_imp == 0:
                print(" ⚠️ no baseline — skip")
                continue
            all_samples, hit_sample, miss_sample = get_reference_tweets(recent, baseline)
            recent_samples = [s for s in [hit_sample, miss_sample] if s]
            all_tweet_samples = all_samples
            recent_for_eco = recent
            cache_author(handle, user, baseline, recent_samples)

        # Track this author ID for reply/quote filtering
        tracked_author_ids.add(user_id)
        author_id_to_handle[user_id] = handle

        # Fetch latest tweet
        tweets_list = get_author_latest_tweet(user_id)
        if not tweets_list:
            print(" ⚠️ no tweets")
            continue
        latest = pick_fresh_tweet(tweets_list, max_age_hours=12, author_id=user_id)
        if not latest:
            print(" ⏭️ none fresh")
            continue
        if any(p["tweet_id"] == latest["id"] for p in existing):
            print(" ⏭️ already predicted")
            continue

        dynamic_target = int(median_imp * 2)
        print(f" ✓ median {median_imp:.0f} | target {dynamic_target:,}")

        # Collect for ecosystem baseline
        # Tag author username on recent tweets for quote search
        for rt in recent_for_eco:
            rt["_author_username"] = handle
        all_author_data.append({
            "handle": handle,
            "followers": followers,
            "median_impressions": median_imp,
            "recent_tweets": recent_for_eco if recent_for_eco else [],
            "user_id": user_id,
        })

        # Store context for pass 2
        # NOTE: tweet_metrics and engagement on the PREDICTION tweet are hidden
        # from the LLM — we're simulating "hasn't been posted yet."
        # Engagement on OTHER tweets (author's history, ecosystem) is fair game.
        media_info = analyze_media(latest) if latest.get("_media") else {"has_media": False, "media_types": [], "descriptions": []}
        context = {
            "tweet_id": latest["id"],
            "text": latest.get("text", ""),
            "created_at": latest.get("created_at", ""),
            "planned_post_time": latest.get("created_at", ""),
            "platform": "x",
            "author": {
                "username": handle,
                "followers": followers,
                "following": cached["following"] if cached else author_metrics.get("following_count", 0),
                "tweet_count": cached["tweet_count"] if cached else author_metrics.get("tweet_count", 0),
            },
            "author_baseline": baseline,
            "author_recent_top_tweets": recent_samples,
            "author_all_tweets": all_tweet_samples,
            "media": media_info,
            "entities": latest.get("entities", {}),
            "referenced_tweets": latest.get("referenced_tweets", []),
            "trending_topics": trending,
            "has_x_data": True,
        }
        author_contexts.append((handle, latest, context, dynamic_target))

    # ── Compute ecosystem baseline ────────────────────────────────
    print(f"\n  Phase 2: Computing ecosystem baseline from {len(all_author_data)} accounts...")
    ecosystem = compute_ecosystem_baseline(all_author_data, tracked_author_ids)
    if ecosystem:
        print(f"    Ecosystem median: {ecosystem['median_impressions']:.0f} imp/tweet")
        print(f"    Median efficiency: {ecosystem['median_imp_per_1k_followers']:.1f} imp per 1K followers")
        if ecosystem.get("best_tweets_normalized"):
            best = ecosystem["best_tweets_normalized"][0]
            print(f"    Best normalized tweet: @{best['handle']} ({best['imp_per_1k']:.1f} imp/1K followers)")

    # ── Pass 2: Predict ───────────────────────────────────────────
    print(f"\n  Phase 3: Predicting {len(author_contexts)} tweets...\n")
    predictions = []
    for handle, latest, context, dynamic_target in author_contexts:
        # Add ecosystem context
        context["ecosystem"] = ecosystem

        print(f"  @{handle}: {context['text'][:60]}...")

        try:
            prediction = predict(context, target=dynamic_target, timeframe=TIMEFRAME_HOURS)
        except Exception as e:
            print(f"    ❌ Prediction error: {e}")
            continue

        # Calculate elapsed time at prediction
        elapsed_str = "unknown"
        try:
            posted_at = datetime.fromisoformat(context["created_at"].replace("Z", "+00:00"))
            elapsed_hrs = (datetime.now(timezone.utc) - posted_at).total_seconds() / 3600
            elapsed_str = f"{elapsed_hrs:.1f}h"
        except Exception:
            pass

        record = {
            "tweet_id": context["tweet_id"],
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "timeframe_hours": TIMEFRAME_HOURS,
            "target": dynamic_target,
            "model": OPENAI_MODEL,
            "harness_version": HARNESS_VERSION,
            "harness_hash": HARNESS_HASH,
            "followers_at_prediction": context["author"]["followers"],
            "elapsed_at_prediction": elapsed_str,
            "has_url": "http" in context["text"] or "https" in context["text"],
            "has_external_url": any(
                u.get("expanded_url", "") and not any(d in u["expanded_url"] for d in ["x.com", "twitter.com"])
                for u in context.get("entities", {}).get("urls", [])
            ) if context.get("entities") else ("http" in context["text"] or "https" in context["text"]),
            "is_quote_tweet": any(
                r.get("type") == "quoted" for r in context.get("referenced_tweets", [])
            ) if context.get("referenced_tweets") else False,
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
        verdict = "YES" if prob >= 0.5 else "NO"
        print(f"    → {verdict} ({prob:.0%} chance of {dynamic_target:,} impressions) | {reasoning}")
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
    low impressions — the API's exclude=replies does NOT filter self-replies.
    Includes media data for each tweet."""
    url = "https://api.twitter.com/2/users/{}/tweets".format(author_id)
    params = {
        "max_results": max(10, min(max_results, 100)),
        "tweet.fields": "public_metrics,created_at,in_reply_to_user_id,referenced_tweets,attachments,entities",
        "exclude": "retweets,replies",
        "expansions": "attachments.media_keys",
        "media.fields": "type,url,preview_image_url",
    }
    resp = httpx.get(url, headers=x_headers(), params=params, timeout=20)
    if resp.status_code != 200:
        return []
    data = resp.json()
    tweets = data.get("data", [])
    media_map = {m["media_key"]: m for m in data.get("includes", {}).get("media", [])}
    for tweet in tweets:
        attachments = tweet.get("attachments", {})
        media_keys = attachments.get("media_keys", []) if attachments else []
        tweet["_media"] = [media_map[k] for k in media_keys if k in media_map]
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
    Uses median — resistant to viral outliers."""
    if not recent_tweets:
        return {"median_impressions": 0, "median_likes": 0, "median_retweets": 0, "sample_size": 0}

    impressions = sorted([t.get("public_metrics", {}).get("impression_count", 0) for t in recent_tweets])
    likes = sorted([t.get("public_metrics", {}).get("like_count", 0) for t in recent_tweets])
    retweets = sorted([t.get("public_metrics", {}).get("retweet_count", 0) for t in recent_tweets])

    n = len(impressions)

    def percentile(data, p):
        if n == 1:
            return data[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return data[f] + (data[c] - data[f]) * (k - f)

    return {
        "median_impressions": percentile(impressions, 0.50),
        "median_likes": percentile(likes, 0.50),
        "median_retweets": percentile(retweets, 0.50),
        "sample_size": n,
    }


# ── Media analysis ──────────────────────────────────────────────────────


def analyze_media(tweet: dict) -> dict:
    """Analyze media attached to a tweet.
    Returns {has_media, media_types, descriptions} — uses gpt-4o-mini vision
    to describe images so the model can assess media quality/relevance."""
    media_list = tweet.get("_media", [])
    if not media_list:
        return {"has_media": False, "media_types": [], "descriptions": []}
    
    media_types = [m.get("type", "unknown") for m in media_list]
    descriptions = []
    
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    
    for m in media_list:
        mtype = m.get("type", "unknown")
        if mtype in ("photo", "animated_gif"):
            url = m.get("url") or m.get("preview_image_url")
            if url and client:
                try:
                    resp = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe this image in one sentence. Is it: a chart/graph, screenshot, photo, meme, infographic, or graphic? Does it look professional or amateur? Is it relevant to a tweet about crypto/AI?"},
                                    {"type": "image_url", "image_url": {"url": url}},
                                ],
                            }
                        ],
                        max_tokens=80,
                        temperature=0.2,
                    )
                    descriptions.append(resp.choices[0].message.content.strip())
                except Exception:
                    descriptions.append(f"{mtype} (analysis failed)")
            else:
                descriptions.append(f"{mtype} (no URL)")
        elif mtype == "video":
            descriptions.append("video (not analyzed)")
        else:
            descriptions.append(f"{mtype}")
    
    return {"has_media": True, "media_types": media_types, "descriptions": descriptions}


def get_engagement_context(tweet: dict, tracked_author_ids: set = None) -> dict:
    """Extract engagement signals from a tweet's public metrics and context.
    
    If tracked_author_ids is provided, fetches reply/quote text but only keeps
    replies/quotes from tracked accounts — filters out spam/bots automatically.
    
    Returns info about replies, quotes, and bookmark counts — plus the actual
    text of replies/quotes from tracked accounts (high-signal engagement)."""
    metrics = tweet.get("public_metrics", {})
    
    reply_count = metrics.get("reply_count", 0)
    quote_count = metrics.get("quote_count", 0)
    bookmark_count = metrics.get("bookmark_count", 0)
    
    # Check if this tweet quotes another (referenced_tweets with type 'quoted')
    refs = tweet.get("referenced_tweets", [])
    quoted_tweet = next((r for r in refs if r.get("type") == "quoted"), None)
    
    # Fetch reply/quote text from tracked accounts only (if requested)
    tracked_replies = []
    tracked_quotes = []
    if tracked_author_ids and X_BEARER_TOKEN:
        tweet_id = tweet.get("id", "")
        if tweet_id:
            # Fetch replies via conversation_id
            try:
                url = "https://api.twitter.com/2/tweets/search/recent"
                params = {
                    "query": f"conversation_id:{tweet_id}",
                    "max_results": 100,
                    "tweet.fields": "public_metrics,created_at,author_id",
                    "expansions": "author_id",
                    "user.fields": "public_metrics,username",
                }
                resp = httpx.get(url, headers=x_headers(), params=params, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
                    for reply in data.get("data", []):
                        author_id = reply.get("author_id", "")
                        if author_id in tracked_author_ids:
                            user = users.get(author_id, {})
                            tracked_replies.append({
                                "handle": user.get("username", "unknown"),
                                "followers": user.get("public_metrics", {}).get("followers_count", 0),
                                "text": reply.get("text", "")[:120],
                                "created_at": reply.get("created_at", ""),
                            })
            except Exception:
                pass
            
            # Fetch quote tweets via URL search
            # Get the author's username for the URL
            author_username = tweet.get("_author_username", "")
            if author_username:
                try:
                    url = "https://api.twitter.com/2/tweets/search/recent"
                    params = {
                        "query": f'url:"x.com/{author_username}/status/{tweet_id}" OR url:"twitter.com/{author_username}/status/{tweet_id}"',
                        "max_results": 100,
                        "tweet.fields": "public_metrics,created_at,author_id",
                        "expansions": "author_id",
                        "user.fields": "public_metrics,username",
                    }
                    resp = httpx.get(url, headers=x_headers(), params=params, timeout=20)
                    if resp.status_code == 200:
                        data = resp.json()
                        users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
                        for quote in data.get("data", []):
                            # Skip the original tweet itself
                            if quote.get("id") == tweet_id:
                                continue
                            author_id = quote.get("author_id", "")
                            if author_id in tracked_author_ids:
                                user = users.get(author_id, {})
                                tracked_quotes.append({
                                    "handle": user.get("username", "unknown"),
                                    "followers": user.get("public_metrics", {}).get("followers_count", 0),
                                    "text": quote.get("text", "")[:120],
                                    "created_at": quote.get("created_at", ""),
                                })
                except Exception:
                    pass
    
    return {
        "reply_count": reply_count,
        "quote_count": quote_count,
        "bookmark_count": bookmark_count,
        "is_quote_tweet": quoted_tweet is not None,
        "quoted_tweet_id": quoted_tweet.get("id") if quoted_tweet else None,
        "tracked_replies": tracked_replies,
        "tracked_quotes": tracked_quotes,
    }




def compute_ecosystem_baseline(all_author_data: list, tracked_author_ids: set = None) -> dict:
    """Compute ecosystem-wide baseline from all tracked accounts.
    
    Normalizes impressions by follower count so we compare tweet quality,
    not account size. A tweet from a 100-follower account that gets 1K
    impressions is performing better than a 100K-follower account getting 5K.
    
    If tracked_author_ids is provided, fetches tracked replies/quotes on
    ecosystem tweets — high-signal engagement from real accounts, no spam.
    
    all_author_data: list of {handle, followers, median_impressions, recent_tweets, user_id}
    """
    if not all_author_data:
        return {}
    
    # Impressions per 1K followers — normalizes for account size
    imp_per_1k = []
    median_imps = []
    best_tweets = []
    worst_tweets = []
    
    for author in all_author_data:
        followers = author.get("followers", 0)
        median_imp = author.get("median_impressions", 0)
        if followers > 0 and median_imp > 0:
            imp_per_1k.append(median_imp / (followers / 1000))
            median_imps.append(median_imp)
        
        # Collect best/worst tweets (already sorted)
        recent = author.get("recent_tweets", [])
        if recent:
            sorted_tweets = sorted(recent, key=lambda t: t.get("public_metrics", {}).get("impression_count", 0), reverse=True)
            best_tweets.append({
                "text": sorted_tweets[0].get("text", "")[:120],
                "impressions": sorted_tweets[0].get("public_metrics", {}).get("impression_count", 0),
                "followers": followers,
                "handle": author.get("handle", ""),
                "imp_per_1k": (sorted_tweets[0].get("public_metrics", {}).get("impression_count", 0) / (followers / 1000)) if followers > 0 else 0,
                "posted_at": sorted_tweets[0].get("created_at", ""),
            })
            worst_tweets.append({
                "text": sorted_tweets[-1].get("text", "")[:120],
                "impressions": sorted_tweets[-1].get("public_metrics", {}).get("impression_count", 0),
                "followers": followers,
                "handle": author.get("handle", ""),
                "imp_per_1k": (sorted_tweets[-1].get("public_metrics", {}).get("impression_count", 0) / (followers / 1000)) if followers > 0 else 0,
                "posted_at": sorted_tweets[-1].get("created_at", ""),
            })
    
    if not imp_per_1k:
        return {}
    
    imp_per_1k.sort()
    median_imps.sort()
    
    def pct(data, p):
        n = len(data)
        if n == 1:
            return data[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return data[f] + (data[c] - data[f]) * (k - f)
    
    # Sort best tweets by normalized performance (imp_per_1k), not raw impressions
    best_tweets.sort(key=lambda t: t["imp_per_1k"], reverse=True)
    
    # Collect recent tweets across ecosystem (for "what's being talked about")
    # Include engagement metrics + age + tracked replies/quotes — these are OTHER
    # accounts' tweets, not the prediction target, so engagement data is fair game.
    # Age is calculated relative to the prediction tweet's post time, not now,
    # and tweets posted AFTER the prediction are excluded (no future leakage).
    # Tracked replies/quotes are only fetched for the top 5 tweets by impressions
    # (to limit API cost) — these show WHO in the ecosystem is engaging.
    recent_ecosystem = []
    # Sort all tweets by impressions to find top 5 for reply/quote fetching
    all_eco_tweets = []
    for author in all_author_data:
        for t in author.get("recent_tweets", []):
            all_eco_tweets.append((author, t))
    all_eco_tweets.sort(key=lambda x: x[1].get("public_metrics", {}).get("impression_count", 0), reverse=True)
    top_5_ids = {t["id"] for _, t in all_eco_tweets[:5]} if all_eco_tweets else set()
    
    for author in all_author_data:
        for t in author.get("recent_tweets", []):
            metrics = t.get("public_metrics", {})
            created = t.get("created_at", "")
            created_dt = None
            if created:
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except Exception:
                    pass
            
            entry = {
                "text": t.get("text", "")[:80],
                "handle": author.get("handle", ""),
                "impressions": metrics.get("impression_count", 0),
                "replies": metrics.get("reply_count", 0),
                "bookmarks": metrics.get("bookmark_count", 0),
                "quotes": metrics.get("quote_count", 0),
                "created_at": created,
                "_created_dt": created_dt,
            }
            
            # Fetch tracked replies/quotes for top tweets only (cost control)
            if tracked_author_ids and t.get("id") in top_5_ids:
                t["_author_username"] = author.get("handle", "")
                eng = get_engagement_context(t, tracked_author_ids)
                if eng.get("tracked_replies"):
                    entry["tracked_replies"] = eng["tracked_replies"]
                if eng.get("tracked_quotes"):
                    entry["tracked_quotes"] = eng["tracked_quotes"]
            
            recent_ecosystem.append(entry)
    
    # Sort by recency — take most recent 15 for "industry pulse"
    recent_ecosystem.sort(key=lambda t: t["created_at"], reverse=True)
    recent_pulse = recent_ecosystem[:15]
    # Store raw for per-prediction filtering (created_dt needed for age calc)
    all_recent_raw = recent_ecosystem
    
    return {
        "account_count": len(all_author_data),
        "median_imp_per_1k_followers": pct(imp_per_1k, 0.50),
        "p75_imp_per_1k": pct(imp_per_1k, 0.75),
        "median_impressions": pct(median_imps, 0.50),
        "best_tweets_normalized": best_tweets[:5],
        "worst_tweets_normalized": worst_tweets[:3],
        "recent_pulse": recent_pulse,
        "all_recent_raw": all_recent_raw,
    }


def get_reference_tweets(recent_tweets: list, baseline: dict) -> tuple:
    """Build reference tweet samples for the prompt.
    Returns (all_samples, hit_sample, miss_sample) where all_samples is
    every tweet ranked by impressions — gives the model a full pattern to learn from."""
    if not recent_tweets:
        return [], None, None

    sorted_tweets = sorted(recent_tweets, key=lambda t: t.get("public_metrics", {}).get("impression_count", 0), reverse=True)

    def format_sample(t, rank):
        return {
            "text": t.get("text", "")[:120],
            "impressions": t.get("public_metrics", {}).get("impression_count", 0),
            "likes": t.get("public_metrics", {}).get("like_count", 0),
            "retweets": t.get("public_metrics", {}).get("retweet_count", 0),
            "rank": rank,
        }

    all_samples = [format_sample(t, i + 1) for i, t in enumerate(sorted_tweets)]
    hit_sample = all_samples[0] if all_samples else None
    hit_sample["tier"] = "hit" if hit_sample else None
    miss_sample = all_samples[-1] if all_samples else None
    miss_sample["tier"] = "miss" if miss_sample else None

    return all_samples, hit_sample, miss_sample


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
    baseline = {"median_impressions": 0, "median_likes": 0, "median_retweets": 0, "sample_size": 0}
    recent_samples = []
    all_tweet_samples = []
    if fetch_baseline and handle:
        cached = get_cached_author(handle)
        if cached:
            baseline = cached["baseline"]
            recent_samples = cached["recent_samples"]
        elif author_id:
            try:
                recent = get_author_recent_tweets(author_id)
                baseline = compute_author_baseline(recent)
                all_samples, hit_sample, miss_sample = get_reference_tweets(recent, baseline)
                recent_samples = [s for s in [hit_sample, miss_sample] if s]
                all_tweet_samples = all_samples
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
        "author_all_tweets": all_tweet_samples,
        "trending_topics": [],
    }


# ── Environment config ───────────────────────────────────────────────────


def load_environment(env_name: str = "twitter") -> dict:
    """Load environment config (scoring dimensions, platform-specific settings).
    
    Each environment defines:
    - dimensions: what to score on (0-10 each)
    - probability_guide: rough mapping from scores to probability
    - extra_considerations: factors beyond scores
    - metric, target_description, timeframe_hours
    
    Environments live in environments/<name>.json.
    New environments (tiktok, email, sales) just add a new JSON file.
    """
    env_path = os.path.join(os.path.dirname(__file__), "environments", f"{env_name}.json")
    if not os.path.exists(env_path):
        print(f"  ⚠️  Environment '{env_name}' not found, falling back to twitter")
        env_path = os.path.join(os.path.dirname(__file__), "environments", "twitter.json")
    with open(env_path) as f:
        return json.load(f)


def build_prompt(env: dict) -> str:
    """Build the prediction prompt from environment config.
    
    Dimensions, probability guide, and considerations are injected dynamically.
    The prompt template is platform-agnostic — the environment config defines
    what the model knows about the platform.
    """
    # Build dimensions section grouped by category
    categories = {}
    for dim in env["dimensions"]:
        cat = dim.get("category", "General")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(dim)
    
    dim_lines = []
    for cat, dims in categories.items():
        dim_lines.append(f"{cat}:")
        for dim in dims:
            dim_lines.append(f"- {dim['name']}: {dim['description']}")
    
    dimensions_str = "\n".join(dim_lines)
    dimension_names = [d["name"] for d in env["dimensions"]]
    dimension_count = len(dimension_names)
    
    # Build probability guide
    guide_str = "\n".join(f"- {g}" for g in env.get("probability_guide", []))
    
    # Build extra considerations
    considerations_str = "\n".join(f"- {c}" for c in env.get("extra_considerations", []))
    
    # Build JSON output template with dynamic dimensions
    # Note: double braces {{ }} are literal braces in .format() strings
    json_fields = []
    for name in dimension_names:
        suffix = " (higher = safer)" if "risk" in name or "safe" in name else ""
        json_fields.append(f'  "{name}": <0-10{suffix}>')
    json_fields.append(f'  "probability": <float 0.0-1.0>')
    json_fields.append(f'  "point_estimate": <integer, your best guess at total {env["metric"]} after {{timeframe}}h>')
    json_fields.append(f'  "reasoning": "<2-3 sentences. Reference specific data: which of their past content is this most similar to? What makes it better or worse?>"')
    json_fields.append(f'  "pattern_analysis": "<1-2 sentences. What specific pattern does their best content use that this should match?>"')
    json_fields.append(f'  "suggestions": "<Rewrite directive. Identify the WEAKEST dimension by name and score. Explain WHY it scored low with a specific reference to this tweet vs their best content. Then prescribe one concrete fix — not a rephrased opener, but a structural change. Example: their best tweet opens with a specific number; this one opens with a vague claim. Fix: replace the first sentence with the actual metric. Another example: their best tweet ends with an implicit question that drives replies; this one ends with a statement. Fix: reframe the ending as a question only an insider would answer. Match the quality of their best work, not generic best practices.>"')
    json_str = ",\n".join(json_fields)
    
    content_type = env.get("content_type", "post")
    platform = env.get("platform", "social media")
    metric = env.get("metric", "impressions")
    target_desc = env.get("target_description", "the target")
    
    return "You are a " + platform + " reach forecasting harness.\n" \
        "\n" \
        "Given a " + content_type + " and its context, predict the probability (0.0 to 1.0) that this " + content_type + " will reach {target} " + metric + " within {timeframe} hours of being posted.\n" \
        "\n" \
        "The target is " + target_desc + ". You're predicting whether this is a standout " + content_type + " for this author.\n" \
        "\n" \
        "STEP 1 — Score the " + content_type + " vs their full history AND ecosystem on " + str(dimension_count) + " dimensions (0-10 each):\n" \
        "\n" + dimensions_str + "\n" \
        "\n" \
        "A " + content_type + " scoring below 6 on ANY dimension is unlikely to hit the target. Be honest — most " + content_type + "s are mediocre.\n" \
        "\n" \
        "STEP 2 — Convert scores to probability. As a rough guide:\n" + guide_str + "\n" \
        "\n" \
        "Also consider:\n" + considerations_str + "\n" \
        "\n" \
        "{llm_only_note}\n" \
        "\n" \
        "Return JSON:\n" \
        "{{\n" + json_str + "\n" \
        "}}\n" \
        "\n" \
        + content_type.capitalize() + ": {text}\n" \
        "Author: @{username} ({followers} followers, {following} following)\n" \
        "Author baseline: median {median_impressions:.0f} " + metric + "/" + content_type + ", median {median_likes:.0f} likes/" + content_type + ", median {median_retweets:.0f} retweets/" + content_type + " (sample: {baseline_sample} " + content_type + "s)\n" \
        "Author's recent " + content_type + "s ranked by " + metric + " (best to worst):\n" \
        "{all_tweets}\n" \
        "Ecosystem context (across {eco_accounts} accounts):\n" \
        "{ecosystem}\n" \
        "Media: {media}\n" \
        "Link context: {link_context}\n" \
        "Calibration feedback: {calibration}\n" \
        "Posted at: {posted_at}\n" \
        "Time elapsed since posting: {elapsed}\n" \
        "Timeframe: {timeframe}h\n" \
        "Target: {target} " + metric + " (" + target_desc + ")\n" \
        "Trending: {trending}"


# ── Prediction harness ───────────────────────────────────────────────────


PREDICTION_PROMPT = build_prompt(load_environment())


# ── Versioning ───────────────────────────────────────────────────────────


def _compute_config_hash() -> str:
    """Compute a hash of the current environment config."""
    import hashlib
    env = load_environment()
    version_input = json.dumps({
        "prompt": build_prompt(env),
        "dimensions": env.get("dimensions", []),
        "probability_guide": env.get("probability_guide", []),
        "extra_considerations": env.get("extra_considerations", []),
    }, sort_keys=True)
    return "v_" + hashlib.sha256(version_input.encode()).hexdigest()[:8]


def load_version() -> dict:
    """Load the stored version info from .version file."""
    version_path = os.path.join(os.path.dirname(__file__), ".version")
    if os.path.exists(version_path):
        with open(version_path) as f:
            return json.load(f)
    return {"version": "1.0.0", "hash": _compute_config_hash()}


def compute_version_bump(current: dict, new_env: dict) -> tuple[str, str]:
    """Determine what semver bump is needed when config changes.
    
    Returns (new_version, change_description).
    
    Rules:
    - Major (1.x.x): dimensions added or removed
    - Minor (x.1.x): dimension descriptions or probability guide changed
    - Patch (x.x.1): prompt wording changed (same structure)
    """
    old_dims = {d["name"] for d in current.get("dimensions", [])}
    new_dims = {d["name"] for d in new_env.get("dimensions", [])}
    
    old_version = current.get("version", "1.0.0")
    parts = [int(x) for x in old_version.split(".")]
    
    if old_dims != new_dims:
        added = new_dims - old_dims
        removed = old_dims - new_dims
        changes = []
        if added:
            changes.append(f"added: {', '.join(sorted(added))}")
        if removed:
            changes.append(f"removed: {', '.join(sorted(removed))}")
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
        return f"{parts[0]}.{parts[1]}.{parts[2]}", f"Major: {'; '.join(changes)}"
    
    # Check if dimension descriptions changed (minor)
    old_descs = {d["name"]: d.get("description", "") for d in current.get("dimensions", [])}
    new_descs = {d["name"]: d.get("description", "") for d in new_env.get("dimensions", [])}
    
    old_guide = current.get("probability_guide", [])
    new_guide = new_env.get("probability_guide", [])
    old_considerations = current.get("extra_considerations", [])
    new_considerations = new_env.get("extra_considerations", [])
    
    if old_descs != new_descs or old_guide != new_guide or old_considerations != new_considerations:
        changes = []
        for name in old_descs:
            if old_descs.get(name, "") != new_descs.get(name, ""):
                changes.append(f"{name} description changed")
        if old_guide != new_guide:
            changes.append("probability guide changed")
        if old_considerations != new_considerations:
            changes.append("considerations changed")
        parts[1] += 1
        parts[2] = 0
        return f"{parts[0]}.{parts[1]}.{parts[2]}", f"Minor: {'; '.join(changes)}"
    
    # Only prompt wording changed (patch)
    parts[2] += 1
    return f"{parts[0]}.{parts[1]}.{parts[2]}", "Patch: prompt wording changed"


def save_version(env: dict):
    """Save current config as the new .version baseline."""
    version_path = os.path.join(os.path.dirname(__file__), ".version")
    version_data = {
        "version": load_version().get("version", "1.0.0"),
        "hash": _compute_config_hash(),
        "dimensions": env.get("dimensions", []),
        "probability_guide": env.get("probability_guide", []),
        "extra_considerations": env.get("extra_considerations", []),
    }
    with open(version_path, "w") as f:
        json.dump(version_data, f, indent=2)


def get_harness_version() -> str:
    """Get the current harness version string (e.g., '1.0.0')."""
    return load_version().get("version", "1.0.0")


def get_harness_hash() -> str:
    """Get the current config hash (e.g., 'v_2e2c9363')."""
    return _compute_config_hash()


def check_version_changed() -> tuple[str, str] | None:
    """Check if the current config differs from the stored .version.
    Returns (new_version, description) if changed, None if same."""
    stored = load_version()
    current_hash = _compute_config_hash()
    if stored.get("hash") == current_hash:
        return None
    
    env = load_environment()
    new_version, description = compute_version_bump(stored, env)
    return new_version, description


HARNESS_VERSION = get_harness_version()
HARNESS_HASH = get_harness_hash()


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
        median_likes = 0
        median_retweets = 0
        baseline_sample = 0
        all_tweets_str = "  (no data — infer from knowledge)"
    else:
        llm_only_note = ""
        followers_str = context["author"]["followers"]
        following_str = context["author"]["following"]
        median_imp = baseline.get("median_impressions", 0)
        median_likes = baseline.get("median_likes", 0)
        median_retweets = baseline.get("median_retweets", 0)
        baseline_sample = baseline.get("sample_size", 0)
        
        # Format all tweets ranked by impressions
        all_tweets = context.get("author_all_tweets", [])
        if all_tweets:
            tweet_lines = []
            for t in all_tweets:
                tweet_lines.append(f"  #{t['rank']} ({t['impressions']:,} imp, {t['likes']} likes): \"{t['text']}\"")
            all_tweets_str = "\n".join(tweet_lines)
        else:
            # Fallback to best/worst from recent_top_tweets
            hit_miss = context.get("author_recent_top_tweets", [])
            if hit_miss:
                tweet_lines = []
                for t in hit_miss:
                    tweet_lines.append(f"  ({t['impressions']:,} imp, {t['likes']} likes): \"{t['text']}\"")
                all_tweets_str = "\n".join(tweet_lines)
            else:
                all_tweets_str = "  (no data)"

    # Format ecosystem context
    eco = context.get("ecosystem", {})
    if eco:
        eco_lines = [
            f"  Median efficiency: {eco.get('median_imp_per_1k_followers', 0):.1f} impressions per 1K followers",
            f"  Top quartile efficiency: {eco.get('p75_imp_per_1k', 0):.1f} imp per 1K followers",
            f"  Ecosystem median: {eco.get('median_impressions', 0):.0f} impressions/tweet",
            f"  Best tweets (normalized by follower count — quality, not just reach):",
        ]
        for bt in eco.get("best_tweets_normalized", [])[:3]:
            eco_lines.append(f"    @{bt['handle']} ({bt['followers']:,} followers, {bt['imp_per_1k']:.1f} imp/1K): \"{bt['text']}\"")
        # Filter ecosystem tweets to only those posted BEFORE the prediction tweet
        # (no future leakage — if we're predicting a tweet from 2pm, don't show
        # ecosystem tweets from 3pm onward)
        pred_posted = None
        posted_at_str = context.get("planned_post_time", context.get("created_at", ""))
        if posted_at_str:
            try:
                pred_posted = datetime.fromisoformat(posted_at_str.replace("Z", "+00:00"))
            except Exception:
                pass
        
        eco_lines.append(f"  Recent industry pulse (what subnets were tweeting about, with engagement + age relative to prediction):")
        # Use all_recent_raw and filter by post time
        all_raw = eco.get("all_recent_raw", eco.get("recent_pulse", []))
        filtered_pulse = []
        for rp in all_raw:
            if pred_posted and rp.get("_created_dt"):
                # Skip tweets posted after the prediction tweet
                if rp["_created_dt"] > pred_posted:
                    continue
                # Calculate age relative to prediction tweet's post time
                age_hrs = (pred_posted - rp["_created_dt"]).total_seconds() / 3600
                if age_hrs < 1:
                    age_str = f"{int(age_hrs * 60)}m before"
                elif age_hrs < 24:
                    age_str = f"{age_hrs:.0f}h before"
                else:
                    age_str = f"{age_hrs/24:.0f}d before"
            else:
                age_str = "unknown age"
            filtered_pulse.append((rp, age_str))
        
        # Sort by recency (most recent first among valid timestamps) and take top 5
        filtered_pulse.sort(key=lambda x: x[0].get("created_at", ""), reverse=True)
        for rp, age_str in filtered_pulse[:5]:
            eng = []
            if rp.get("replies", 0) > 0:
                eng.append(f"{rp['replies']} replies")
            if rp.get("bookmarks", 0) > 0:
                eng.append(f"{rp['bookmarks']} saves")
            if rp.get("quotes", 0) > 0:
                eng.append(f"{rp['quotes']} quotes")
            eng_str = ", ".join(eng) if eng else "no engagement"
            eco_lines.append(f"    @{rp['handle']} ({age_str}, {eng_str}): \"{rp['text']}\"")
            # Show tracked replies (from our accounts only — high signal)
            # Enrich with the replier's own cached metrics so the model knows
            # not just who replied, but how much reach that account has
            for tr in rp.get("tracked_replies", [])[:3]:
                reply_cache = get_cached_author(tr["handle"])
                if reply_cache:
                    reply_median = reply_cache.get("baseline", {}).get("median_impressions", 0)
                    reply_followers = reply_cache.get("followers", 0)
                    eco_lines.append(f"      ↳ @{tr['handle']} ({reply_followers:,} followers, median {reply_median:.0f} imp/tweet) replied: \"{tr['text'][:60]}\"")
                else:
                    eco_lines.append(f"      ↳ @{tr['handle']} ({tr['followers']:,} followers) replied: \"{tr['text'][:60]}\"")
            for tq in rp.get("tracked_quotes", [])[:2]:
                quote_cache = get_cached_author(tq["handle"])
                if quote_cache:
                    quote_median = quote_cache.get("baseline", {}).get("median_impressions", 0)
                    quote_followers = quote_cache.get("followers", 0)
                    eco_lines.append(f"      ↳ @{tq['handle']} ({quote_followers:,} followers, median {quote_median:.0f} imp/tweet) quoted: \"{tq['text'][:60]}\"")
                else:
                    eco_lines.append(f"      ↳ @{tq['handle']} ({tq['followers']:,} followers) quoted: \"{tq['text'][:60]}\"")
        ecosystem_str = "\n".join(eco_lines)
        eco_accounts = eco.get("account_count", 0)
    else:
        ecosystem_str = "  (no ecosystem data)"
        eco_accounts = 0

    # Format media context
    media_info = context.get("media", {})
    if media_info.get("has_media"):
        media_types = media_info.get("media_types", [])
        descriptions = media_info.get("descriptions", [])
        media_parts = []
        for i, (mt, desc) in enumerate(zip(media_types, descriptions)):
            media_parts.append(f"  {mt}: {desc}")
        media_str = "\n".join(media_parts) if media_parts else "  (has media, no details)"
    else:
        media_str = "  no media attached"

    # Format link context
    has_url = context.get("has_url", False) or ("http" in context.get("text", "") or "https" in context.get("text", ""))
    has_ext = context.get("has_external_url", False)
    is_quote = context.get("is_quote_tweet", False)
    if is_quote:
        link_context_str = "  Quote tweet (link is X-internal, NOT penalized by algorithm)"
    elif has_ext:
        link_context_str = "  External URL present (penalized 30-94% by algorithm)"
    elif has_url:
        link_context_str = "  URL present in text (treat as external unless clearly X-internal)"
    else:
        link_context_str = "  No links"

    # Compute calibration feedback from resolved predictions
    cal = compute_calibration()
    calibration_str = format_calibration_for_prompt(cal) if cal else "  (not enough resolved predictions yet — keep predicting)"

    prompt = PREDICTION_PROMPT.format(
        target=tgt,
        timeframe=tf,
        text=context["text"],
        username=context["author"]["username"],
        followers=followers_str,
        following=following_str,
        median_impressions=median_imp,
        median_likes=median_likes,
        median_retweets=median_retweets,
        baseline_sample=baseline_sample,
        all_tweets=all_tweets_str,
        eco_accounts=eco_accounts,
        ecosystem=ecosystem_str,
        media=media_str,
        link_context=link_context_str,
        calibration=calibration_str,
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
    Lower is better. 0 = perfect, 0.25 = no skill (always 0.5), 1 = worst.
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


def compute_calibration(predictions: list[dict] = None) -> dict:
    """Compute calibration stats from resolved predictions.
    
    Buckets predictions by probability range and compares predicted
    vs actual hit rate. Identifies systematic over/underconfidence.
    
    Returns {buckets, total_count, brier, bias_summary}
    """
    if predictions is None:
        predictions = load_predictions()
    
    resolved = [p for p in predictions if p.get("resolved") is True]
    if len(resolved) < 5:
        return {}  # Not enough data for meaningful calibration
    
    # Define buckets
    bucket_ranges = [
        (0.0, 0.2, "0-20%"),
        (0.2, 0.4, "20-40%"),
        (0.4, 0.6, "40-60%"),
        (0.6, 0.8, "60-80%"),
        (0.8, 1.01, "80-100%"),
    ]
    
    buckets = []
    for low, high, label in bucket_ranges:
        in_bucket = [p for p in resolved if low <= p["prediction"]["probability"] < high]
        if not in_bucket:
            continue
        predicted_avg = sum(p["prediction"]["probability"] for p in in_bucket) / len(in_bucket)
        actual_hits = sum(1 for p in in_bucket if p["actual_impressions"] >= p["target"])
        actual_rate = actual_hits / len(in_bucket)
        bias = predicted_avg - actual_rate  # positive = overconfident
        buckets.append({
            "range": label,
            "count": len(in_bucket),
            "predicted_avg": predicted_avg,
            "actual_rate": actual_rate,
            "bias": bias,
        })
    
    # Overall stats
    total_brier = brier_score(resolved)
    
    # Identify biggest biases
    biases = []
    for b in buckets:
        if b["count"] >= 2:  # Only flag buckets with enough data
            if b["bias"] > 0.1:
                biases.append(f"When you predict {b['range']}, you're right only {b['actual_rate']:.0%} of the time. You're overconfident here.")
            elif b["bias"] < -0.1:
                biases.append(f"When you predict {b['range']}, you're right {b['actual_rate']:.0%} of the time. You're underconfident here.")
    
    # Pattern-based biases (which types of tweets does it get wrong?)
    pattern_biases = []
    
    # URL tweets
    url_preds = [p for p in resolved if p.get("has_url")]
    if len(url_preds) >= 3:
        url_brier = brier_score(url_preds)
        url_bias = sum(p["prediction"]["probability"] - (1.0 if p["actual_impressions"] >= p["target"] else 0.0) for p in url_preds) / len(url_preds)
        if url_bias > 0.1:
            pattern_biases.append(f"Tweets with URLs: you're overconfident by {url_bias:.0%}. Links hurt reach more than you predict.")
    
    # High follower accounts (>10K)
    big_accounts = [p for p in resolved if p.get("followers_at_prediction", 0) > 10000]
    if len(big_accounts) >= 3:
        big_bias = sum(p["prediction"]["probability"] - (1.0 if p["actual_impressions"] >= p["target"] else 0.0) for p in big_accounts) / len(big_accounts)
        if big_bias > 0.1:
            pattern_biases.append(f"Large accounts (>10K followers): you're overconfident by {big_bias:.0%}.")
        elif big_bias < -0.1:
            pattern_biases.append(f"Large accounts (>10K followers): you're underconfident by {abs(big_bias):.0%}.")
    
    # Small accounts (<2K)
    small_accounts = [p for p in resolved if 0 < p.get("followers_at_prediction", 0) < 2000]
    if len(small_accounts) >= 3:
        small_bias = sum(p["prediction"]["probability"] - (1.0 if p["actual_impressions"] >= p["target"] else 0.0) for p in small_accounts) / len(small_accounts)
        if small_bias > 0.1:
            pattern_biases.append(f"Small accounts (<2K followers): you're overconfident by {small_bias:.0%}.")
        elif small_bias < -0.1:
            pattern_biases.append(f"Small accounts (<2K followers): you're underconfident by {abs(small_bias):.0%}.")
    
    return {
        "buckets": buckets,
        "total_count": len(resolved),
        "brier": total_brier,
        "bias_summary": biases,
        "pattern_biases": pattern_biases,
    }


def format_calibration_for_prompt(cal: dict) -> str:
    """Format calibration data as a string for the prompt."""
    if not cal or not cal.get("buckets"):
        return ""  # Not enough data yet
    
    lines = [f"Your track record ({cal['total_count']} resolved predictions, Brier: {cal['brier']:.3f}):"]
    
    for b in cal["buckets"]:
        arrow = "↑ overconfident" if b["bias"] > 0.05 else ("↓ underconfident" if b["bias"] < -0.05 else "calibrated")
        lines.append(f"  {b['range']}: predicted {b['predicted_avg']:.0%}, actual {b['actual_rate']:.0%} ({b['count']} preds) — {arrow}")
    
    if cal.get("bias_summary"):
        lines.append("Calibration notes:")
        for note in cal["bias_summary"]:
            lines.append(f"  - {note}")
    
    if cal.get("pattern_biases"):
        lines.append("Pattern biases:")
        for note in cal["pattern_biases"]:
            lines.append(f"  - {note}")
    
    return "\n".join(lines)


def enrich_tweet_text(tweet: dict) -> str:
    """Enrich tweet text by resolving t.co URLs and fetching linked content.
    
    For bare links to external articles, fetches the page and extracts
    readable text (headings, paragraphs) so the model can evaluate the
    content. For X-internal links (quote tweets, media), keeps original.
    """
    text = tweet.get("text", "")
    entities = tweet.get("entities", {})
    urls = entities.get("urls", [])
    
    if not urls:
        return text
    
    # Check if text is basically just a URL (link-only tweet)
    clean = re.sub(r'https?://\S+', '', text).strip()
    is_link_only = len(clean) < 15
    
    # Link-only tweet — skip entirely (can't evaluate without the content)
    if is_link_only:
        return ""  # Empty signals "skip this tweet"
    
    return text


def backfill_predictions(max_tweets_per_account: int = 20):
    """Backfill historical predictions for faster data accumulation.
    
    Fetches 20 tweets per account. Tweets 1-10 = baseline (shown to model).
    Tweets 11-20 = prediction targets (older, already past 24h, resolved immediately).
    
    No waiting for resolution — impressions are final since these tweets are >24h old.
    No extra resolve API calls — impressions come from the same fetch.
    
    Cost: ~$0.10 per account (20 tweet reads + 10 LLM calls). 90 accounts = ~$9.
    """
    print("\n" + "=" * 60)
    print("  PostProphet — Historical Backfill")
    print("=" * 60 + "\n")
    
    handles = load_tracked_accounts()
    existing = load_predictions()
    existing_ids = {p["tweet_id"] for p in existing}
    
    print(f"  Accounts: {len(handles)}")
    print(f"  Tweets per account: {max_tweets_per_account} (10 baseline + {max_tweets_per_account - 10} targets)")
    print(f"  Already predicted: {len(existing_ids)} tweets (skipped)")
    print()
    
    all_new_predictions = []
    stats = {"accounts_processed": 0, "tweets_predicted": 0, "errors": 0, "skipped": 0}
    
    for i, handle in enumerate(handles, 1):
        print(f"  [{i}/{len(handles)}] @{handle}...", end="")
        
        # Get user
        user = get_user_by_username(handle)
        if not user:
            print(" ❌ not found")
            stats["errors"] += 1
            continue
        
        user_id = user.get("id", "")
        followers = user.get("public_metrics", {}).get("followers_count", 0)
        if followers == 0:
            print(" ⚠️ 0 followers")
            stats["skipped"] += 1
            continue
        
        # Fetch 20 tweets (one API call)
        try:
            all_tweets = get_author_recent_tweets(user_id, max_results=max_tweets_per_account)
        except Exception as e:
            print(f" ❌ API error: {e}")
            stats["errors"] += 1
            continue
        
        if len(all_tweets) < 11:
            print(f" ⏭️ only {len(all_tweets)} tweets (need 11+)")
            stats["skipped"] += 1
            continue
        
        # Tweets 1-10 = baseline (most recent), tweets 11+ = prediction targets
        baseline_tweets = all_tweets[:10]
        target_tweets = all_tweets[10:]
        
        # Compute baseline from first 10
        baseline = compute_author_baseline(baseline_tweets)
        median_imp = baseline.get("median_impressions", 0)
        if median_imp == 0:
            print(" ⚠️ no baseline")
            stats["skipped"] += 1
            continue
        
        dynamic_target = int(median_imp * 2)
        
        # Build reference tweets from baseline only
        all_samples, hit_sample, miss_sample = get_reference_tweets(baseline_tweets, baseline)
        recent_samples = [s for s in [hit_sample, miss_sample] if s]
        
        # Cache author data
        cache_author(handle, user, baseline, recent_samples)
        
        # Predict on each target tweet (older tweets, already resolved)
        account_preds = []
        for tweet in target_tweets:
            tweet_id = tweet["id"]
            if tweet_id in existing_ids:
                stats["skipped"] += 1
                continue
            
            # Skip tweets with no text at all
            if not tweet.get("text", "").strip():
                continue
            
            # Skip quote tweets — can't see quoted content yet (future feature)
            refs = tweet.get("referenced_tweets", [])
            if any(r.get("type") == "quoted" for r in refs):
                continue
            
            # Skip tweets that are replies in threads — need full thread context (future feature)
            if any(r.get("type") == "replied_to" for r in refs):
                continue
    
            # Skip link-only tweets that resolve to X-internal URLs
            # (quote tweets, self-quotes, X articles) — we can't see the content
            import re as _re_skip
            _clean = _re_skip.sub(r'https?://\S+', '', tweet.get("text", "")).strip()
            if len(_clean) < 15:
                _entities = tweet.get("entities", {})
                _urls = _entities.get("urls", [])
                _all_x_links = all(
                    any(d in u.get("expanded_url", "") for d in ["x.com", "twitter.com"])
                    for u in _urls
                ) if _urls else True
                if _all_x_links:
                    continue  # Can't evaluate — skip X-internal link-only tweets
            
            # This tweet is >24h old — impressions are final
            actual_impressions = tweet.get("public_metrics", {}).get("impression_count", 0)
            
            # Build context (same as track mode, but NO engagement leakage)
            media_info = analyze_media(tweet) if tweet.get("_media") else {"has_media": False, "media_types": [], "descriptions": []}
            
            # Enrich tweet text (resolves link-only tweets — returns empty if should skip)
            enriched_text = enrich_tweet_text(tweet)
            if not enriched_text:
                continue  # Link-only tweet we can't evaluate
            
            # Check if tweet is >24h old
            created_at = tweet.get("created_at", "")
            try:
                posted_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - posted_dt).total_seconds() / 3600
                if age_hours < 24:
                    continue  # Skip tweets that haven't reached 24h yet
            except Exception:
                continue
            
            context = {
                "tweet_id": tweet_id,
                "text": enriched_text,
                "created_at": created_at,
                "planned_post_time": created_at,
                "platform": "x",
                "author": {
                    "username": handle,
                    "followers": followers,
                    "following": user.get("public_metrics", {}).get("following_count", 0),
                    "tweet_count": user.get("public_metrics", {}).get("tweet_count", 0),
                },
                "author_baseline": baseline,
                "author_recent_top_tweets": recent_samples,
                "author_all_tweets": all_samples,
                "media": media_info,
                "entities": tweet.get("entities", {}),
                "referenced_tweets": tweet.get("referenced_tweets", []),
                "trending_topics": [],
                "has_x_data": True,
            }
            
            # Predict
            try:
                prediction = predict(context, target=dynamic_target)
                time.sleep(2)  # Rate limit: stay under 200K TPM on gpt-4o-mini
            except Exception as e:
                print(f" ❌ predict error: {e}")
                stats["errors"] += 1
                continue
            
            # Calculate elapsed time at prediction
            elapsed_str = "unknown"
            try:
                posted_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                elapsed_hrs = age_hours
                elapsed_str = f"{elapsed_hrs:.1f}h"
            except Exception:
                pass
            
            record = {
                "tweet_id": tweet_id,
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "timeframe_hours": TIMEFRAME_HOURS,
                "target": dynamic_target,
                "model": OPENAI_MODEL,
                "harness_version": HARNESS_VERSION,
                "harness_hash": HARNESS_HASH,
                "followers_at_prediction": followers,
                "elapsed_at_prediction": elapsed_str,
                "has_url": "http" in context["text"] or "https" in context["text"],
                "has_external_url": any(
                    u.get("expanded_url", "") and not any(d in u["expanded_url"] for d in ["x.com", "twitter.com"])
                    for u in context.get("entities", {}).get("urls", [])
                ) if context.get("entities") else ("http" in context["text"] or "https" in context["text"]),
                "is_quote_tweet": any(
                    r.get("type") == "quoted" for r in context.get("referenced_tweets", [])
                ) if context.get("referenced_tweets") else False,
                "context": context,
                "prediction": prediction,
                "resolve_after": (
                    datetime.fromisoformat(created_at.replace("Z", "+00:00")) + timedelta(hours=TIMEFRAME_HOURS)
                ).isoformat(),
                "resolved": True,  # Already resolved — impressions are final
                "actual_impressions": actual_impressions,
            }
            
            save_prediction(record)
            all_new_predictions.append(record)
            account_preds.append(record)
            existing_ids.add(tweet_id)
            stats["tweets_predicted"] += 1
        
        prob_str = ""
        if account_preds:
            avg_prob = sum(p["prediction"]["probability"] for p in account_preds) / len(account_preds)
            hits = sum(1 for p in account_preds if p["actual_impressions"] >= p["target"])
            prob_str = f" ✓ {len(account_preds)} preds | avg {avg_prob:.0%} | {hits} hits"
        
        print(f" median {median_imp:.0f} | target {dynamic_target:,}{prob_str}")
        stats["accounts_processed"] += 1
    
    # Compute Brier for the backfill batch
    print()
    print(f"  ────────────────────────────────────────")
    print(f"  Backfill complete:")
    print(f"  Accounts processed: {stats['accounts_processed']}")
    print(f"  Predictions made:   {stats['tweets_predicted']}")
    print(f"  Skipped:            {stats['skipped']}")
    print(f"  Errors:             {stats['errors']}")
    
    if all_new_predictions:
        brier = brier_score(all_new_predictions)
        hits = sum(1 for p in all_new_predictions if p["actual_impressions"] >= p["target"])
        print(f"  Hits:               {hits}/{len(all_new_predictions)}")
        print(f"  Brier:              {brier:.4f}")
        
        # Save score record
        score_record = {
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "batch_size": len(all_new_predictions),
            "brier_score": brier,
            "model": OPENAI_MODEL,
            "harness_version": HARNESS_VERSION,
            "harness_hash": HARNESS_HASH,
            "timeframe_hours": TIMEFRAME_HOURS,
            "target": "2x_median",
            "batch_type": "backfill",
        }
        save_score(score_record)
        print(f"  Score saved.")
    
    print(f"  ────────────────────────────────────────")
    
    # Rebuild eval set if we have enough
    all_preds = load_predictions()
    resolved = [p for p in all_preds if p.get("resolved")]
    print(f"\n  Total resolved predictions: {len(resolved)}")
    if len(resolved) >= 10:
        build_eval_set(all_preds)
    
    return stats


EVAL_FILE = os.path.join(DATA_DIR, "eval_set.jsonl")


def build_eval_set(predictions: list[dict] = None, min_size: int = 50):
    """Build or update the frozen eval set from resolved predictions.
    
    The eval set is a snapshot of resolved predictions used to score PRs.
    Miners never see which tweets are in the eval set.
    
    Takes the most recent resolved predictions up to min_size.
    Refreshed periodically (not on every run) to prevent overfitting.
    """
    if predictions is None:
        predictions = load_predictions()
    
    resolved = [p for p in predictions if p.get("resolved") is True]
    if len(resolved) < 10:
        print(f"  Not enough resolved predictions for eval set ({len(resolved)}/10 min)")
        return
    
    # Take most recent resolved, up to min_size
    eval_preds = sorted(resolved, key=lambda p: p.get("resolved_at", p.get("predicted_at", "")), reverse=True)[:min_size]
    
    # Strip the prediction (probability) — eval re-runs the harness
    # Keep: tweet text, author, baseline, target, actual impressions
    eval_set = []
    for p in eval_preds:
        eval_set.append({
            "tweet_id": p["tweet_id"],
            "text": p["context"]["text"],
            "author": p["context"]["author"],
            "author_baseline": p["context"].get("author_baseline", {}),
            "author_all_tweets": p["context"].get("author_all_tweets", []),
            "author_recent_top_tweets": p["context"].get("author_recent_top_tweets", []),
            "media": p["context"].get("media", {}),
            "target": p["target"],
            "actual_impressions": p["actual_impressions"],
            "hit": p["actual_impressions"] >= p["target"],
            "created_at": p["context"].get("created_at", ""),
            "original_probability": p["prediction"]["probability"],
            "original_version": p.get("harness_version", "unknown"),
        })
    
    with open(EVAL_FILE, "w") as f:
        for item in eval_set:
            f.write(json.dumps(item) + "\n")
    
    print(f"  Eval set: {len(eval_set)} predictions saved to {EVAL_FILE}")
    print(f"  Version: {HARNESS_VERSION} ({HARNESS_HASH})")


def run_eval() -> dict:
    """Run the current harness against the frozen eval set.
    
    Re-predicts each tweet in the eval set using the current prompt/config,
    then computes Brier score. Used to evaluate PRs.
    
    Returns {version, hash, brier, count, results}
    """
    if not os.path.exists(EVAL_FILE):
        print("  No eval set found. Run 'python postprophet.py eval-set' first.")
        return {}
    
    with open(EVAL_FILE) as f:
        eval_items = [json.loads(line) for line in f if line.strip()]
    
    if not eval_items:
        print("  Eval set is empty.")
        return {}
    
    print(f"\n  Running eval: {len(eval_items)} predictions")
    print(f"  Harness: {HARNESS_VERSION} ({HARNESS_HASH})")
    print()
    
    results = []
    total_brier = 0.0
    
    for i, item in enumerate(eval_items, 1):
        # Build context from eval item (no engagement leakage)
        context = {
            "tweet_id": item["tweet_id"],
            "text": item["text"],
            "created_at": item["created_at"],
            "planned_post_time": item["created_at"],
            "platform": "x",
            "author": item["author"],
            "author_baseline": item["author_baseline"],
            "author_recent_top_tweets": item.get("author_recent_top_tweets", []),
            "author_all_tweets": item.get("author_all_tweets", []),
            "media": item.get("media", {"has_media": False, "media_types": [], "descriptions": []}),
            "entities": {},
            "referenced_tweets": [],
            "trending_topics": [],
            "has_x_data": True,
        }
        
        # Predict
        try:
            prediction = predict(context, target=item["target"])
            prob = prediction.get("probability", 0.5)
        except Exception as e:
            print(f"  [{i}/{len(eval_items)}] Error: {e}")
            prob = 0.5
        
        actual = 1.0 if item["hit"] else 0.0
        brier_contrib = (prob - actual) ** 2
        total_brier += brier_contrib
        results.append({
            "tweet_id": item["tweet_id"],
            "author": item["author"]["username"],
            "probability": prob,
            "actual": actual,
            "brier": brier_contrib,
            "hit": item["hit"],
            "target": item["target"],
            "actual_impressions": item["actual_impressions"],
        })
        
        verdict = "YES" if prob >= 0.5 else "NO"
        print(f"  [{i}/{len(eval_items)}] @{item['author']['username']}: {verdict} ({prob:.0%}) | target {item['target']:,} | actual {item['actual_impressions']:,} | {'✅' if (prob >= 0.5) == item['hit'] else '❌'}")
    
    brier = total_brier / len(results)
    
    # Compare to original
    original_brier = sum(
        (item["original_probability"] - (1.0 if item["hit"] else 0.0)) ** 2
        for item in eval_items
    ) / len(eval_items)
    
    print()
    print(f"  ────────────────────────────────────────")
    print(f"  Eval results: {len(results)} predictions")
    print(f"  Current harness:  {HARNESS_VERSION} ({HARNESS_HASH})")
    print(f"  Brier:            {brier:.4f}")
    print(f"  Original:         {eval_items[0].get('original_version', '?')} → {original_brier:.4f}")
    delta = original_brier - brier
    if delta > 0:
        print(f"  Improvement:      {delta:.4f} ✅")
    elif delta < 0:
        print(f"  Regression:       {abs(delta):.4f} ❌")
    else:
        print(f"  No change.")
    print(f"  ────────────────────────────────────────")
    
    return {
        "version": HARNESS_VERSION,
        "hash": HARNESS_HASH,
        "brier": brier,
        "original_brier": original_brier,
        "count": len(results),
        "results": results,
    }


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
    # Strip non-serializable datetime objects from ecosystem data
    def clean_obj(obj):
        if isinstance(obj, dict):
            return {k: clean_obj(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, list):
            return [clean_obj(item) for item in obj]
        return obj
    clean_prediction = clean_obj(prediction)
    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(clean_prediction) + "\n")


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

        # Calculate elapsed time at prediction
        elapsed_str = "unknown"
        try:
            posted_at = datetime.fromisoformat(context["created_at"].replace("Z", "+00:00"))
            elapsed_hrs = (datetime.now(timezone.utc) - posted_at).total_seconds() / 3600
            elapsed_str = f"{elapsed_hrs:.1f}h"
        except Exception:
            pass

        record = {
            "tweet_id": context["tweet_id"],
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "timeframe_hours": TIMEFRAME_HOURS,
            "target": TARGET_IMPRESSIONS,
            "model": OPENAI_MODEL,
            "harness_version": HARNESS_VERSION,
            "harness_hash": HARNESS_HASH,
            "followers_at_prediction": context["author"]["followers"],
            "elapsed_at_prediction": elapsed_str,
            "has_url": "http" in context["text"] or "https" in context["text"],
            "has_external_url": any(
                u.get("expanded_url", "") and not any(d in u["expanded_url"] for d in ["x.com", "twitter.com"])
                for u in context.get("entities", {}).get("urls", [])
            ) if context.get("entities") else ("http" in context["text"] or "https" in context["text"]),
            "is_quote_tweet": any(
                r.get("type") == "quoted" for r in context.get("referenced_tweets", [])
            ) if context.get("referenced_tweets") else False,
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
        target = p["target"]
        username = p["context"]["author"]["username"]
        verdict = "YES" if prob >= 0.5 else "NO"
        print(f"  @{username}: {verdict} ({prob:.0%}) | target {target:,} | actual {actual:,} → {'✅ HIT' if hit else '❌ MISS'}")

        resolved_batch.append(p)

    if not resolved_batch:
        print(f"  No predictions ready to resolve. {unresolved} still pending.")
        return

    # Calculate Brier score for this batch
    score = brier_score(resolved_batch)
    print(f"\n  Brier score: {score:.4f} (0=perfect, 1=worst)")

    # Save score record
    score_record = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "batch_size": len(resolved_batch),
        "brier_score": score,
        "model": OPENAI_MODEL,
        "harness_version": HARNESS_VERSION,
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
        brier = s.get("brier_score", 1.0)
        batch = s["batch_size"]
        print(f"  {date:<24} {model:<20} {brier:<8.4f} {batch:<6}")

    # Overall stats
    all_briers = [s.get("brier_score", 1.0) for s in scores]
    avg = sum(all_briers) / len(all_briers)
    total_preds = sum(s["batch_size"] for s in scores)
    print(f"\n  Average Brier:     {avg:.4f}")
    print(f"  Best Brier:        {min(all_briers):.4f}")
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

            print(f"  @{username} ({followers:,} followers, median {avg_imp:.0f} imp/tweet)")
            print(f"  Tweet: {tweet_text}...")
            verdict = "YES" if prob >= 0.5 else "NO"
            print(f"  Target: {target:,} impressions | Predicted: {verdict} ({prob:.0%}) | Actual: {actual:,} | {'✅ HIT' if hit else '❌ MISS'}")
            if elapsed:
                print(f"  Tweet age at prediction: {elapsed}")
            print(f"  Reasoning: {reasoning}")
            if suggestions:
                print(f"  Suggestion: {suggestions}")
            print()


# ── Predict for an unpublished tweet (product mode) ─────────────────────


def build_ecosystem_from_cache(exclude_handle: str = None) -> dict:
    """Build ecosystem baseline from cached author data.
    Zero API cost — uses only what's already in the cache.
    Returns empty dict if not enough cached accounts."""
    cache = load_author_cache()
    if not cache:
        return {}
    
    all_author_data = []
    for handle, data in cache.items():
        if exclude_handle and handle == exclude_handle:
            continue
        followers = data.get("followers", 0)
        baseline = data.get("baseline", {})
        median_imp = baseline.get("median_impressions", 0)
        if followers > 0 and median_imp > 0:
            all_author_data.append({
                "handle": handle,
                "followers": followers,
                "median_impressions": median_imp,
                "recent_tweets": [],  # Cache doesn't store full tweet list
            })
    
    if len(all_author_data) < 3:
        return {}  # Not enough accounts for meaningful ecosystem
    
    return compute_ecosystem_baseline(all_author_data)


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
    tf = timeframe_hours or TIMEFRAME_HOURS
    post_time = planned_post_time or datetime.now(timezone.utc).isoformat()

    # Try to fetch author info from X API
    # If no bearer token or API fails, fall back to LLM-only mode
    author_metrics = {"followers_count": 0, "following_count": 0, "tweet_count": 0}
    baseline = {"median_impressions": 0, "median_likes": 0, "median_retweets": 0, "sample_size": 0}
    recent_samples = []
    all_tweet_samples = []
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
                        all_samples, hit_sample, miss_sample = get_reference_tweets(recent, baseline)
                        recent_samples = [s for s in [hit_sample, miss_sample] if s]
                        all_tweet_samples = all_samples
                    except Exception:
                        pass

                # Cache for future calls
                cache_author(author_username, user_data, baseline, recent_samples)
                has_x_data = True

            except Exception as e:
                print(f"  (X API unavailable: {e}. Using LLM-only mode.)")

        # Fetch trending (not cached — changes frequently)
        try:
            trending = get_trending_topics()
        except Exception:
            pass

    # Use author's 2x median as target if no explicit target given
    if target is None and baseline.get("median_impressions"):
        tgt = int(baseline["median_impressions"] * 2)
    else:
        tgt = target or TARGET_IMPRESSIONS

    # Build ecosystem context from cache (zero API cost)
    ecosystem = build_ecosystem_from_cache(exclude_handle=author_username)
    if ecosystem:
        print(f"  Ecosystem: {ecosystem.get('account_count', 0)} accounts | median efficiency {ecosystem.get('median_imp_per_1k_followers', 0):.1f} imp/1K followers")

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
        "author_all_tweets": all_tweet_samples,
        "ecosystem": ecosystem,
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
  python postprophet.py add <handle> [handle2...]  — Add accounts to tracking + fetch their data
  python postprophet.py track            — Capture fresh tweets from tracked accounts (<12h old) and predict
  python postprophet.py resolve          — Resolve pending predictions at posted_at + 24h and score with Brier
  python postprophet.py report           — Show score history + detailed predictions with reasoning
  python postprophet.py predict <tweet> — Predict reach for an unpublished tweet
                                           (requires --author, optional: --target, --timeframe, --post-time)
  python craft.py --author <handle> --topic "<topic>"  — Iterative tweet improvement loop

Scoring: content quality + algorithmic signals (0-10 scale)
Target:  2x author's median impressions (aspirational, grounded in real history)
Eval:    Brier score (0=perfect, 1=worst)

Environment:
  X_BEARER_TOKEN          — X API v2 bearer token (pay-per-use)
  OPENAI_API_KEY          — OpenAI API key
  POSTPROPHET_MODEL       — LLM model (default: gpt-4o-mini, model-agnostic)
  POSTPROPHET_TIMEFRAME   — Hours before resolving (default: 24)
  POSTPROPHET_TARGET      — Fallback target (default: 10000, overridden by 2x median in track mode)
  POSTPROPHET_BATCH       — Tweets per keyword capture batch (default: 20)
""")
        return

    cmd = sys.argv[1]

    if cmd == "backfill":
        backfill_predictions()
    elif cmd == "eval-set":
        build_eval_set()
    elif cmd == "eval":
        run_eval()
    elif cmd == "version":
        changed = check_version_changed()
        if changed:
            new_ver, desc = changed
            print(f"  Current: {HARNESS_VERSION} ({HARNESS_HASH})")
            print(f"  Config changed → {new_ver}")
            print(f"  {desc}")
            print(f"\n  Run 'python postprophet.py commit-version' to accept")
        else:
            print(f"  Current: {HARNESS_VERSION} ({HARNESS_HASH})")
            print(f"  No changes.")
    elif cmd == "commit-version":
        changed = check_version_changed()
        if not changed:
            print("  No changes to commit.")
        else:
            new_ver, desc = changed
            env = load_environment()
            version_data = load_version()
            version_data["version"] = new_ver
            version_data["hash"] = _compute_config_hash()
            version_data["dimensions"] = env["dimensions"]
            version_data["probability_guide"] = env.get("probability_guide", [])
            version_data["extra_considerations"] = env.get("extra_considerations", [])
            with open(os.path.join(os.path.dirname(__file__), ".version"), "w") as f:
                json.dump(version_data, f, indent=2)
            print(f"  Committed: {new_ver} ({version_data['hash']})")
            print(f"  {desc}")
    elif cmd == "add":
        # Add accounts to tracking list and fetch their data
        if len(sys.argv) < 3:
            print("Usage: python postprophet.py add <handle1> [handle2] [handle3] ...")
            return
        handles = sys.argv[2:]
        add_accounts(handles)
    elif cmd == "capture":
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
