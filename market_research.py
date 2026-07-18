#!/usr/bin/env python3
"""PostProphet — Market Research Agent

Runs daily to monitor the AI agent / open source / Bittensor ecosystem.
Gathers: key voices, trending themes, competitor moves, relevant research.
Outputs a market brief that feeds into the content agent.

Runs as a cron job. Saves briefs to data/market_briefs/.
"""

import os
import json
import httpx
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "market_briefs")
os.makedirs(DATA_DIR, exist_ok=True)

X_BEARER_TOKEN = os.environ.get("X_BEARER_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("POSTPROPHET_MODEL", "gpt-4o-mini")

# Key voices to monitor — AI agents, open source, Bittensor, content/marketing
KEY_VOICES = [
    # AI agent builders
    "AndrewYNg",
    "swyx",
    "simonw",
    # Open source ML
    "AnthropicAI",
    "OpenAI",
    "huggingface",
    # Bittensor ecosystem
    "bittensor_",
    "const_reborn",
    # Content / marketing / prediction
    "garyvee",
    "jeremyhoward",
]

# Themes to track
THEMES = [
    "AI agents",
    "agent loops",
    "evaluation harness",
    "trustless systems",
    "content prediction",
    "open source AI",
    "Bittensor",
    "Gittensor",
    "LLM calibration",
    "prediction markets",
]

# Competitors / adjacent tools
COMPETITORS = [
    "Buffer",
    "Hypefury",
    "Taplio",
    "PostPerfect",
    "Ocoya",
]


def x_headers():
    return {"Authorization": f"Bearer {X_BEARER_TOKEN}"}


def fetch_recent_tweets(handles, max_per_account=5):
    """Fetch recent tweets from key voices."""
    all_tweets = []
    
    for handle in handles:
        # Get user ID
        url = f"https://api.twitter.com/2/users/by/username/{handle}"
        resp = httpx.get(url, headers=x_headers(), timeout=20)
        if resp.status_code != 200:
            continue
        data = resp.json()
        user = data.get("data", {})
        user_id = user.get("id")
        if not user_id:
            continue
        
        followers = user.get("public_metrics", {}).get("followers_count", 0)
        
        # Get recent tweets
        url = f"https://api.twitter.com/2/users/{user_id}/tweets"
        params = {
            "max_results": max_per_account,
            "tweet.fields": "public_metrics,created_at,entities",
            "exclude": "retweets,replies",
        }
        resp = httpx.get(url, headers=x_headers(), params=params, timeout=20)
        if resp.status_code != 200:
            continue
        tweets = resp.json().get("data", [])
        
        for t in tweets:
            all_tweets.append({
                "author": handle,
                "followers": followers,
                "text": t.get("text", "")[:200],
                "impressions": t.get("public_metrics", {}).get("impression_count", 0),
                "likes": t.get("public_metrics", {}).get("like_count", 0),
                "replies": t.get("public_metrics", {}).get("reply_count", 0),
                "created_at": t.get("created_at", ""),
            })
    
    # Sort by engagement (impressions + likes + replies)
    all_tweets.sort(key=lambda t: t["impressions"] + t["likes"] * 10 + t["replies"] * 20, reverse=True)
    return all_tweets[:30]  # Top 30 most engaging


def search_theme_tweets(theme, max_results=10):
    """Search for tweets about a theme."""
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": f"{theme} -is:retweet lang:en",
        "max_results": max(10, min(max_results, 10)),
        "tweet.fields": "public_metrics,created_at,author_id",
    }
    resp = httpx.get(url, headers=x_headers(), params=params, timeout=20)
    if resp.status_code != 200:
        return []
    
    tweets = resp.json().get("data", [])
    results = []
    for t in tweets:
        metrics = t.get("public_metrics", {})
        results.append({
            "text": t.get("text", "")[:200],
            "impressions": metrics.get("impression_count", 0),
            "likes": metrics.get("like_count", 0),
            "created_at": t.get("created_at", ""),
        })
    
    # Sort by engagement
    results.sort(key=lambda t: t["impressions"] + t["likes"] * 10, reverse=True)
    return results[:5]


def generate_brief(key_tweets, theme_tweets, git_log=""):
    """Use LLM to synthesize a market brief from the gathered data."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Format key voice tweets
    voices_text = []
    for t in key_tweets[:20]:
        voices_text.append(f"@{t['author']} ({t['followers']:,} followers, {t['impressions']:,} imp): {t['text']}")
    voices_str = "\n".join(voices_text)
    
    # Format theme tweets
    themes_text = []
    for theme, tweets in theme_tweets.items():
        if tweets:
            themes_text.append(f"\n{theme.upper()}:")
            for t in tweets[:3]:
                themes_text.append(f"  - {t['text']} ({t['impressions']:,} imp)")
    themes_str = "\n".join(themes_text)
    
    prompt = f"""You are a market research analyst for PostProphet, a tweet reach prediction harness.

Your job is to synthesize the following data into a concise market brief that a content agent can use to write tweets. Focus on what's interesting, what's trending, and what matters for an AI agent / open source / Bittensor audience.

KEY VOICES (recent tweets, sorted by engagement):
{voices_str}

TRENDING THEMES:
{themes_str}

DEV CONTEXT (recent git activity):
{git_log if git_log else '(no dev context provided)'}

Write a market brief with these sections:
1. WHAT'S HOT — 2-3 themes or topics that are getting traction right now
2. KEY VOICES — 1-2 notable takes from monitored accounts that are worth engaging with
3. ANGLES — 2-3 specific angles for PostProphet content that connect dev progress to market trends
4. COMPETITOR WATCH — any notable moves from adjacent tools (if visible in the data)

Keep it concise. Bullet points. No fluff. This feeds a content agent that writes tweets."""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.7,
    )
    
    return resp.choices[0].message.content


def get_git_log():
    """Get recent git log for dev context."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-10"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        return result.stdout.strip()
    except Exception:
        return ""


def main():
    print("=" * 60)
    print("  PostProphet — Market Research Agent")
    print("=" * 60 + "\n")
    
    if not X_BEARER_TOKEN:
        print("  ⚠️  No X_BEARER_TOKEN set. Skipping live data.")
        key_tweets = []
        theme_tweets = {}
    else:
        print("  Fetching key voices...")
        key_tweets = fetch_recent_tweets(KEY_VOICES)
        print(f"  ✓ {len(key_tweets)} tweets from {len(KEY_VOICES)} voices")
        
        print("  Searching themes...")
        theme_tweets = {}
        for theme in THEMES[:5]:  # Limit to 5 themes per run (API cost)
            theme_tweets[theme] = search_theme_tweets(theme)
            print(f"  ✓ {theme}: {len(theme_tweets[theme])} tweets")
    
    print("  Getting git log...")
    git_log = get_git_log()
    
    print("  Generating brief...")
    if key_tweets or theme_tweets or git_log:
        brief = generate_brief(key_tweets, theme_tweets, git_log)
    else:
        brief = "No data available. Check API credentials."
    
    # Save brief
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    brief_file = os.path.join(DATA_DIR, f"brief_{timestamp}.md")
    with open(brief_file, "w") as f:
        f.write(f"# Market Brief — {timestamp}\n\n")
        f.write(brief)
    
    print(f"\n  Brief saved: {brief_file}")
    print("\n" + "=" * 60)
    print(brief)
    print("\n" + "=" * 60)
    
    return brief


if __name__ == "__main__":
    main()
