"""
PostProphet Craft — Iterative tweet improvement loop.

Drafts a tweet, gets PostProphet's prediction + reasoning, revises based on
feedback, repeats until target confidence or max iterations.

The writing agent sees the author's actual hit/miss reference tweets and
PostProphet's pattern analysis — so it can match what works instead of
guessing.

Usage:
  python craft.py --author <handle> --topic "<what the tweet should be about>"
                  [--target-confidence 0.80] [--max-iterations 5]

Requires the same .env as postprophet.py (X_BEARER_TOKEN, OPENAI_API_KEY).
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone

from openai import OpenAI

# Import PostProphet's predict function
sys.path.insert(0, os.path.dirname(__file__))
from postprophet import predict_tweet, OPENAI_MODEL, get_cached_author, get_user_by_username, get_author_recent_tweets, compute_author_baseline, get_reference_tweets, cache_author, x_headers, X_BEARER_TOKEN


def fetch_author_examples(handle):
    """Fetch the author's reference hit/miss tweets for the writing agent."""
    if not X_BEARER_TOKEN:
        return None, None
    
    # Check cache first
    cached = get_cached_author(handle)
    if cached and cached.get("recent_samples"):
        samples = cached["recent_samples"]
        hit = next((s for s in samples if s.get("tier") == "hit"), None)
        miss = next((s for s in samples if s.get("tier") == "miss"), None)
        if hit and miss:
            return hit, miss
    
    # Cold fetch
    user = get_user_by_username(handle)
    if not user:
        return None, None
    user_id = user.get("id", "")
    if not user_id:
        return None, None
    
    recent = get_author_recent_tweets(user_id)
    if not recent:
        return None, None
    
    baseline = compute_author_baseline(recent)
    all_samples, hit, miss = get_reference_tweets(recent, baseline)
    
    if hit or miss:
        cache_author(handle, user, baseline, [s for s in [hit, miss] if s])
    
    return hit, miss


def draft_tweet(client, topic, author, hit_tweet=None, miss_tweet=None, previous_tweet=None, feedback=None):
    """Draft or revise a tweet using an LLM guided by PostProphet's feedback."""
    
    from postprophet import load_environment
    env = load_environment()
    
    # Build reference context
    ref_section = ""
    if hit_tweet or miss_tweet:
        ref_lines = []
        if hit_tweet:
            ref_lines.append(f"THEIR BEST TWEET ({hit_tweet['impressions']:,} impressions): \"{hit_tweet['text']}\"")
        if miss_tweet:
            ref_lines.append(f"THEIR WORST TWEET ({miss_tweet['impressions']:,} impressions): \"{miss_tweet['text']}\"")
        ref_section = f"""
Author's reference tweets (REAL examples of what works and what doesn't for this account):
{chr(10).join(ref_lines)}

CRITICAL: Your tweet must structurally resemble their best tweet — same hook style, similar length, same tone. Do NOT write generic hype. Do NOT use exclamation marks unless their best tweet does. Do NOT add hashtags unless their best tweet has them. Do NOT ask rhetorical questions unless their best tweet does. Mirror what actually works.
"""
    
    if previous_tweet is None:
        # First draft — just write the tweet
        prompt = f"""You are a ghostwriter for @{author} on X. You write tweets that sound like them and perform like their best work.

Write a single tweet about: {topic}
{ref_section}
Rules:
- Maximum 280 characters
- Match the voice, length, and hook style of their best tweet
- Be specific and concrete, not vague or hype-y
- No exclamation marks, no hashtags, no rhetorical questions unless their best tweet uses them
- Say something real, not "the future is here"
- One tweet only, no threads

Return only the tweet text, nothing else."""
    else:
        # Revision — use PostProphet's feedback to improve
        pattern = feedback.get("pattern_analysis", "")
        verdict = "YES" if feedback.get("probability", 0) >= 0.5 else "NO"
        # Build scores string dynamically from feedback
        _score_parts = []
        for dim in env["dimensions"]:
            name = dim["name"]
            short = name.split("_")[0]
            _score_parts.append(f"{short} {feedback.get(name, 0)}")
        _scores_str = " | ".join(_score_parts)
        prompt = f"""You are revising a tweet for @{author} on X. The previous version wasn't good enough.

Current tweet:
"{previous_tweet}"

PostProphet feedback (expert editorial guidance):
- Verdict: {verdict} ({feedback['probability']:.0%} chance of hitting 2x median)
- Scores (0-10): {_scores_str}
- Reasoning: {feedback['reasoning']}
- Pattern analysis: {pattern}
- Suggestion: {feedback['suggestions']}
{ref_section}
Rewrite the tweet. Do NOT just tweak words — change the approach entirely if needed.

If the feedback says "open with a bold claim under 15 words," your tweet MUST start with a bold claim under 15 words. Not approximately. Exactly.
If the feedback says "drop hashtags," there must be zero hashtags.
If the feedback says "match the structure of their best tweet," look at the best tweet above and use the same opening style, similar length, same tone.

Rules:
- Maximum 280 characters
- No exclamation marks, no hashtags, no rhetorical questions unless their best tweet uses them
- Be concrete and specific, not vague
- One tweet only, no threads
- Return only the tweet text, nothing else"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a social media content writer. You write tweets that perform. You take editorial feedback seriously and make real changes, not cosmetic ones. Return only the tweet text, never commentary."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    
    text = resp.choices[0].message.content.strip().strip('"').strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="PostProphet Craft — iterative tweet improvement")
    parser.add_argument("--author", required=True, help="X handle (without @)")
    parser.add_argument("--topic", required=True, help="What the tweet should be about")
    parser.add_argument("--target-confidence", type=float, default=0.80, help="Stop when probability reaches this (default: 0.80)")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max revision rounds (default: 5)")
    args = parser.parse_args()
    
    author = args.author.lstrip("@")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    
    print()
    print("=" * 60)
    print(f"  PostProphet Craft")
    print(f"  Author: @{author}")
    print(f"  Topic: {args.topic}")
    print(f"  Target: {args.target_confidence:.0%} confidence | Max iterations: {args.max_iterations}")
    print("=" * 60)
    print()
    
    # Fetch author's reference tweets once (reused across all iterations)
    print("  Fetching author reference tweets...")
    hit_tweet, miss_tweet = fetch_author_examples(author)
    if hit_tweet:
        print(f"  Hit:  \"{hit_tweet['text'][:60]}...\" ({hit_tweet['impressions']:,} impressions)")
    if miss_tweet:
        print(f"  Miss: \"{miss_tweet['text'][:60]}...\" ({miss_tweet['impressions']:,} impressions)")
    print()
    
    current_tweet = None
    feedback = None
    prob = 0
    i = 0
    
    for i in range(1, args.max_iterations + 1):
        print(f"  ── Iteration {i}/{args.max_iterations} ──")
        print()
        
        # Draft or revise
        current_tweet = draft_tweet(client, args.topic, author, hit_tweet, miss_tweet, current_tweet, feedback)
        current_tweet = current_tweet.strip().strip('"').strip()
        
        print(f"  Tweet: {current_tweet}")
        print(f"  ({len(current_tweet)} chars)")
        print()
        
        # Get PostProphet prediction
        result = predict_tweet(
            text=current_tweet,
            author_username=author,
        )
        
        prob = result.get("probability", 0)
        hook = result.get("hook_strength", 0)
        reasoning = result.get("reasoning", "")
        suggestions = result.get("suggestions", "")
        pattern = result.get("pattern_analysis", "")
        
        # Build scores dynamically from environment config
        from postprophet import load_environment
        env = load_environment()
        score_parts = []
        for dim in env["dimensions"]:
            name = dim["name"]
            short = name.split("_")[0]  # First word as label
            val = result.get(name, 0)
            score_parts.append(f"{short} {val}")
        scores_str = " | ".join(score_parts)
        print(f"  Scores: {scores_str}")
        verdict = "YES" if prob >= 0.5 else "NO"
        print(f"  Prediction: {verdict} ({prob:.0%})")
        print(f"  Reasoning: {reasoning}")
        if pattern:
            print(f"  Pattern: {pattern}")
        if suggestions:
            print(f"  Suggestion: {suggestions}")
        print()
        
        # Prepare feedback for next iteration (pass through all scores dynamically)
        feedback = {
            "probability": prob,
            "reasoning": reasoning,
            "suggestions": suggestions,
            "pattern_analysis": pattern,
        }
        # Pass through all dimension scores from the result
        for dim in env["dimensions"]:
            feedback[dim["name"]] = result.get(dim["name"], 0)
        
        if prob >= args.target_confidence:
            print(f"  ✅ Target confidence reached ({prob:.0%} ≥ {args.target_confidence:.0%})")
            break
        
        if i < args.max_iterations:
            print(f"  → Revising...")
            print()
    
    print()
    print("=" * 60)
    print(f"  FINAL TWEET")
    print(f"  ──────────")
    print(f"  {current_tweet}")
    print(f"  ({len(current_tweet)} chars)")
    verdict = "YES" if prob >= 0.5 else "NO"
    print(f"  Final: {verdict} ({prob:.0%})")
    print(f"  Iterations: {i}")
    print("=" * 60)
    print()
    
    if prob < args.target_confidence:
        print(f"  ⚠️  Did not reach {args.target_confidence:.0%} confidence in {args.max_iterations} iterations.")
        print(f"  Best attempt above. Consider adjusting the topic or target.")
        print()


if __name__ == "__main__":
    main()
