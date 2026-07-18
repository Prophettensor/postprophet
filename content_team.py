#!/usr/bin/env python3
"""PostProphet — Content Team Orchestrator

Runs the full content loop:
1. Pull latest market brief (from market_research.py)
2. Write dev brief (what happened, why it matters)
3. Content agent drafts tweets from both briefs
4. PostProphet evaluates each draft
5. Content agent revises based on feedback
6. Loop until confidence target, plateau, or max iterations
7. Output final draft for approval

Usage:
  python content_team.py [--author postprophet] [--target-confidence 0.40] [--max-iterations 5]

No auto-posting yet — requires X API write credentials.
Outputs final draft for human approval.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

X_BEARER_TOKEN = os.environ.get("X_BEARER_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("POSTPROPHET_MODEL", "gpt-4o-mini")

MARKET_BRIEFS_DIR = os.path.join(os.path.dirname(__file__), "data", "market_briefs")


def get_latest_market_brief():
    """Pull the most recent market brief."""
    if not os.path.exists(MARKET_BRIEFS_DIR):
        return None
    
    briefs = sorted([f for f in os.listdir(MARKET_BRIEFS_DIR) if f.endswith(".md")], reverse=True)
    if not briefs:
        return None
    
    brief_file = os.path.join(MARKET_BRIEFS_DIR, briefs[0])
    with open(brief_file) as f:
        content = f.read()
    
    # Strip the header
    lines = content.split("\n")
    brief = "\n".join(lines[2:]) if len(lines) > 2 else content
    return brief.strip()


def get_dev_brief():
    """Write a dev brief — what happened, why it matters.
    
    This is where the main agent (me) would normally write context.
    For now, pulls from git log + dashboard stats.
    """
    import subprocess
    
    # Get git log
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-15"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        git_log = result.stdout.strip()
    except Exception:
        git_log = ""
    
    # Get current stats
    stats = {}
    preds_file = os.path.join(os.path.dirname(__file__), "data", "predictions.jsonl")
    if os.path.exists(preds_file):
        with open(preds_file) as f:
            preds = [json.loads(line) for line in f if line.strip()]
        resolved = [p for p in preds if p.get("resolved")]
        hits = sum(1 for p in resolved if p["actual_impressions"] >= p["target"])
        brier = sum(
            (p["prediction"]["probability"] - (1.0 if p["actual_impressions"] >= p["target"] else 0.0)) ** 2
            for p in resolved
        ) / len(resolved) if resolved else 0
        stats = {
            "total_resolved": len(resolved),
            "hits": hits,
            "hit_rate": f"{hits/len(resolved)*100:.0f}%" if resolved else "0%",
            "brier": f"{brier:.4f}",
            "accounts": len(set(p["context"]["author"]["username"] for p in preds)),
        }
    
    return f"""DEV BRIEF — {datetime.now(timezone.utc).strftime("%Y-%m-%d")}

Recent development:
{git_log}

Current stats:
  Predictions resolved: {stats.get('total_resolved', 0)}
  Hit rate: {stats.get('hit_rate', '0%')}
  Brier score: {stats.get('brier', '0')}
  Tracked accounts: {stats.get('accounts', 0)}

What's interesting right now:
- Quote tweet analysis built — model now sees full context of quoted tweets
- Thread detection from same API batch (zero extra cost)
- Self-learning calibration: model sees its own track record and adjusts
- Link penalty fixed: X-internal links (video, quote tweets) no longer penalized
- Environment config: scoring dimensions are now platform-agnostic (JSON config)
- Semver versioning: automatic, hash-based, for trustless PR evaluation
- Account discovery: quoted accounts auto-suggested for ecosystem growth
- Live eval dashboard: buckzz7.github.io/postprophet/eval.html"""


def draft_tweets(market_brief, dev_brief, author, count=3):
    """Content agent drafts tweets from market + dev briefs."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Get author's recent tweets for voice matching
    from postprophet import get_user_by_username, get_author_recent_tweets, compute_author_baseline
    
    user = get_user_by_username(author)
    if not user:
        print(f"  ⚠️  Could not find @{author} — writing without voice match")
        ref_section = ""
    else:
        user_id = user["id"]
        recent = get_author_recent_tweets(user_id, max_results=10)
        if recent:
            baseline = compute_author_baseline(recent)
            sorted_tweets = sorted(recent, key=lambda t: t.get("public_metrics", {}).get("impression_count", 0), reverse=True)
            best = sorted_tweets[0] if sorted_tweets else None
            if best:
                ref_section = f"\nAUTHOR'S BEST TWEET ({best.get('public_metrics', {}).get('impression_count', 0):,} impressions):\n{best.get('text', '')}\n\nMatch this voice, length, and structure."
            else:
                ref_section = ""
        else:
            ref_section = ""
    
    prompt = f"""You are a content writer for @{author} on X. You write tweets that perform.

MARKET BRIEF:
{market_brief or '(no market brief available)'}

DEV BRIEF:
{dev_brief}
{ref_section}

Write {count} tweet drafts. Each should:
- Connect a dev milestone to a market trend or pain point
- Be specific, not vague ("Brier 0.20" not "we're getting better")
- Sound like @{author}, not a corporate account
- Max 280 characters
- No hashtags unless the best tweet uses them
- No "excited to announce" or "thrilled to share"
- One idea per tweet, not a list

Return only the tweets, one per line, numbered 1-N."""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.8,
    )
    
    text = resp.choices[0].message.content
    # Parse numbered tweets
    drafts = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove number prefix
            draft = line.split(".", 1)[1].strip() if "." in line[:3] else line
            if draft:
                drafts.append(draft)
    
    return drafts


def evaluate_and_revise(drafts, author, target_confidence, max_iterations):
    """Run PostProphet evaluation + content agent revision loop.
    
    Takes each draft, evaluates with PostProphet, revises based on feedback,
    loops until confidence target, plateau, or max iterations.
    Returns the best draft + its prediction.
    """
    from postprophet import predict_tweet
    
    results = []
    
    for i, draft in enumerate(drafts, 1):
        print(f"\n  Draft {i}: \"{draft[:60]}...\"")
        
        best_draft = draft
        best_prob = 0
        best_prediction = None
        plateau_count = 0
        iteration = 0
        
        for iteration in range(1, max_iterations + 1):
            # Predict
            result = predict_tweet(text=best_draft, author_username=author)
            prediction = result.get("prediction", {})
            prob = prediction.get("probability", 0)
            
            verdict = "YES" if prob >= 0.5 else "NO"
            reasoning = prediction.get("reasoning", "")[:80]
            feedback = prediction.get("suggestions", "")[:80]
            
            print(f"    [{iteration}] {verdict} ({prob:.0%}) | {reasoning}")
            if feedback:
                print(f"         Feedback: {feedback}")
            
            if prob > best_prob:
                best_prob = prob
                best_prediction = prediction
                plateau_count = 0
            else:
                plateau_count += 1
            
            # Check stop conditions
            if prob >= target_confidence:
                print(f"    ✅ Target confidence reached ({prob:.0%} ≥ {target_confidence:.0%})")
                break
            
            if plateau_count >= 2:
                print(f"    ⏸️  Plateau — confidence not improving")
                break
            
            if iteration >= max_iterations:
                print(f"    ⏹️  Max iterations reached")
                break
            
            # Revise
            revised = revise_tweet(best_draft, prediction, author)
            if revised and revised != best_draft:
                best_draft = revised
                print(f"    → Revised: \"{best_draft[:60]}...\"")
            else:
                print(f"    ⏸️  No revision produced")
                break
        
        results.append({
            "draft": best_draft,
            "probability": best_prob,
            "prediction": best_prediction,
            "iterations": iteration,
        })
    
    # Sort by probability
    results.sort(key=lambda r: r["probability"], reverse=True)
    return results


def revise_tweet(current_tweet, feedback, author):
    """Content agent revises a tweet based on PostProphet feedback."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    reasoning = feedback.get("reasoning", "")
    suggestions = feedback.get("suggestions", "")
    scores = {}
    for key in ["hook_strength", "specificity", "emotional_trigger",
                "reply_inducement", "bookmark_worthiness",
                "structure_readability", "clarity_density", "link_penalty_risk"]:
        if key in feedback:
            scores[key] = feedback[key]
    
    scores_str = " | ".join(f"{k.split('_')[0]} {v}" for k, v in scores.items())
    
    prompt = f"""You are revising a tweet for @{author} on X. The previous version wasn't good enough.

Current tweet:
"{current_tweet}"

PostProphet feedback:
- Verdict: {"YES" if feedback.get("probability", 0) >= 0.5 else "NO"} ({feedback.get("probability", 0):.0%} chance)
- Scores (0-10): {scores_str}
- Reasoning: {reasoning}
- Feedback: {suggestions}

Revise the tweet based on the feedback. Keep what works, fix what's weak.
Max 280 characters. Return only the revised tweet text, nothing else."""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=280,
        temperature=0.7,
    )
    
    revised = resp.choices[0].message.content
    if revised:
        revised = revised.strip().strip('"')
        return revised
    return None


def main():
    parser = argparse.ArgumentParser(description="PostProphet Content Team")
    parser.add_argument("--author", default="postprophet", help="X handle to post as")
    parser.add_argument("--drafts", type=int, default=3, help="Number of initial drafts")
    args = parser.parse_args()
    
    # Baked in — no config knobs
    target_confidence = 0.40  # Realistic for a new account
    max_iterations = 5  # Hard cap to limit LLM cost
    
    print("=" * 60)
    print("  PostProphet — Content Team")
    print("=" * 60 + "\n")
    
    # 1. Get market brief
    print("  1. Loading market brief...")
    market_brief = get_latest_market_brief()
    if market_brief:
        print(f"     + {len(market_brief)} chars")
    else:
        print("     No market brief found. Run market_research.py first.")
        market_brief = ""
    
    # 2. Get dev brief
    print("  2. Writing dev brief...")
    dev_brief = get_dev_brief()
    print(f"     + {len(dev_brief)} chars")
    
    # 3. Draft tweets
    print(f"\n  3. Drafting {args.drafts} tweets...")
    drafts = draft_tweets(market_brief, dev_brief, args.author, count=args.drafts)
    for i, d in enumerate(drafts, 1):
        print(f"     Draft {i}: \"{d[:60]}...\" ({len(d)} chars)")
    
    if not drafts:
        print("\n  No drafts generated. Check API credentials.")
        return
    
    # 4. Evaluate + revise loop
    print(f"\n  4. Evaluating + revising (target: {target_confidence:.0%}, max: {max_iterations} iterations)...")
    results = evaluate_and_revise(drafts, args.author, target_confidence, max_iterations)
    
    # 5. Output winner
    print("\n" + "=" * 60)
    print("  RESULTS (sorted by confidence)")
    print("=" * 60 + "\n")
    
    for i, r in enumerate(results, 1):
        verdict = "YES" if r["probability"] >= 0.5 else "NO"
        print(f"  #{i} - {verdict} ({r['probability']:.0%}) after {r['iterations']} iterations")
        print(f"  Tweet: \"{r['draft']}\"")
        print(f"  ({len(r['draft'])} chars)")
        if r["prediction"]:
            print(f"  Reasoning: {r['prediction'].get('reasoning', '')[:100]}")
        print()
    
    best = results[0]
    print("=" * 60)
    print(f"  WINNER: {best['probability']:.0%} confidence")
    print(f"  \"{best['draft']}\"")
    print("=" * 60)
    print("\n  Awaiting approval. Set X API write credentials to auto-post.")


if __name__ == "__main__":
    main()
