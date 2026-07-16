"""
PostProphet Craft — Iterative tweet improvement loop.

Drafts a tweet, gets PostProphet's prediction + reasoning, revises based on
feedback, repeats until target confidence or max iterations.

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
from postprophet import predict_tweet, OPENAI_MODEL


def draft_tweet(client, topic, author, previous_tweet=None, feedback=None):
    """Draft or revise a tweet using an LLM guided by PostProphet's feedback."""
    
    if previous_tweet is None:
        # First draft — just write the tweet
        prompt = f"""You are a social media content writer for @{author} on X.

Write a single tweet about: {topic}

Rules:
- Maximum 280 characters
- No hashtags unless they're organic to the content
- No "AI" or "Agent" buzzwords unless the topic demands it
- Match the voice of a crypto/AI infrastructure account
- One tweet only, no threads
- Be bold and specific, not vague

Return only the tweet text, nothing else."""
    else:
        # Revision — use PostProphet's feedback to improve
        prompt = f"""You are revising a tweet for @{author} on X.

Current tweet:
"{previous_tweet}"

PostProphet feedback (treat this as expert editorial guidance):
- Probability of beating their p75: {feedback['probability']:.0%}
- Reasoning: {feedback['reasoning']}
- Suggestion: {feedback['suggestions']}

Revise the tweet to address the feedback. Keep what's working, fix what isn't.

Rules:
- Maximum 280 characters
- Don't just add words to please the feedback — genuinely improve the tweet
- If the feedback says it's already strong, make targeted refinements
- One tweet only, no threads
- Return only the tweet text, nothing else"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a social media content writer. You write tweets that perform. Return only the tweet text, never commentary."},
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
    
    current_tweet = None
    feedback = None
    
    for i in range(1, args.max_iterations + 1):
        print(f"  ── Iteration {i}/{args.max_iterations} ──")
        print()
        
        # Draft or revise
        current_tweet = draft_tweet(client, args.topic, author, current_tweet, feedback)
        
        # Clean up any quotes or extra whitespace
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
        reasoning = result.get("reasoning", "")
        suggestions = result.get("suggestions", "")
        
        print(f"  Probability: {prob:.0%}")
        print(f"  Reasoning: {reasoning}")
        if suggestions:
            print(f"  Suggestion: {suggestions}")
        print()
        
        # Prepare feedback for next iteration
        feedback = {
            "probability": prob,
            "reasoning": reasoning,
            "suggestions": suggestions,
        }
        
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
    print(f"  Probability: {prob:.0%}")
    print(f"  Iterations: {i}")
    print("=" * 60)
    print()
    
    if prob < args.target_confidence:
        print(f"  ⚠️  Did not reach {args.target_confidence:.0%} confidence in {args.max_iterations} iterations.")
        print(f"  Best attempt above. Consider adjusting the topic or target.")
        print()


if __name__ == "__main__":
    main()
