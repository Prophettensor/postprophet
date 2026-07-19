"""Content Loop: daily content engine.

Connects market research -> content agent -> PostProphet eval -> ranked output.
Runs after the tracking cron. Outputs top tweets for human review and posting.

Usage:
  python content_loop.py              # Full loop
  python content_loop.py --dry-run    # Skip posting (default, no write API needed)
"""
import json, time, os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from postprophet import predict, load_environment, build_prompt

# ── Config ──────────────────────────────────────────────────────

MIN_PROBABILITY = 0.30  # Only show drafts above this probability
MAX_DRAFTS = 5           # Show top N drafts
REVISION_TARGET = 0.40   # Keep revising until this probability
REVISION_PLATEAU = 2     # Stop after N iterations without improvement
REVISION_MAX = 5         # Max revision iterations

# ── Step 1: Market Research ─────────────────────────────────────

def run_market_research():
    """Get market brief from market_research.py if it exists."""
    brief_file = 'data/market_briefs/latest.json'
    if os.path.exists(brief_file):
        with open(brief_file) as f:
            return json.load(f)
    return None

# ── Step 2: Content Agent ───────────────────────────────────────

def draft_tweets(market_brief, count=3):
    """Draft tweets based on market brief + author history.
    
    Uses content_team.py if available, otherwise falls back to
    a simpler direct LLM draft.
    """
    try:
        from content_team import ContentAgent
        agent = ContentAgent()
        return agent.draft(market_brief, count=count)
    except Exception as e:
        print(f"  Content agent not available ({e}), using fallback")
        return fallback_draft(market_brief, count=count)

def fallback_draft(market_brief, count=3):
    """Fallback: direct LLM draft using market brief."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    
    brief_text = ""
    if market_brief:
        brief_text = f"""
Market context:
- Hot themes: {market_brief.get('hot_themes', [])}
- Key voices: {market_brief.get('key_voices', [])}
- Trending: {market_brief.get('trending', [])}
"""
    
    prompt = f"""You are drafting tweets for @PostProphet, a Bittensor content account.

{brief_text}

Draft {count} tweets about Bittensor/AI topics that would perform well.
Each tweet should:
- Have a specific, concrete hook (not "excited to share")
- Reference real projects, numbers, or announcements
- Be under 280 characters
- Feel like insider knowledge, not generic marketing

Return JSON: {{"tweets": ["tweet1", "tweet2", "tweet3"]}}"""
    
    resp = client.chat.completions.create(
        model=os.environ.get("POSTPROPHET_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={"type": "json_object"},
    )
    
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        return [{"text": t, "source": "llm_draft"} for t in data.get("tweets", [])]
    except:
        return [{"text": content.strip(), "source": "llm_draft"}]

# ── Step 3: PostProphet Eval ────────────────────────────────────

def evaluate_draft(text, author="postprophet"):
    """Evaluate a draft tweet with PostProphet."""
    context = {
        "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planned_post_time": datetime.now(timezone.utc).isoformat(),
        "platform": "x",
        "author": {
            "username": author,
            "followers": 0,
            "following": 0,
            "tweet_count": 0,
        },
        "author_baseline": {
            "median_impressions": 500,
            "median_likes": 10,
            "median_retweets": 1,
            "sample_size": 10,
        },
        "author_all_tweets": [],
        "author_recent_top_tweets": [],
        "media": {"has_media": False, "media_types": [], "descriptions": []},
        "entities": {},
        "referenced_tweets": [],
        "trending_topics": [],
        "has_x_data": False,
    }
    
    try:
        result = predict(context, target=1000)
        return result
    except Exception as e:
        return {"probability": 0.5, "error": str(e), "suggestions": ""}

# ── Step 4: Revision Loop ───────────────────────────────────────

def revise_draft(text, result, author="postprophet"):
    """Revise a draft based on PostProphet feedback."""
    feedback = result.get("suggestions", "")
    if not feedback:
        return text, result
    
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    
    prompt = f"""Revise this tweet based on feedback. Keep it under 280 characters.

Original: {text}
Feedback: {feedback}

Revised tweet:"""
    
    resp = client.chat.completions.create(
        model=os.environ.get("POSTPROPHET_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=280,
    )
    
    revised = resp.choices[0].message.content
    if revised:
        revised = revised.strip().strip('"')
    
    # Re-evaluate
    new_result = evaluate_draft(revised, author)
    return revised, new_result

# ── Main Loop ───────────────────────────────────────────────────

def run_loop():
    print("=" * 60)
    print("CONTENT LOOP")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    
    # Step 1: Market research
    print("\n[1/4] Market research...")
    market_brief = run_market_research()
    if market_brief:
        print(f"  Hot themes: {market_brief.get('hot_themes', [])}")
    else:
        print("  No market brief available")
    
    # Step 2: Draft tweets
    print(f"\n[2/4] Drafting tweets...")
    drafts = draft_tweets(market_brief, count=3)
    print(f"  Drafted {len(drafts)} tweets")
    
    # Step 3: Evaluate + revise
    print(f"\n[3/4] Evaluating and revising...")
    evaluated = []
    
    for i, draft in enumerate(drafts, 1):
        text = draft.get("text", "")
        print(f"\n  Draft {i}: {text[:60]}...")
        
        # Initial eval
        result = evaluate_draft(text)
        prob = result.get("probability", 0.5)
        print(f"    Probability: {prob:.0%}")
        
        # Revision loop
        best_text = text
        best_result = result
        best_prob = prob
        plateau = 0
        iteration = 0
        
        while best_prob < REVISION_TARGET and plateau < REVISION_PLATEAU and iteration < REVISION_MAX:
            iteration += 1
            revised, new_result = revise_draft(best_text, best_result)
            new_prob = new_result.get("probability", 0.5)
            
            if new_prob > best_prob:
                improvement = new_prob - best_prob
                best_text = revised
                best_result = new_result
                best_prob = new_prob
                plateau = 0
                print(f"    Rev {iteration}: {new_prob:.0%} (+{improvement:.0%})")
            else:
                plateau += 1
                print(f"    Rev {iteration}: {new_prob:.0%} (plateau {plateau})")
        
        evaluated.append({
            "original_text": text,
            "final_text": best_text,
            "probability": best_prob,
            "feedback": best_result.get("suggestions", ""),
            "iterations": iteration,
            "improved": best_text != text,
        })
    
    # Step 4: Rank and output
    print(f"\n[4/4] Ranking...")
    evaluated.sort(key=lambda x: x["probability"], reverse=True)
    
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_brief": market_brief is not None,
        "drafts": evaluated[:MAX_DRAFTS],
    }
    
    # Save
    output_file = 'data/content_loop_output.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Display
    print(f"\n{'='*60}")
    print(f"TOP DRAFTS FOR REVIEW")
    print(f"{'='*60}")
    
    for i, d in enumerate(evaluated[:MAX_DRAFTS], 1):
        marker = "🟢" if d["probability"] >= REVISION_TARGET else "🟡" if d["probability"] >= MIN_PROBABILITY else "🔴"
        print(f"\n{marker} #{i} ({d['probability']:.0%})")
        print(f"  {d['final_text']}")
        if d["improved"]:
            print(f"  (revised {d['iterations']} times)")
        if d["feedback"]:
            print(f"  Feedback: {d['feedback'][:120]}")
    
    print(f"\n{'='*60}")
    print(f"Output saved: {output_file}")
    print(f"Review and post manually.")
    print(f"{'='*60}")
    
    return output

if __name__ == '__main__':
    run_loop()
