"""Guidance eval with FIXED NEUTRAL SCORER.

The approach only provides feedback. A fixed content agent revises.
A fixed cosine scorer measures whether the revision moved closer to hits.
The approach never touches the scoring. No self-grading.

Flow:
1. Approach gives feedback on tweet (only thing miner controls)
2. Fixed content agent revises based on feedback (same for all)
3. Fixed cosine scorer measures: did revision get closer to author's hits?
4. Guidance = cosine(revised, hits) - cosine(original, hits)

Usage:
  python guidance_eval.py --approach cosine_with_feedback
  python guidance_eval.py  (default rubric)
"""
import json, time, os, sys, hashlib, argparse, importlib.util, math
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--approach", default=None)
parser.add_argument("--limit", type=int, default=50)
args = parser.parse_args()

# Load custom approach
custom_approach = None
if args.approach:
    approach_path = os.path.join(os.path.dirname(__file__), "approaches", f"{args.approach}.py")
    if not os.path.exists(approach_path):
        print(f"Approach not found: {approach_path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(args.approach, approach_path)
    custom_approach = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_approach)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

from postprophet import EVAL_FILE, OPENAI_MODEL, predict

# ── Fixed components (same for all approaches) ──────────────────

FIXED_CONTENT_PROMPT = """You are a tweet revision agent. Revise the tweet to address the feedback.

Rules:
- Under 280 characters
- Same topic and intent
- Address the specific feedback
- Do NOT copy the author's existing tweets
- Return ONLY the revised tweet text

Author: @{author}
Original tweet: {text}
Feedback: {feedback}

Revised tweet:"""

def fixed_embed(text: str) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text[:8000])
    return resp.data[0].embedding

def fixed_cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0

def fixed_score(text: str, author_tweets: list, target: int) -> float:
    """Fixed neutral scorer: cosine similarity to hits minus similarity to misses.
    
    Higher = closer to hits, further from misses.
    This is the SAME for all approaches. The approach never touches this.
    """
    hits = [t.get("text", "") for t in author_tweets 
            if t.get("impressions", t.get("public_metrics", {}).get("impression_count", 0)) >= target]
    misses = [t.get("text", "") for t in author_tweets 
              if t.get("impressions", t.get("public_metrics", {}).get("impression_count", 0)) < target]
    
    if not hits or not misses:
        return 0.0
    
    emb = fixed_embed(text)
    time.sleep(0.2)
    
    hit_sims = []
    for h in hits[:5]:
        hit_sims.append(fixed_cosine(emb, fixed_embed(h)))
        time.sleep(0.2)
    
    miss_sims = []
    for m in misses[:5]:
        miss_sims.append(fixed_cosine(emb, fixed_embed(m)))
        time.sleep(0.2)
    
    hit_avg = sum(hit_sims) / len(hit_sims) if hit_sims else 0
    miss_avg = sum(miss_sims) / len(miss_sims) if miss_sims else 0
    
    # Score: how much closer to hits than misses
    return hit_avg - miss_avg

# ── Load eval set ───────────────────────────────────────────────

with open(EVAL_FILE) as f:
    eval_items = [json.loads(l) for l in f if l.strip()]

eval_items = eval_items[:args.limit]

print(f"Guidance eval (fixed neutral scorer): {len(eval_items)} tweets")
print(f"Approach: {args.approach or 'rubric'}")
print(f"Scorer: fixed cosine (same for all approaches)")
print()

# ── Phase 1: Get feedback from approach ─────────────────────────

print("Phase 1: Getting feedback from approach...")
original_data = []

for i, item in enumerate(eval_items, 1):
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
    
    # Get feedback from approach
    feedback = ""
    if custom_approach:
        try:
            if hasattr(custom_approach, 'predict_with_feedback'):
                result = custom_approach.predict_with_feedback(context, item["target"])
                feedback = result.get("suggestions", "")
            else:
                # No feedback capability
                feedback = ""
        except Exception as e:
            feedback = ""
    else:
        # Default rubric
        os.environ["POSTPROPHET_TEMPERATURE"] = "0"
        try:
            prediction = predict(context, target=item["target"])
            feedback = prediction.get("suggestions", "")
            time.sleep(1.5)
        except Exception:
            feedback = ""
    
    original_data.append({
        "item": item,
        "context": context,
        "feedback": feedback,
    })
    
    has_fb = "yes" if feedback else "no"
    print(f"  [{i}/{len(eval_items)}] @{item['author']['username']}: feedback={has_fb}")

# ── Phase 2: Score originals with fixed scorer ─────────────────

print(f"\nPhase 2: Scoring originals with fixed cosine scorer...")
for i, od in enumerate(original_data, 1):
    od["orig_score"] = fixed_score(
        od["item"]["text"],
        od["item"].get("author_all_tweets", []),
        od["item"]["target"],
    )
    print(f"  [{i}/{len(eval_items)}] @{od['item']['author']['username']}: orig_score={od['orig_score']:.4f}")

# ── Phase 3: Revise with fixed content agent ───────────────────

print(f"\nPhase 3: Revising with fixed content agent...")

for i, od in enumerate(original_data, 1):
    item = od["item"]
    feedback = od["feedback"]
    
    if not feedback:
        od["revised_text"] = item["text"]
        od["revised_score"] = od["orig_score"]
        continue
    
    prompt = FIXED_CONTENT_PROMPT.format(
        author=item["author"]["username"],
        text=item["text"],
        feedback=feedback,
    )
    
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=280,
            temperature=0.0,
        )
        revised = resp.choices[0].message.content
        revised = revised.strip().strip('"') if revised else item["text"]
        time.sleep(0.5)
    except Exception:
        revised = item["text"]
    
    od["revised_text"] = revised
    
    # Score revised with SAME fixed scorer
    od["revised_score"] = fixed_score(
        revised,
        item.get("author_all_tweets", []),
        item["target"],
    )
    
    score_delta = od["revised_score"] - od["orig_score"]
    status = "BETTER" if score_delta > 0.001 else "WORSE" if score_delta < -0.001 else "SAME"
    print(f"  [{i}/{len(eval_items)}] @{item['author']['username']}: {od['orig_score']:.4f} -> {od['revised_score']:.4f} ({score_delta:+.4f}) {status}")

# ── Phase 4: Compute guidance score ────────────────────────────

# For each tweet: did the revision move CLOSER to hits?
# score = cosine(revised, hits) - cosine(revised, misses)
# guidance_delta = revised_score - orig_score
# 
# Positive = revision moved closer to hits (good guidance)
# Negative = revision moved away from hits (bad guidance)
# Zero = no change or no feedback

hits_data = [od for od in original_data if od["item"]["hit"]]
misses_data = [od for od in original_data if not od["item"]["hit"]]

hits_deltas = [od["revised_score"] - od["orig_score"] for od in hits_data]
misses_deltas = [od["revised_score"] - od["orig_score"] for od in misses_data]

avg_hit_delta = sum(hits_deltas) / len(hits_deltas) if hits_deltas else 0
avg_miss_delta = sum(misses_deltas) / len(misses_deltas) if misses_deltas else 0

# Guidance score: did feedback move ALL tweets toward hits?
# Not just hits up and misses down — did the revision get closer
# to the hit pattern regardless of the original outcome?
avg_overall_delta = sum(od["revised_score"] - od["orig_score"] for od in original_data) / len(original_data)

# But we also want: did it move hits MORE than misses?
# If feedback says "add specificity" and both hits and misses improve
# equally, the feedback isn't discriminating — it's just making
# everything slightly better (which is trivial).
# Real guidance: hits improve more than misses.
discrimination = avg_hit_delta - avg_miss_delta

print(f"\n{'─' * 50}")
print(f"APPROACH: {args.approach or 'rubric'}")
print(f"SCORER: fixed cosine (neutral)")
print(f"{'─' * 50}")
print(f"Overall score lift:    {avg_overall_delta:+.4f}")
print(f"  (positive = revisions closer to hits)")
print(f"")
print(f"Hits ({len(hits_data)}):")
print(f"  Avg score change:    {avg_hit_delta:+.4f}")
print(f"Misses ({len(misses_data)}):")
print(f"  Avg score change:    {avg_miss_delta:+.4f}")
print(f"")
print(f"GUIDANCE SCORE:        {avg_overall_delta:+.4f}")
print(f"  (positive = feedback moves tweets toward hit pattern)")
print(f"")
print(f"DISCRIMINATION:        {discrimination:+.4f}")
print(f"  (positive = hits improve more than misses)")
print(f"  (zero = feedback doesn't discriminate, just noise)")
print(f"{'─' * 50}")

# Save
results = {
    "approach": args.approach or "rubric",
    "scorer": "fixed_cosine",
    "guidance_score": round(avg_overall_delta, 4),
    "discrimination": round(discrimination, 4),
    "hits_delta": round(avg_hit_delta, 4),
    "misses_delta": round(avg_miss_delta, 4),
    "per_tweet": [
        {
            "author": od["item"]["author"]["username"],
            "hit": od["item"]["hit"],
            "orig_score": round(od["orig_score"], 4),
            "revised_score": round(od["revised_score"], 4),
            "score_delta": round(od["revised_score"] - od["orig_score"], 4),
            "orig_text": od["item"]["text"][:100],
            "revised_text": od["revised_text"][:100],
            "feedback": od["feedback"][:200],
        }
        for od in original_data
    ],
}

results_file = os.path.join(os.path.dirname(__file__), "data", "guidance_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {results_file}")
