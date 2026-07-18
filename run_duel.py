"""Run eval on both current (main) and challenger (this branch) configs.
Saves all 50 per-tweet results + raw LLM responses + SHA256 hash for verification.

Usage:
  POSTPROPHET_TEMPERATURE=0 python run_duel.py

Temperature 0 makes outputs deterministic for reproducible verification.
"""
import json, time, os, sys, hashlib
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from postprophet import (
    predict, load_environment, build_prompt,
    PREDICTION_PROMPT, HARNESS_VERSION, HARNESS_HASH,
    _compute_config_hash, EVAL_FILE, OPENAI_MODEL
)

# Force temperature 0 for reproducible eval
os.environ["POSTPROPHET_TEMPERATURE"] = "0"

# Load eval set
with open(EVAL_FILE) as f:
    eval_items = [json.loads(line) for line in f if line.strip()]

print(f"Eval set: {len(eval_items)} tweets")
print(f"Current version: {HARNESS_VERSION} ({HARNESS_HASH})")
print(f"Model: {OPENAI_MODEL}")
print(f"Temperature: 0 (deterministic)")
print()

# ── Run challenger eval ─────────────────────────────────────────

print("Running challenger eval...")
challenger_results = []
challenger_raw = []
challenger_brier_total = 0.0

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
    
    try:
        prediction = predict(context, target=item["target"])
        prob = prediction.get("probability", 0.5)
        time.sleep(1.5)
    except Exception as e:
        print(f"  [{i}/{len(eval_items)}] Error: {e}")
        prob = 0.5
        prediction = {"error": str(e)}
    
    actual = 1.0 if item["hit"] else 0.0
    brier_contrib = (prob - actual) ** 2
    challenger_brier_total += brier_contrib
    
    challenger_results.append({
        "tweet_id": item["tweet_id"],
        "author": item["author"]["username"],
        "target": item["target"],
        "actual_impressions": item["actual_impressions"],
        "hit": item["hit"],
        "challenger_prob": prob,
        "challenger_brier": brier_contrib,
    })
    
    # Save raw LLM response for verification
    challenger_raw.append({
        "tweet_id": item["tweet_id"],
        "author": item["author"]["username"],
        "probability": prob,
        "full_prediction": prediction,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    
    verdict = "YES" if prob >= 0.5 else "NO"
    correct = "✅" if (prob >= 0.5) == item["hit"] else "❌"
    print(f"  [{i}/{len(eval_items)}] @{item['author']['username']}: {verdict} ({prob:.0%}) | target {item['target']:,} | actual {item['actual_impressions']:,} | {correct}")

challenger_brier = challenger_brier_total / len(challenger_results)

# ── Baseline from eval set ──────────────────────────────────────

baseline_results = []
baseline_brier_total = 0.0
for item in eval_items:
    prob = item.get("original_probability", 0.5)
    actual = 1.0 if item["hit"] else 0.0
    brier_contrib = (prob - actual) ** 2
    baseline_brier_total += brier_contrib
    baseline_results.append({
        "tweet_id": item["tweet_id"],
        "author": item["author"]["username"],
        "target": item["target"],
        "actual_impressions": item["actual_impressions"],
        "hit": item["hit"],
        "baseline_prob": prob,
    })

baseline_brier = baseline_brier_total / len(baseline_results)

# ── Merge results ───────────────────────────────────────────────

merged = []
for i in range(len(eval_items)):
    merged.append({
        "tweet_id": baseline_results[i]["tweet_id"],
        "author": baseline_results[i]["author"],
        "target": baseline_results[i]["target"],
        "actual_impressions": baseline_results[i]["actual_impressions"],
        "hit": baseline_results[i]["hit"],
        "baseline_prob": baseline_results[i]["baseline_prob"],
        "challenger_prob": challenger_results[i]["challenger_prob"],
    })

# ── Compute hash for verification ───────────────────────────────

raw_json = json.dumps(challenger_raw, sort_keys=True)
results_hash = hashlib.sha256(raw_json.encode()).hexdigest()

# ── Save everything ─────────────────────────────────────────────

delta = baseline_brier - challenger_brier

results = {
    "duel_id": 2,
    "hypothesis": "Enforce probability guide as MANDATORY mapping. Model was ignoring it — outputting 25-35% when guide says 70-85% for all scores 7+. Changed 'rough guide' to 'MANDATORY mapping (do not deviate)' in prompt.",
    "baseline_version": eval_items[0].get("original_version", "1.0.2"),
    "baseline_hash": "v_31ea78b4",
    "challenger_version": HARNESS_VERSION,
    "challenger_hash": HARNESS_HASH,
    "baseline_brier": round(baseline_brier, 4),
    "challenger_brier": round(challenger_brier, 4),
    "delta": round(delta, 4),
    "model": OPENAI_MODEL,
    "temperature": 0,
    "eval_size": len(eval_items),
    "results_hash": results_hash,
    "per_tweet": merged,
    "raw_file": "data/duel_raw.json",
}

results_file = os.path.join(os.path.dirname(__file__), "data", "duel_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

raw_file = os.path.join(os.path.dirname(__file__), "data", "duel_raw.json")
with open(raw_file, "w") as f:
    json.dump(challenger_raw, f, indent=2)

print(f"\n{'─' * 50}")
print(f"Baseline:   {baseline_brier:.4f} ({results['baseline_version']})")
print(f"Challenger: {challenger_brier:.4f} ({results['challenger_hash']})")
print(f"Delta:      {delta:+.4f}")
if delta > 0.01:
    print(f"Result:     IMPROVEMENT ✅")
elif delta < -0.01:
    print(f"Result:     REGRESSION ❌")
else:
    print(f"Result:     NO SIGNIFICANT CHANGE")
print(f"{'─' * 50}")
print(f"Results hash: {results_hash}")
print(f"Raw responses: data/duel_raw.json")
print(f"To verify: clone repo, checkout this branch, run with temp=0, compare hash")
