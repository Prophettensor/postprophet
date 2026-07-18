"""Run eval on both current (main) and challenger (this branch) configs.
Saves all 50 per-tweet results to data/duel_results.json for the dashboard.
"""
import json, time, os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENAI_API_KEY", "")

from postprophet import (
    predict, run_eval, load_environment, build_prompt,
    PREDICTION_PROMPT, HARNESS_VERSION, HARNESS_HASH,
    _compute_config_hash, EVAL_FILE, OPENAI_MODEL
)

# Load eval set
with open(EVAL_FILE) as f:
    eval_items = [json.loads(line) for line in f if line.strip()]

print(f"Eval set: {len(eval_items)} tweets")
print(f"Current version: {HARNESS_VERSION} ({HARNESS_HASH})")
print()

# ── Run challenger eval (current branch config) ───────────────

print("Running challenger eval (this branch)...")
challenger_results = []
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
        prediction = {}
    
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
        "challenger_scores": {k: v for k, v in prediction.items() if isinstance(v, (int, float))},
    })
    
    verdict = "YES" if prob >= 0.5 else "NO"
    correct = "✅" if (prob >= 0.5) == item["hit"] else "❌"
    print(f"  [{i}/{len(eval_items)}] @{item['author']['username']}: {verdict} ({prob:.0%}) | target {item['target']:,} | actual {item['actual_impressions']:,} | {correct}")

challenger_brier = challenger_brier_total / len(challenger_results)

# ── Get baseline from eval_set original predictions ───────────────

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
        "baseline_brier": brier_contrib,
    })

baseline_brier = baseline_brier_total / len(baseline_results)

# ── Merge results ────────────────────────────────────────────────

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

# ── Save results ─────────────────────────────────────────────────

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
    "eval_size": len(eval_items),
    "per_tweet": merged,
}

results_file = os.path.join(os.path.dirname(__file__), "data", "duel_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'─' * 50}")
print(f"Baseline:  {baseline_brier:.4f} ({results['baseline_version']})")
print(f"Challenger: {challenger_brier:.4f} ({results['challenger_hash']})")
print(f"Delta:     {delta:+.4f}")
if delta > 0.01:
    print(f"Result:    IMPROVEMENT ✅")
elif delta < -0.01:
    print(f"Result:    REGRESSION ❌")
else:
    print(f"Result:    NO SIGNIFICANT CHANGE")
print(f"{'─' * 50}")
print(f"Results saved: {results_file}")
