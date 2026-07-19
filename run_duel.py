"""Run eval on an approach (config-based or custom code-based).

Usage:
  # Config-based (LLM prompt)
  python run_duel.py
  
  # Custom approach (code in approaches/)
  python run_duel.py --approach cosine_similarity

Temperature 0 for deterministic, reproducible eval.
"""
import json, time, os, sys, hashlib, argparse, importlib.util
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["POSTPROPHET_TEMPERATURE"] = "0"

from postprophet import (
    predict, run_eval, load_environment, build_prompt,
    PREDICTION_PROMPT, HARNESS_VERSION, HARNESS_HASH,
    _compute_config_hash, EVAL_FILE, OPENAI_MODEL
)

parser = argparse.ArgumentParser()
parser.add_argument("--approach", default=None, help="Custom approach name (e.g. cosine_similarity)")
args = parser.parse_args()

# Load custom approach if specified
custom_approach = None
if args.approach:
    approach_path = os.path.join(os.path.dirname(__file__), "approaches", f"{args.approach}.py")
    if not os.path.exists(approach_path):
        print(f"Approach not found: {approach_path}")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location(args.approach, approach_path)
    custom_approach = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_approach)
    
    if not hasattr(custom_approach, "predict_probability"):
        print(f"Approach {args.approach} does not implement predict_probability()")
        sys.exit(1)
    
    print(f"Using custom approach: {args.approach}")
    
    # Security check: approach must not import postprophet
    with open(approach_path) as f:
        source = f.read()
    forbidden = ["import postprophet", "from postprophet", "eval_set", "predictions.jsonl", "data/eval"]
    for term in forbidden:
        if term in source:
            print(f"SECURITY: Approach contains forbidden term: {term}")
            sys.exit(1)
    
    print("  Security check passed")

# Load eval set
with open(EVAL_FILE) as f:
    eval_items = [json.loads(line) for line in f if line.strip()]

print(f"Eval set: {len(eval_items)} tweets")
print(f"Model: {OPENAI_MODEL}")
print(f"Temperature: 0 (deterministic)")
print(f"Approach: {args.approach or 'default (LLM config)'}")
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
    
    if custom_approach:
        # Custom approach: call predict_probability directly
        try:
            prob = custom_approach.predict_probability(context, item["target"])
            prediction = {"probability": prob, "approach": args.approach}
        except Exception as e:
            print(f"  [{i}/{len(eval_items)}] Error: {e}")
            prob = 0.5
            prediction = {"error": str(e)}
    else:
        # Default LLM approach
        try:
            prediction = predict(context, target=item["target"])
            prob = prediction.get("probability", 0.5)
            time.sleep(1.5)
        except Exception as e:
            print(f"  [{i}/{len(eval_items)}] Error: {e}")
            prob = 0.5
            prediction = {"error": str(e)}
    
    # Validate probability
    if not isinstance(prob, (int, float)) or prob < 0.0 or prob > 1.0:
        prob = max(0.0, min(1.0, float(prob))) if isinstance(prob, (int, float)) else 0.5
    
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

# ── Hash ───────────────────────────────────────────────────────

raw_json = json.dumps(challenger_raw, sort_keys=True)
results_hash = hashlib.sha256(raw_json.encode()).hexdigest()

# ── Save ───────────────────────────────────────────────────────

delta = baseline_brier - challenger_brier

approach_name = args.approach or "llm_config"

results = {
    "approach": approach_name,
    "hypothesis": f"Custom approach: {approach_name}" if args.approach else "Config-based LLM approach",
    "baseline_version": eval_items[0].get("original_version", "1.0.2"),
    "baseline_hash": "v_31ea78b4",
    "challenger_version": HARNESS_VERSION,
    "challenger_hash": HARNESS_HASH if not args.approach else f"approach:{args.approach}",
    "baseline_brier": round(baseline_brier, 4),
    "challenger_brier": round(challenger_brier, 4),
    "delta": round(delta, 4),
    "model": OPENAI_MODEL if not args.approach else f"{OPENAI_MODEL} + text-embedding-3-small",
    "temperature": 0,
    "eval_size": len(eval_items),
    "results_hash": results_hash,
    "per_tweet": merged,
}

results_file = os.path.join(os.path.dirname(__file__), "data", "duel_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

raw_file = os.path.join(os.path.dirname(__file__), "data", "duel_raw.json")
with open(raw_file, "w") as f:
    json.dump(challenger_raw, f, indent=2)

print(f"\n{'─' * 50}")
print(f"Baseline ({results['baseline_version']}):  {baseline_brier:.4f}")
print(f"Challenger ({approach_name}): {challenger_brier:.4f}")
print(f"Delta:                     {delta:+.4f}")
if delta > 0.01:
    print(f"Result:                    IMPROVEMENT ✅")
elif delta < -0.01:
    print(f"Result:                    REGRESSION ❌")
else:
    print(f"Result:                    WITHIN NOISE")
print(f"Hash:                      {results_hash}")
print(f"{'─' * 50}")
print(f"To verify: set POSTPROPHET_TEMPERATURE=0, run with --approach {approach_name}, compare hash")
