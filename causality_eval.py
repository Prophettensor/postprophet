"""Causality eval: do the approach's dimension scores correlate with real outcomes?

For each dimension an approach uses:
- Split tweets into high (7+) and low (<5) scoring groups
- Check if high-scoring group actually hits more often
- Signal = hit_rate(high) - hit_rate(low)

This is the guidance score. Non-circular. No revised tweets needed.
If the approach says "strong specificity" on tweets that actually hit more,
the feedback is grounded in reality.

Usage:
  python causality_eval.py
  python causality_eval.py --approach cosine_with_feedback
"""
import json, os, sys, argparse, importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from postprophet import EVAL_FILE, OPENAI_MODEL
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

# Load eval set
with open(EVAL_FILE) as f:
    eval_items = [json.loads(l) for l in f if l.strip()]

eval_items = eval_items[:args.limit]

print(f"Causality eval: {len(eval_items)} tweets")
print(f"Approach: {args.approach or 'rubric'}")
print(f"Model: {OPENAI_MODEL}")
print()

# ── Get dimension scores from approach ──────────────────────────

print("Getting dimension scores...")
scored_tweets = []

for i, item in enumerate(eval_items, 1):
    context = {
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
    
    scores = {}
    feedback = ""
    
    if custom_approach:
        try:
            if hasattr(custom_approach, 'predict_with_feedback'):
                result = custom_approach.predict_with_feedback(context, item["target"])
                scores = {k: v for k, v in result.items() if isinstance(v, (int, float)) and k != "probability"}
                feedback = result.get("suggestions", "")
            else:
                prob = custom_approach.predict_probability(context, item["target"])
                scores = {}
        except Exception as e:
            print(f"  [{i}/{len(eval_items)}] Error: {e}")
            continue
    else:
        # Rubric
        os.environ["POSTPROPHET_TEMPERATURE"] = "0"
        from postprophet import predict
        try:
            prediction = predict(context, target=item["target"])
            scores = {k: v for k, v in prediction.items() if isinstance(v, (int, float)) and k not in ("probability", "point_estimate")}
            feedback = prediction.get("suggestions", "")
            import time
            time.sleep(1.5)
        except Exception as e:
            print(f"  [{i}/{len(eval_items)}] Error: {e}")
            continue
    
    scored_tweets.append({
        "item": item,
        "scores": scores,
        "feedback": feedback,
    })
    
    dim_str = ", ".join(f"{k}={v}" for k, v in scores.items()) if scores else "no scores"
    print(f"  [{i}/{len(eval_items)}] @{item['author']['username']}: {dim_str}")

# ── Compute causality signal per dimension ─────────────────────

print(f"\n{'─' * 60}")
print(f"CAUSALITY ANALYSIS")
print(f"{'─' * 60}")
print(f"\n{'Dimension':<30} {'Low (<5)':>12} {'Mid (5-6)':>12} {'High (7+)':>12} {'Signal':>10}")
print("-" * 76)

# Collect all dimension names
all_dims = set()
for st in scored_tweets:
    all_dims.update(st["scores"].keys())

dim_signals = {}

for dim in sorted(all_dims):
    low = [st for st in scored_tweets if dim in st["scores"] and st["scores"][dim] < 5]
    mid = [st for st in scored_tweets if dim in st["scores"] and 5 <= st["scores"][dim] < 7]
    high = [st for st in scored_tweets if dim in st["scores"] and st["scores"][dim] >= 7]
    
    def hit_rate(group):
        if not group:
            return None
        hits = sum(1 for st in group if st["item"]["hit"])
        return hits / len(group) * 100
    
    lr = hit_rate(low)
    mr = hit_rate(mid)
    hr = hit_rate(high)
    
    if lr is not None and hr is not None:
        signal = hr - lr
    else:
        signal = 0
    
    dim_signals[dim] = signal
    
    def fmt(r, n):
        if r is None:
            return "N/A"
        return f"{r:.1f}% ({n})"
    
    print(f"{dim:<30} {fmt(lr, len(low)):>12} {fmt(mr, len(mid)):>12} {fmt(hr, len(high)):>12} {signal:>+8.1f}%")

# ── Overall guidance score ─────────────────────────────────────

# Average signal across all dimensions
signals = list(dim_signals.values())
avg_signal = sum(signals) / len(signals) if signals else 0

# Count dimensions with positive signal (grounded)
positive_dims = sum(1 for s in signals if s > 5)  # >5% is meaningful
negative_dims = sum(1 for s in signals if s < -5)
neutral_dims = sum(1 for s in signals if -5 <= s <= 5)

print(f"\n{'─' * 60}")
print(f"GUIDANCE SCORE (causality)")
print(f"{'─' * 60}")
print(f"Dimensions tested:    {len(signals)}")
print(f"  Positive (>5%):     {positive_dims} (grounded feedback)")
print(f"  Neutral (-5 to 5%): {neutral_dims} (noise)")
print(f"  Negative (<-5%):    {negative_dims} (misleading feedback)")
print(f"")
print(f"Average signal:       {avg_signal:+.1f}%")
print(f"Guidance score:       {avg_signal:+.4f}")
print(f"  (positive = dimensions correlate with real outcomes)")
print(f"  (negative = dimensions are misleading)")
print(f"  (zero = no signal or no dimensions)")
print(f"{'─' * 60}")

# ── Save ───────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print(f"CONFUSION MATRIX (threshold 0.5)")
print(f"{'─' * 60}")
tp = tn = fp = fn = 0
for st in scored_tweets:
    actual = st["item"]["hit"]
    predicted = st.get("prob", 0.5) >= 0.5
    if actual and predicted: tp += 1
    elif actual and not predicted: fn += 1
    elif not actual and predicted: fp += 1
    else: tn += 1

print(f"              Pred NO  Pred YES")
print(f"  Actual NO    {tn:4d}      {fp:4d}")
print(f"  Actual YES   {fn:4d}      {tp:4d}")
if tp + fn > 0:
    print(f"  Recall: {tp/(tp+fn)*100:.0f}% ({tp}/{tp+fn} hits caught)")
if tp + fp > 0:
    print(f"  Precision: {tp/(tp+fp)*100:.0f}% ({tp}/{tp+fp} predicted hits correct)")
print(f"  Accuracy: {(tp+tn)/len(scored_tweets)*100:.1f}%")
print(f"{'─' * 60}")

print(f"\n{'─' * 60}")
print(f"FEEDBACK QUALITY CHECK")
print(f"{'─' * 60}")

import re as _re
_DIM_NAMES = ['hook_strength', 'specificity', 'emotional_trigger', 'reply_inducement',
              'bookmark_worthiness', 'structure_readability', 'clarity_density', 'link_penalty_risk',
              'timing_optimization', 'novelty']

def _has_dim_name(text):
    t = text.lower()
    for dim in _DIM_NAMES:
        p1 = r'\b' + _re.escape(dim) + r'\b'
        p2 = r'\b' + _re.escape(dim.replace('_', ' ')) + r'\b'
        if _re.search(p1, t) or _re.search(p2, t):
            return True
    return False

example_based = 0
dimension_based = 0
no_feedback = 0

for st in scored_tweets:
    fb = st.get("feedback", "")
    if not fb:
        no_feedback += 1
        continue
    has_imp = bool(_re.search(r'\d[\d,]*\s*(imp|impressions?)', fb.lower()))
    has_rank = bool(_re.search(r'rank\s*#?\d|best tweet|worst tweet|top tweet', fb.lower()))
    has_comparison = bool(_re.search(r'their best|this tweet|this one|your best', fb.lower()))
    has_dim = _has_dim_name(fb)
    has_score = bool(_re.search(r'score\s*:?\s*\d|dimension|weakest', fb.lower()))
    
    positive = sum([has_imp, has_rank, has_comparison])
    negative = sum([has_dim, has_score])
    
    if positive >= 1 and negative == 0:
        example_based += 1
    else:
        dimension_based += 1

total_fb = example_based + dimension_based + no_feedback
print(f"Example-based feedback:  {example_based}/{total_fb} ({example_based/total_fb*100:.0f}%)")
print(f"Dimension-based feedback: {dimension_based}/{total_fb} ({dimension_based/total_fb*100:.0f}%)")
print(f"No feedback:              {no_feedback}/{total_fb}")
if total_fb > 0:
    fb_score = example_based / total_fb
    print(f"\nFeedback quality: {fb_score:.2f}")
    print(f"  (1.0 = all example-based, 0.0 = all dimension-based)")
    if fb_score < 0.5:
        print(f"  ⚠️  WARNING: feedback is not example-based")
print(f"{'─' * 60}")

# Save
results = {
    "approach": args.approach or "rubric",
    "scorer": "causality",
    "dimensions_tested": len(signals),
    "positive_dims": positive_dims,
    "neutral_dims": neutral_dims,
    "negative_dims": negative_dims,
    "avg_signal": round(avg_signal, 4),
    "guidance_score": round(avg_signal, 4),
    "dim_signals": {k: round(v, 2) for k, v in dim_signals.items()},
}

results_file = os.path.join(os.path.dirname(__file__), "data", "causality_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {results_file}")
