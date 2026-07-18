#!/usr/bin/env python3
"""Regenerate dashboard data from predictions and scores.
Run after each resolve to update the live eval dashboard."""

import json
import os

PREDICTIONS_FILE = os.path.join(os.path.dirname(__file__), "data", "predictions.jsonl")
SCORES_FILE = os.path.join(os.path.dirname(__file__), "data", "scores.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "docs", "dashboard-data.json")


def main():
    with open(PREDICTIONS_FILE) as f:
        preds = [json.loads(line) for line in f if line.strip()]
    with open(SCORES_FILE) as f:
        scores = [json.loads(line) for line in f if line.strip()]

    resolved = [p for p in preds if p.get("resolved")]
    new_arch = [p for p in resolved if p.get("prediction", {}).get("hook_strength")]

    # Brier history — one cumulative data point per day
    # Instead of per-batch Brier (which spikes and is noisy), show
    # cumulative Brier across ALL resolved predictions up to that date
    from collections import defaultdict
    by_day = defaultdict(list)
    for p in new_arch:
        day = p.get("predicted_at", "")[:10]
        if not day:
            day = p.get("resolve_after", "")[:10]
        if not day:
            day = "unknown"
        by_day[day].append(p)
    
    # Sort days and compute cumulative Brier
    sorted_days = sorted(by_day.keys())
    cumulative_preds = []
    brier_data = []
    for day in sorted_days:
        cumulative_preds.extend(by_day[day])
        if cumulative_preds:
            day_brier = sum(
                (p["prediction"]["probability"] - (1.0 if p["actual_impressions"] >= p["target"] else 0.0)) ** 2
                for p in cumulative_preds
            ) / len(cumulative_preds)
            brier_data.append({
                "date": day,
                "brier": round(day_brier, 4),
                "cumulative_count": len(cumulative_preds),
            })

    # Calibration (new arch only)
    buckets = {}
    for p in new_arch:
        prob = p["prediction"]["probability"]
        actual = 1.0 if p["actual_impressions"] >= p["target"] else 0.0
        bucket = int(prob * 10) * 10
        bucket_label = f"{bucket}-{min(bucket+20,100)}%"
        if bucket_label not in buckets:
            buckets[bucket_label] = {"predicted": [], "actual": []}
        buckets[bucket_label]["predicted"].append(prob)
        buckets[bucket_label]["actual"].append(actual)

    cal_data = []
    for label, data in sorted(buckets.items()):
        avg_pred = sum(data["predicted"]) / len(data["predicted"])
        avg_actual = sum(data["actual"]) / len(data["actual"])
        cal_data.append({
            "range": label,
            "predicted": round(avg_pred * 100),
            "actual": round(avg_actual * 100),
            "count": len(data["predicted"]),
        })

    # Recent predictions (new arch, last 20)
    recent_preds = []
    for p in new_arch[-20:]:
        pred = p["prediction"]
        hit = p["actual_impressions"] >= p["target"]
        verdict = "YES" if pred["probability"] >= 0.5 else "NO"
        # Prediction is correct if:
        # - Said YES (>=50%) and tweet hit, OR
        # - Said NO (<50%) and tweet missed
        prediction_correct = (verdict == "YES" and hit) or (verdict == "NO" and not hit)
        text = p["context"]["text"]
        # Build scores dict
        dim_scores = {}
        for key in ["hook_strength", "specificity", "emotional_trigger",
                      "reply_inducement", "bookmark_worthiness",
                      "structure_readability", "clarity_density", "link_penalty_risk"]:
            if key in pred:
                dim_scores[key] = pred[key]
        recent_preds.append({
            "author": p["context"]["author"]["username"],
            "target": p["target"],
            "probability": round(pred["probability"] * 100),
            "verdict": verdict,
            "actual": p["actual_impressions"],
            "hit": hit,
            "prediction_correct": prediction_correct,
            "text": text,
            "reasoning": pred.get("reasoning", ""),
            "suggestions": pred.get("suggestions", ""),
            "scores": dim_scores,
            "followers": p.get("followers_at_prediction", p["context"]["author"].get("followers", 0)),
            "elapsed": p.get("elapsed_at_prediction", ""),
            "has_url": p.get("has_url", False),
            "has_external_url": p.get("has_external_url", False),
            "is_quote_tweet": p.get("is_quote_tweet", False),
        })

    # Overall new arch Brier
    total_brier = (
        sum(
            (p["prediction"]["probability"] - (1.0 if p["actual_impressions"] >= p["target"] else 0.0)) ** 2
            for p in new_arch
        )
        / len(new_arch)
        if new_arch
        else 0
    )

    # Hit rate
    hits = sum(1 for p in new_arch if p["actual_impressions"] >= p["target"])

    dashboard_data = {
        "brier_history": brier_data,
        "calibration": cal_data,
        "recent_predictions": recent_preds,
        "stats": {
            "total_resolved": len(resolved),
            "new_arch_resolved": len(new_arch),
            "new_arch_brier": round(total_brier, 4),
            "pending": len(preds) - len(resolved),
            "hit_rate": round(hits / len(new_arch) * 100) if new_arch else 0,
            "last_updated": scores[-1]["scored_at"] if scores else "",
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"Dashboard data generated: {OUTPUT_FILE}")
    print(f"  Brier history: {len(brier_data)} batches")
    print(f"  Calibration buckets: {len(cal_data)}")
    print(f"  Recent predictions: {len(recent_preds)}")
    print(f"  New arch Brier: {total_brier:.4f}")


if __name__ == "__main__":
    main()
