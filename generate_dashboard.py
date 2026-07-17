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

    # Brier history
    brier_data = []
    for s in scores:
        brier_data.append({
            "date": s["scored_at"][:10],
            "brier": round(s["brier_score"], 4),
            "batch_size": s["batch_size"],
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
        text = p["context"]["text"]
        if len(text) > 80:
            text = text[:80] + "..."
        recent_preds.append({
            "author": p["context"]["author"]["username"],
            "target": p["target"],
            "probability": round(pred["probability"] * 100),
            "verdict": verdict,
            "actual": p["actual_impressions"],
            "hit": hit,
            "text": text,
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
