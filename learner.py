"""Learner: auto-retrain after each tweet resolves.

Runs after the daily cron resolves predictions.
Retrains logistic regression coefficients with all resolved data.
Updates calibration curve.
Tracks model improvement over time.
"""
import json, time, os, sys, math
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def retrain_coefficients():
    """Retrain logistic regression on all resolved predictions.
    
    Uses original dimension scores (temp=0.3 or temp=0, whatever was used
    at prediction time). For best results, re-score at temp=0 periodically.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
    except ImportError:
        print("sklearn not installed")
        return None
    
    with open('data/predictions.jsonl') as f:
        preds = [json.loads(l) for l in f if l.strip()]
    
    resolved = [p for p in preds if p.get('resolved')]
    if len(resolved) < 50:
        print(f"Not enough data: {len(resolved)} resolved")
        return None
    
    dims = ['hook_strength', 'specificity', 'emotional_trigger',
            'bookmark_worthiness', 'structure_readability']
    
    X, y = [], []
    for p in resolved:
        pred = p.get('prediction', {})
        scores = [pred.get(d, 5) for d in dims]
        X.append(scores)
        y.append(1 if p['actual_impressions'] >= p['target'] else 0)
    
    X, y = np.array(X), np.array(y)
    
    # Train with regularization
    model = LogisticRegression(max_iter=1000, C=0.01)
    model.fit(X, y)
    
    # CV Brier
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(LogisticRegression(max_iter=1000, C=0.01), X, y, cv=cv, method='predict_proba')[:, 1]
    brier = sum((p - a) ** 2 for p, a in zip(probs, y)) / len(y)
    
    coefs = {d: float(c) for d, c in zip(dims, model.coef_[0])}
    intercept = float(model.intercept_[0])
    
    # Calibration
    calibration = {}
    for prob_val, actual in zip(probs, y):
        bucket = round(prob_val * 10) / 10
        if bucket not in calibration:
            calibration[bucket] = {'total': 0, 'hits': 0}
        calibration[bucket]['total'] += 1
        calibration[bucket]['hits'] += actual
    
    cal_data = {}
    for bucket in sorted(calibration.keys()):
        c = calibration[bucket]
        cal_data[f"{bucket:.1f}"] = {
            'predicted': bucket,
            'actual': round(c['hits'] / c['total'], 3) if c['total'] > 0 else 0,
            'count': c['total']
        }
    
    return {
        'coefs': coefs,
        'intercept': intercept,
        'brier_cv': round(brier, 4),
        'n_predictions': len(resolved),
        'n_hits': sum(y),
        'hit_rate': round(sum(y) / len(y), 3),
        'calibration': cal_data,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }

def update_model(coefs, intercept):
    """Update the harness coefficients in postprophet.py."""
    import re
    
    with open('postprophet.py') as f:
        content = f.read()
    
    # Update _COEFS
    coefs_str = "    _COEFS = {\n"
    for dim, coef in coefs.items():
        coefs_str += f'        "{dim}": {coef:.4f},\n'
    coefs_str += "    }"
    
    # Find and replace _COEFS
    pattern = r'    _COEFS = \{[^}]+\}'
    content = re.sub(pattern, coefs_str, content)
    
    # Update _INTERCEPT
    pattern = r'_INTERCEPT = [-\d.]+'
    content = re.sub(pattern, f'_INTERCEPT = {intercept:.4f}', content)
    
    with open('postprophet.py', 'w') as f:
        f.write(content)
    
    print(f"Updated coefficients in postprophet.py")

def run_learner():
    """Main learner: retrain, update, report."""
    print("=" * 50)
    print("LEARNER: Retraining model with latest data")
    print("=" * 50)
    
    result = retrain_coefficients()
    if not result:
        print("Not enough data to retrain")
        return
    
    print(f"\nData: {result['n_predictions']} predictions, {result['n_hits']} hits ({result['hit_rate']:.1%})")
    print(f"CV Brier: {result['brier_cv']}")
    
    # Load previous Brier for comparison
    prev_file = 'data/model_history.jsonl'
    prev_brier = None
    if os.path.exists(prev_file):
        with open(prev_file) as f:
            lines = f.readlines()
            if lines:
                prev = json.loads(lines[-1])
                prev_brier = prev.get('brier_cv')
    
    if prev_brier:
        delta = result['brier_cv'] - prev_brier
        direction = "improved" if delta < 0 else "regressed" if delta > 0 else "unchanged"
        print(f"Previous Brier: {prev_brier} | Delta: {delta:+.4f} ({direction})")
    
    print(f"\nCoefficients:")
    for d, c in result['coefs'].items():
        print(f"  {d}: {c:+.4f}")
    print(f"  intercept: {result['intercept']:+.4f}")
    
    print(f"\nCalibration:")
    for bucket, data in result['calibration'].items():
        error = abs(data['predicted'] - data['actual'])
        print(f"  {bucket}: predicted {data['predicted']}, actual {data['actual']}, count {data['count']}, error {error:.2f}")
    
    # Update the harness
    update_model(result['coefs'], result['intercept'])
    
    # Save history
    result_clean = {
        'coefs': result['coefs'],
        'intercept': result['intercept'],
        'brier_cv': result['brier_cv'],
        'n_predictions': result['n_predictions'],
        'n_hits': result['n_hits'],
        'hit_rate': result['hit_rate'],
        'trained_at': result['trained_at'],
    }
    with open(prev_file, 'a') as f:
        f.write(json.dumps(result_clean) + '\n')
    
    print(f"\nModel updated. History saved to {prev_file}")
    print(f"Trained at: {result['trained_at']}")

if __name__ == '__main__':
    run_learner()
