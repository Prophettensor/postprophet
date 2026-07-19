"""Statistical probability approach with logistic regression.

The HARNESS calls OpenAI to get dimension scores.
This approach only does MATH on the scores it receives.
No API keys. No file access. No network calls. Pure math.

This is the secure pattern for all approaches:
- Harness scores dimensions (LLM call)
- Approach computes probability (math only)
- Harness evaluates the result

Logistic regression coefficients trained on 600 predictions at temp=0.
"""
import math

# Coefficients from 5-fold CV logistic regression on 6 causal dimensions
# Trained on 600 resolved predictions at temp=0, Brier 0.2004 (CV)
COEFS = {
    "hook_strength": 0.0251,
    "specificity": 0.0900,
    "emotional_trigger": 0.0174,
    "bookmark_worthiness": 0.0563,
    "structure_readability": 0.0202,
    "clarity_density": -0.0062,
}
INTERCEPT = -2.2056

# Timing coefficients (from hit rate analysis)
# Thursday = 34% hit, Saturday = 18% hit, base = 28%
TIMING_COEFS = {
    "hour_13_22_utc": 0.03,   # US business hours: 31-44% hit rate
    "hour_00_06_utc": -0.05,  # Dead zone: 0-22% hit rate
    "is_weekend": -0.03,       # Saturday worst at 18%
    "is_thursday": 0.02,       # Best day at 34%
}

DIMS = list(COEFS.keys())

def predict_with_feedback(scores: dict) -> dict:
    """Compute probability from pre-scored dimensions.
    
    Args:
        scores: dict of dimension scores (0-10) from the harness LLM call.
                Also includes 'suggestions' and 'reasoning' from the LLM.
    
    Returns:
        dict with probability and all scores/feedback
    """
    z = INTERCEPT
    for dim in DIMS:
        score = scores.get(dim, 5)
        z += COEFS[dim] * score
    
    # Timing adjustments
    hour = scores.get("hour_utc", 12)
    if 13 <= hour <= 22:
        z += TIMING_COEFS["hour_13_22_utc"]
    if 0 <= hour <= 6:
        z += TIMING_COEFS["hour_00_06_utc"]
    if scores.get("is_weekend", False):
        z += TIMING_COEFS["is_weekend"]
    if scores.get("day_of_week") == "Thu":
        z += TIMING_COEFS["is_thursday"]
    
    prob = 1.0 / (1.0 + math.exp(-z))
    prob = max(0.01, min(0.99, prob))
    
    return {
        "probability": prob,
        "suggestions": scores.get("suggestions", ""),
        "reasoning": scores.get("reasoning", ""),
        **{dim: scores.get(dim, 5) for dim in DIMS},
        "_method": "logistic_regression",
    }

def predict_probability(scores: dict) -> float:
    """Simple probability from pre-scored dimensions."""
    z = INTERCEPT
    for dim in DIMS:
        score = scores.get(dim, 5)
        z += COEFS[dim] * score
    
    # Timing adjustments
    hour = scores.get("hour_utc", 12)
    if 13 <= hour <= 22:
        z += TIMING_COEFS["hour_13_22_utc"]
    if 0 <= hour <= 6:
        z += TIMING_COEFS["hour_00_06_utc"]
    if scores.get("is_weekend", False):
        z += TIMING_COEFS["is_weekend"]
    if scores.get("day_of_week") == "Thu":
        z += TIMING_COEFS["is_thursday"]
    prob = 1.0 / (1.0 + math.exp(-z))
    return max(0.01, min(0.99, prob))
