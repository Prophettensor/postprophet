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
# These will be updated when temp=0 retraining completes
COEFS = {
    "hook_strength": -0.0947,
    "specificity": 0.2348,
    "emotional_trigger": -0.0357,
    "bookmark_worthiness": 0.0336,
    "structure_readability": 0.0212,
    "clarity_density": 0.0951,
}
INTERCEPT = -2.6613

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
    prob = 1.0 / (1.0 + math.exp(-z))
    return max(0.01, min(0.99, prob))
