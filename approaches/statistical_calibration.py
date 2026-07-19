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

# Interaction coefficients (initial estimate, to be calibrated)
INTERACTION_COEFS = {
    "spec_x_emot": 0.5,   # specificity * emotional_trigger
    "spec_x_clar": 0.3,   # specificity * clarity_density
}

DIMS = list(COEFS.keys())

def predict_with_feedback(scores: dict) -> dict:
    """Compute probability from pre-scored dimensions + interaction terms.
    
    Interaction features capture compound effects:
    - specificity * emotional_trigger (40% hit rate when both high vs 28% base)
    - specificity * clarity_density (37% hit rate when both high)
    """
    z = INTERCEPT
    for dim in DIMS:
        score = scores.get(dim, 5)
        z += COEFS[dim] * score
    
    # Interaction terms
    spec = scores.get("specificity", 5)
    emot = scores.get("emotional_trigger", 5)
    clar = scores.get("clarity_density", 5)
    
    # Normalize interactions to 0-1 range (scores are 0-10, product is 0-100)
    z += INTERACTION_COEFS["spec_x_emot"] * (spec * emot / 100.0)
    z += INTERACTION_COEFS["spec_x_clar"] * (spec * clar / 100.0)
    
    prob = 1.0 / (1.0 + math.exp(-z))
    prob = max(0.01, min(0.99, prob))
    
    return {
        "probability": prob,
        "suggestions": scores.get("suggestions", ""),
        "reasoning": scores.get("reasoning", ""),
        **{dim: scores.get(dim, 5) for dim in DIMS},
        "_method": "logistic_regression_interactions",
    }

def predict_probability(scores: dict) -> float:
    """Simple probability from pre-scored dimensions."""
    z = INTERCEPT
    for dim in DIMS:
        score = scores.get(dim, 5)
        z += COEFS[dim] * score
    prob = 1.0 / (1.0 + math.exp(-z))
    return max(0.01, min(0.99, prob))
