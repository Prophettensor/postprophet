"""Statistical probability approach with logistic regression.

LLM scores dimensions (what it's good at).
Logistic regression computes probability (what the LLM is bad at).

Brier 0.20 (cross-validated on 600 predictions)
Beats cosine (0.24) AND has grounded feedback (causality +18.6%).

Coefficients learned from 600 resolved predictions via sklearn.
"""
import json
import os
import math
from openai import OpenAI

client = None

def _get_client():
    global client
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return client

# ── Logistic regression model (pre-computed from 600 predictions) ──

# Coefficients from 5-fold CV logistic regression on 6 causal dimensions
# Trained on 600 resolved predictions, Brier 0.2018 (CV)
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

def _logistic_predict(scores: dict) -> float:
    """Compute probability using logistic regression."""
    z = INTERCEPT
    for dim in DIMS:
        score = scores.get(dim, 5)
        z += COEFS[dim] * score
    prob = 1.0 / (1.0 + math.exp(-z))
    return max(0.01, min(0.99, prob))

# ── LLM dimension scoring ───────────────────────────────────────

def _get_dimension_scores(context: dict, target: int) -> dict:
    """Call LLM to score dimensions. Self-contained."""
    c = _get_client()
    model = os.environ.get("POSTPROPHET_MODEL", "gpt-4o-mini")
    
    env_path = os.path.join(os.path.dirname(__file__), "..", "environments", "twitter.json")
    with open(env_path) as f:
        env = json.load(f)
    
    dim_lines = []
    dim_names = []
    for dim in env["dimensions"]:
        dim_lines.append(f"- {dim['name']}: {dim['description']}")
        dim_names.append(dim["name"])
    
    json_fields = []
    for name in dim_names:
        json_fields.append(f'  "{name}": <0-10>')
    json_fields.append('  "reasoning": "<2-3 sentences>"')
    json_fields.append('  "suggestions": "<Rewrite directive. Identify the WEAKEST dimension. Explain WHY it scored low. Prescribe one concrete fix.>"')
    json_str = ",\n".join(json_fields)
    
    baseline = context.get("author_baseline", {})
    author = context.get("author", {})
    all_tweets = context.get("author_all_tweets", [])
    tweets_str = "\n".join(f"  - {t.get('text', '')[:150]} ({t.get('impressions', 0):,} imp)" for t in all_tweets[:10]) if all_tweets else "  (none)"
    
    prompt = f"""You are a social media reach forecasting harness.

Score this tweet on {len(dim_names)} dimensions (0-10 each):

{chr(10).join(dim_lines)}

Return JSON:
{{
{json_str}
}}

Tweet: {context.get('text', '')}
Author: @{author.get('username', 'unknown')} ({author.get('followers', 0):,} followers)
Author baseline: median {baseline.get('median_impressions', 0):.0f} impressions/tweet
Author's recent tweets:
{tweets_str}
"""
    
    resp = c.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a social media reach prediction harness. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=float(os.environ.get("POSTPROPHET_TEMPERATURE", "0.3")),
        response_format={"type": "json_object"},
    )
    
    content = resp.choices[0].message.content
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}

def predict_with_feedback(context: dict, target: int) -> dict:
    """LLM scores dimensions, logistic regression computes probability."""
    prediction = _get_dimension_scores(context, target)
    prob = _logistic_predict(prediction)
    
    return {
        "probability": prob,
        "suggestions": prediction.get("suggestions", ""),
        "reasoning": prediction.get("reasoning", ""),
        **{dim: prediction.get(dim, 5) for dim in DIMS},
        "_method": "logistic_regression",
    }

def predict_probability(context: dict, target: int) -> float:
    return predict_with_feedback(context, target)["probability"]
