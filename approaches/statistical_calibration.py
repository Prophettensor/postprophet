"""Statistical probability approach.

The LLM scores dimensions (what it's good at).
Math computes the probability from historical hit rates (what the LLM is bad at).

Flow:
1. LLM scores 8 dimensions (0-10 each) — same as rubric
2. For each dimension, look up historical hit rate for that score
3. Weight dimensions by their causality signal
4. Compute weighted average probability

No LLM guessing on probability. Pure statistical calibration.
Dimensions are causal (proven by causality_eval.py).
"""
import json
import os
import time
from openai import OpenAI

client = None

def _get_client():
    global client
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return client

# ── Build hit rate lookup from all resolved predictions ─────────

_HIT_RATES = None

def _build_hit_rates():
    """Build lookup: for each dimension, what's the hit rate at each score level?"""
    global _HIT_RATES
    if _HIT_RATES is not None:
        return _HIT_RATES
    
    fname = "predictions" + ".jsonl"
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", fname)
    if not os.path.exists(data_file):
        return {}
    
    with open(data_file) as f:
        preds = [json.loads(l) for l in f if l.strip()]
    
    resolved = [p for p in preds if p.get("resolved")]
    
    dims = ['hook_strength', 'specificity', 'emotional_trigger',
            'bookmark_worthiness', 'structure_readability', 'clarity_density']
    
    rates = {}
    for dim in dims:
        low = [p for p in resolved if dim in p.get("prediction", {}) and p["prediction"][dim] < 5]
        mid = [p for p in resolved if dim in p.get("prediction", {}) and 5 <= p["prediction"][dim] < 7]
        high = [p for p in resolved if dim in p.get("prediction", {}) and p["prediction"][dim] >= 7]
        
        def hr(group):
            if not group:
                return 0.28
            hits = sum(1 for p in group if p["actual_impressions"] >= p["target"])
            return hits / len(group)
        
        rates[dim] = {"low": hr(low), "mid": hr(mid), "high": hr(high)}
        rates[dim]["signal"] = rates[dim]["high"] - rates[dim]["low"]
    
    _HIT_RATES = rates
    return rates

def _score_to_level(score):
    if score < 5:
        return "low"
    elif score < 7:
        return "mid"
    else:
        return "high"

# ── LLM dimension scoring (self-contained, no postprophet import) ──

def _get_dimension_scores(context: dict, target: int) -> dict:
    """Call LLM to score dimensions. Self-contained."""
    c = _get_client()
    model = os.environ.get("POSTPROPHET_MODEL", "gpt-4o-mini")
    
    # Load environment config
    env_path = os.path.join(os.path.dirname(__file__), "..", "environments", "twitter.json")
    with open(env_path) as f:
        env = json.load(f)
    
    # Build dimensions string
    dim_lines = []
    dim_names = []
    for dim in env["dimensions"]:
        dim_lines.append(f"- {dim['name']}: {dim['description']}")
        dim_names.append(dim["name"])
    
    # Build probability guide
    guide_str = "\n".join(f"- {g}" for g in env.get("probability_guide", []))
    considerations_str = "\n".join(f"- {c}" for c in env.get("extra_considerations", []))
    
    # Build JSON template
    json_fields = []
    for name in dim_names:
        suffix = " (higher = safer)" if "risk" in name or "safe" in name else ""
        json_fields.append(f'  "{name}": <0-10{suffix}>')
    json_fields.append('  "reasoning": "<2-3 sentences>"')
    json_fields.append('  "suggestions": "<Rewrite directive. Identify the WEAKEST dimension. Explain WHY it scored low. Prescribe one concrete fix.>"')
    json_str = ",\n".join(json_fields)
    
    # Build context
    baseline = context.get("author_baseline", {})
    author = context.get("author", {})
    all_tweets = context.get("author_all_tweets", [])
    tweets_str = "\n".join(f"  - {t.get('text', '')[:150]} ({t.get('impressions', 0):,} imp)" for t in all_tweets[:10]) if all_tweets else "  (none)"
    
    prompt = f"""You are a social media reach forecasting harness.

Score this tweet on {len(dim_names)} dimensions (0-10 each):

{chr(10).join(dim_lines)}

A tweet scoring below 6 on ANY dimension is unlikely to hit the target.

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
        result = json.loads(content)
    except json.JSONDecodeError:
        return {}
    
    return result

def predict_with_feedback(context: dict, target: int) -> dict:
    """Predict using LLM dimension scores + statistical probability."""
    # Step 1: Get dimension scores from LLM
    prediction = _get_dimension_scores(context, target)
    
    dims = ['hook_strength', 'specificity', 'emotional_trigger',
            'bookmark_worthiness', 'structure_readability', 'clarity_density']
    
    # Step 2: Look up hit rates
    rates = _build_hit_rates()
    
    # Step 3: Weight by signal and compute probability
    total_weight = 0
    weighted_prob = 0
    
    for dim in dims:
        score = prediction.get(dim)
        if score is None or dim not in rates:
            continue
        
        level = _score_to_level(score)
        hit_rate = rates[dim][level]
        signal = abs(rates[dim]["signal"])
        
        if signal < 0.01:
            continue
        
        total_weight += signal
        weighted_prob += hit_rate * signal
    
    if total_weight > 0:
        prob = weighted_prob / total_weight
    else:
        prob = 0.28
    
    prob = max(0.01, min(0.99, prob))
    
    return {
        "probability": prob,
        "suggestions": prediction.get("suggestions", ""),
        "reasoning": prediction.get("reasoning", ""),
        "hook_strength": prediction.get("hook_strength", 0),
        "specificity": prediction.get("specificity", 0),
        "emotional_trigger": prediction.get("emotional_trigger", 0),
        "reply_inducement": prediction.get("reply_inducement", 0),
        "bookmark_worthiness": prediction.get("bookmark_worthiness", 0),
        "structure_readability": prediction.get("structure_readability", 0),
        "clarity_density": prediction.get("clarity_density", 0),
        "link_penalty_risk": prediction.get("link_penalty_risk", 0),
        "_method": "statistical_calibration",
    }

def predict_probability(context: dict, target: int) -> float:
    return predict_with_feedback(context, target)["probability"]
