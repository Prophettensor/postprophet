"""Cosine similarity approach with grounded feedback.

Uses embeddings for both prediction AND feedback.
Prediction: cosine similarity ratio to hits vs misses
Feedback: explains WHY the tweet is similar to hits or misses,
referencing specific patterns from the author's actual best/worst tweets.

The feedback is derived from the same signal that makes the prediction
good — vector similarity to real outcomes. Not circular.
"""
import math
import os
import time
from openai import OpenAI

client = None

def _get_client():
    global client
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return client

def _get_embedding(text: str) -> list[float]:
    c = _get_client()
    resp = c.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000],
    )
    return resp.data[0].embedding

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def _generate_feedback(
    target_text: str,
    hit_tweets: list[str],
    miss_tweets: list[str],
    hit_sim: float,
    miss_sim: float,
    author: str,
) -> str:
    """Generate feedback grounded in cosine similarity to real outcomes.
    
    Uses LLM to explain WHY the tweet is similar to hits or misses,
    referencing the author's actual best/worst tweets.
    """
    c = _get_client()
    
    # Show the model the target + closest hits + closest misses
    # with actual tweet text so it can identify patterns
    hit_examples = "\n".join(f"  - {t[:150]}" for t in hit_tweets[:3])
    miss_examples = "\n".join(f"  - {t[:150]}" for t in miss_tweets[:3])
    
    prompt = f"""You are analyzing a tweet for @{author}.

The tweet being analyzed:
"{target_text}"

Their HIGHEST performing tweets (these hit 2x median):
{hit_examples}

Their LOWEST performing tweets (these missed):
{miss_examples}

Similarity analysis:
- Similarity to their best tweets: {hit_sim:.2f}
- Similarity to their worst tweets: {miss_sim:.2f}

Based on the actual tweet text above, explain in 1-2 sentences:
1. What pattern do their best tweets share that this tweet does NOT have?
2. What is the ONE most important change to make this tweet more like their best work?

Be specific. Reference actual words, structure, or content from their best tweets.
Do not use generic advice like "make it more engaging." 
Reference what specifically makes their hits work.

Example good feedback: "Their best tweets open with a specific number (e.g., 'Brier 0.21 on 600 predictions'). This tweet opens with a vague announcement. Fix: replace the opener with the actual metric."

Return only the feedback text."""

    resp = c.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
    )
    
    feedback = resp.choices[0].message.content
    return feedback.strip() if feedback else ""


def predict_with_feedback(context: dict, target: int) -> dict:
    """Predict probability AND generate grounded feedback.
    
    Returns:
        {
            "probability": float 0.0-1.0,
            "suggestions": str (feedback grounded in cosine similarity),
            "hit_similarity": float,
            "miss_similarity": float,
        }
    """
    target_text = context.get("text", "")
    author_tweets = context.get("author_all_tweets", [])
    author_name = context.get("author", {}).get("username", "unknown")
    
    if not target_text or not author_tweets:
        return {"probability": 0.28, "suggestions": "", "hit_similarity": 0, "miss_similarity": 0}
    
    # Split author's tweets into hits and misses
    hits_texts = []
    misses_texts = []
    for t in author_tweets:
        text = t.get("text", "")
        impressions = t.get("impressions", t.get("public_metrics", {}).get("impression_count", 0))
        if not text or not impressions:
            continue
        if impressions >= target:
            hits_texts.append(text)
        else:
            misses_texts.append(text)
    
    if not hits_texts and not misses_texts:
        return {"probability": 0.28, "suggestions": "", "hit_similarity": 0, "miss_similarity": 0}
    
    # Embed target tweet
    target_emb = _get_embedding(target_text)
    time.sleep(0.3)
    
    # Embed up to 5 hits and 5 misses
    hit_embs = []
    for t in hits_texts[:5]:
        hit_embs.append((_get_embedding(t), t))
        time.sleep(0.2)
    
    miss_embs = []
    for t in misses_texts[:5]:
        miss_embs.append((_get_embedding(t), t))
        time.sleep(0.2)
    
    # Compute average similarity
    hit_sims = [(s, _cosine_similarity(target_emb, e)) for e, s in hit_embs] if hit_embs else []
    miss_sims = [(s, _cosine_similarity(target_emb, e)) for e, s in miss_embs] if miss_embs else []
    
    hit_sim = sum(s for _, s in hit_sims) / len(hit_sims) if hit_sims else 0
    miss_sim = sum(s for _, s in miss_sims) / len(miss_sims) if miss_sims else 0
    
    # Probability: ratio of hit similarity to total, blended with base rate
    total_sim = hit_sim + miss_sim
    if total_sim > 0:
        raw_prob = hit_sim / total_sim
    else:
        raw_prob = 0.28
    
    prob = 0.5 * raw_prob + 0.5 * 0.28
    prob = max(0.0, min(1.0, prob))
    
    # Generate feedback grounded in the similarity analysis
    feedback = _generate_feedback(
        target_text,
        [s for s, _ in hit_sims],  # hit tweet texts sorted by similarity
        [s for s, _ in miss_sims],  # miss tweet texts sorted by similarity
        hit_sim,
        miss_sim,
        author_name,
    )
    
    return {
        "probability": prob,
        "suggestions": feedback,
        "hit_similarity": hit_sim,
        "miss_similarity": miss_sim,
    }


def predict_probability(context: dict, target: int) -> float:
    """Simple probability-only interface (for backward compatibility)."""
    return predict_with_feedback(context, target)["probability"]
