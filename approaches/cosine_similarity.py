"""Cosine similarity approach.

Embeds the target tweet and the author's historical tweets.
Computes average cosine similarity to hits vs misses.
Uses the ratio as probability, blended with 28% base rate.

Cost: ~$0.01 per 50 predictions (embeddings are cheap)
No LLM scoring — pure vector similarity.
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
    """Get embedding for text, truncated to API limit."""
    c = _get_client()
    resp = c.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000],
    )
    return resp.data[0].embedding

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def predict_probability(context: dict, target: int) -> float:
    """Predict probability using cosine similarity to author's hits/misses."""
    target_text = context.get("text", "")
    author_tweets = context.get("author_all_tweets", [])
    
    if not target_text or not author_tweets:
        return 0.28  # Base rate
    
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
        return 0.28
    
    # Embed target tweet
    target_emb = _get_embedding(target_text)
    time.sleep(0.3)
    
    # Embed up to 5 hits and 5 misses
    hit_embs = []
    for t in hits_texts[:5]:
        hit_embs.append(_get_embedding(t))
        time.sleep(0.2)
    
    miss_embs = []
    for t in misses_texts[:5]:
        miss_embs.append(_get_embedding(t))
        time.sleep(0.2)
    
    # Compute average similarity to hits vs misses
    hit_sim = sum(_cosine_similarity(target_emb, e) for e in hit_embs) / len(hit_embs) if hit_embs else 0
    miss_sim = sum(_cosine_similarity(target_emb, e) for e in miss_embs) / len(miss_embs) if miss_embs else 0
    
    # Probability: ratio of hit similarity to total
    total_sim = hit_sim + miss_sim
    if total_sim > 0:
        raw_prob = hit_sim / total_sim
    else:
        raw_prob = 0.28
    
    # Blend with base rate for calibration
    prob = 0.5 * raw_prob + 0.5 * 0.28
    
    return max(0.0, min(1.0, prob))
