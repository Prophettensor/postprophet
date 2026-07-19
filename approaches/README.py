"""
PostProphet — Custom approach interface.

To submit a new prediction approach:
1. Create a file in approaches/ (e.g. approaches/my_approach.py)
2. Implement a predict_probability(context: dict, target: int) -> float function
3. The function must return a float between 0.0 and 1.0
4. The function CANNOT import or modify postprophet.py internals
5. The function CANNOT access the eval set or know which tweets are being evaluated

Your function receives:
- context: dict with tweet text, author info, baseline, history, media, etc.
- target: int, the impression count to beat (2x median)

Your function returns:
- float: probability (0.0-1.0) that the tweet will hit the target

Allowed:
- Calling external APIs (OpenAI embeddings, etc.)
- Computing similarity scores
- Using any data in the context dict
- Adding helper functions in your file

NOT allowed (will be rejected by PR validation):
- Importing postprophet module
- Reading data/eval_set.jsonl or data/predictions.jsonl
- Hardcoding tweet IDs, author handles, or probability values
- if/else branching on specific tweet content or authors
- Modifying any file outside approaches/

Example:

def predict_probability(context: dict, target: int) -> float:
    from openai import OpenAI
    client = OpenAI()
    
    text = context.get('text', '')
    
    # Get embedding
    resp = client.embeddings.create(
        model='text-embedding-3-small',
        input=text,
    )
    embedding = resp.data[0].embedding
    
    # ... compute similarity to reference embeddings ...
    
    return 0.28  # your probability estimate
"""
