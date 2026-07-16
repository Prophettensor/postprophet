# PostProphet

## What this is

A prediction harness that forecasts the probability of a tweet hitting 2x the author's median impressions within 24 hours of posting. Write a tweet, get a probability — plus structured scoring and editorial feedback on what would improve it.

## Who it's for

Anyone who posts on X and wants to know if their tweet will perform before they post it. Also for anyone building agent pipelines that craft tweets programmatically.

## How it works

**Inputs:** Who (author + their engagement baseline + ecosystem context), What (the tweet text + attached media), When (planned post time)

**Output:** Probability (0.0–1.0), a point estimate, 8-dimension scoring, pattern analysis, and directive editorial feedback.

**The feedback loop:** Write a tweet → PostProphet scores it → if probability is low, feedback suggests specific changes → tweak → re-predict → post when it clears your threshold.

## Key decisions

- **2x median target.** The target is aspirational — "will this perform twice as well as their typical tweet?" Grounded in the author's real history, not a fixed number.
- **8 scoring dimensions.** Hook, specificity, emotional trigger, reply inducement, bookmark worthiness, structure, clarity, link penalty risk. Grounded in the X algorithm's engagement weights.
- **Ecosystem context.** All tracked accounts form a normalized baseline (impressions per 1K followers). The model sees what "good" looks like across the niche, not just for one author.
- **No engagement leakage.** The prediction tweet's own metrics are hidden. Ecosystem tweets posted after the prediction are filtered out. The model predicts as if the tweet hasn't been posted yet.
- **Media analysis.** gpt-4o-mini vision analyzes attached images (type, quality, relevance).
- **Directive feedback.** Not "add a personal touch" but "Open with 'We just shipped...' — match their best tweet's first-person announcement structure."
- **24h resolve at posted_at.** Predictions resolve at the tweet's actual 24h mark, not 24h from prediction time.
- **Model-agnostic.** Swap any LLM with one env var. Compare Brier scores.
