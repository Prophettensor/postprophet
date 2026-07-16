# PostProphet

## What this is

A prediction harness that forecasts the probability of a tweet hitting a target impression count within a given timeframe. Write a tweet, set a target, get a probability — plus reasoning on what would improve it.

## Who it's for

Anyone who posts on X and wants to know if their tweet will perform before they post it.

## How it works

**Inputs:** Who (author + their engagement baseline), What (the tweet text), When (planned post time), Target (impression count + timeframe)

**Output:** Probability (0.0–1.0), a point estimate, reasoning explaining the prediction, and suggestions for what would improve it.

**The feedback loop:** Write a tweet → PostProphet predicts → if probability is low, reasoning suggests changes → tweak → re-predict → post when it clears your threshold.

## Key decisions

- **Probability + point estimate.** Outputs "70% chance of hitting 10k in 24h" and "estimated 8,200 impressions."
- **Reasoning modifies content.** It doesn't just predict — it explains why and what to change.
- **Timeframe is required.** "Will this hit 10k?" is meaningless without "in how long?"
- **24h resolve.** 95% of tweets stop getting impressions after 24 hours (Pfeffer et al., 2023).
- **Model-agnostic.** Swap any LLM with one env var. Compare Brier scores.

## Open questions

- **Point estimate utility:** Will the LLM's point estimates be accurate enough to use as a secondary metric?
- **Model comparison:** gpt-4o-mini is the default. Worth testing other models and comparing Brier scores.
