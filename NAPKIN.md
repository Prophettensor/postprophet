# PostProphet

## What this is

A prediction harness that forecasts the probability of a tweet hitting a target impression count within a given timeframe. Given who is posting, what they're posting, when they're posting, where they're posting, and a target — it predicts the likelihood of hitting that target, plus a point estimate and reasoning that can be used to improve the content before it goes live.

## Who it's for

Developers building marketing agents, social media tools, and growth products. Anyone who needs their agent to answer "will this post perform?" before it goes live.

## How it works

**Inputs:** Who (author handle, follower count, engagement baseline from recent tweets, top recent tweets), What (content text, media), When (planned post time — when it will go live), Where (platform — X for now), Target (impression count + timeframe to measure)

**Output:** Probability (0.0–1.0) that the content will hit the target within the timeframe, a point estimate (predicted impression count), reasoning explaining the prediction, and suggestions for what would improve it.

**The feedback loop:** An agent generates content → PostProphet predicts → if probability is low, reasoning suggests changes → agent modifies → PostProphet re-predicts → repeat until probability clears a threshold → content ships.

## Key decisions

- **Probability + point estimate.** The harness outputs "70% chance of hitting 10k in 24h" and "estimated 8,200 impressions." Probability is scored with Brier score. Point estimate is scored with accuracy ratio. Both metrics track prediction quality from different angles.
- **No engagement metrics shown to the LLM.** The harness never shows likes, retweets, replies, or impressions for the tweet being predicted. It predicts from content, author baseline (their other tweets), timing, and trending topics only. This means old tweets and new tweets eval identically — the LLM can't cheat.
- **Reasoning modifies content.** The harness doesn't just predict — it explains why and what would improve the prediction. This reasoning is what enables the feedback loop.
- **Timeframe is a required input.** "Will this hit 10k?" is meaningless without "in how long?" Every prediction includes a timeframe.
- **Live eval on real tweets.** The harness is evaluated against real tweets from tracked accounts. It predicts, impressions resolve naturally over 24 hours, predictions are scored with Brier score. No sandbox, no historical replay.
- **Dynamic targets.** Each account gets a target of 1.2x their average impressions. This creates predictions that range from 30-80% instead of all-hits or all-misses, giving meaningful Brier scores.
- **24h resolve.** 95% of tweets stop getting impressions after 24 hours (Pfeffer et al., 2023). Resolving at 24h captures the full impression lifecycle.
- **Model-agnostic.** Swap any LLM with one env var. Compare Brier scores across models to find which predicts best.
- **Generalizes beyond X.** Start with X, but the concept applies to any platform with engagement metrics and an API.

## MVP scope

**In:**
- Prediction harness: takes who + what + when + where + target → outputs probability + point estimate + reasoning + suggestions
- X platform support (timeline fetch for eval, API for author/context data)
- Live eval: capture real tweets from tracked accounts, predict, resolve at 24h, score with Brier score
- Feedback loop: reasoning suggests content changes to improve probability
- Account tracking: read from accounts.txt, fetch latest tweets, compute dynamic targets
- Regression testing: change the harness, rerun on fresh tweets, compare Brier scores

**Out:**
- Generating tweets (that's the product layer, not the harness)
- Posting tweets (the harness never publishes)
- Platforms beyond X (phase 2)
- A user-facing UI (that's a product, not the harness)
- Historical/sandbox eval (live eval only — real data, real resolution, ungameable)

## Open questions

- **X API tier:** Free tier gives timeline fetch + impression counts. Rate limits may require upgrading for larger account lists.
- **Eval batch size:** How many tweets per batch for meaningful Brier score? Starting with whatever accounts produce per day, accumulating over time.
- **Data point discovery:** How does the harness discover and test new data sources automatically vs. manually adding them?
- **Point estimate utility:** Will the LLM's point estimates be accurate enough to use as a secondary metric, or should we drop it?
- **Model comparison:** gpt-4o-mini is the default. Worth testing grok-2 (X-native training data) and comparing Brier scores.
