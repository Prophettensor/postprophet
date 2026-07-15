# PostProphet

## What this is

A prediction engine that forecasts the probability of social media content reaching a target engagement metric within a given timeframe. Given who is posting, what they're posting, where they're posting, and a target — it predicts the likelihood of hitting that target. The prediction comes with reasoning that can be used to improve the content before it goes live.

This is the technology layer, not the product. Products built on top — tweet generators, marketing agents, growth tools — use PostProphet to grade themselves and iterate toward content that performs.

## Who it's for

Developers building social media tools, agents, and marketing products. Not end users directly. Anyone who needs their agent or tool to answer "will this post perform?" can use PostProphet as the prediction layer.

## How it works

**Inputs:** Who (author handle, follower count, engagement baseline from recent tweets, top recent tweets), What (content text, media), When (planned post time — when it will go live), Where (platform — X for now, architecture supports more), Target (impression count + timeframe to measure)

**Output:** Probability (0.0–1.0) that the content will hit the target within the timeframe, plus reasoning explaining the prediction and what would increase it.

**The feedback loop:** An agent generates content → PostProphet predicts → if probability is low, reasoning suggests changes → agent modifies → PostProphet re-predicts → repeat until probability clears a threshold → content ships.

**Extensible data inputs:** The engine tests any data point that might improve prediction accuracy — trending topics, competitor activity, time of day, media type, hashtag patterns, reply context. New data sources are A/B tested against real outcomes. If a data point improves accuracy, it stays.

## Key decisions

- **Probability, not exact numbers.** The engine outputs "70% chance of hitting 10k in 24h," not "this will get exactly 8,200 impressions." Probabilities fit Brier score eval naturally and are more useful for go/no-go decisions.
- **Reasoning modifies content.** The engine doesn't just predict — it explains why and what would improve the prediction. This reasoning is what enables the feedback loop.
- **Timeframe is a required input.** "Will this hit 10k?" is meaningless without "in how long?" Every prediction includes a timeframe.
- **Live eval on real tweets.** The engine is evaluated against real tweets captured in real-time via X API filtered stream. The engine predicts, the impressions resolve naturally, predictions are scored. No sandbox, no historical replay — the engine gets the same data a real marketing agent would have.
- **The engine is not the product.** PostProphet is the prediction layer. Tweet generators, autonomous posting tools, marketing dashboards — those are products built on top.
- **Generalizes beyond X.** Start with X, but the concept applies to any platform with engagement metrics and an API.

## MVP scope

**In:**
- Prediction engine: takes who + what + where + target + timeframe → outputs probability + reasoning
- X platform support (filtered stream for eval, API for author/context data)
- Live eval: capture real tweets, predict, wait for timeframe, score with Brier score
- Feedback loop: reasoning suggests content changes to improve probability
- Extensible data inputs: test new data sources against real outcomes
- Regression testing: PRs to the engine are scored against the same tweet batch, does accuracy improve?

**Out:**
- Generating tweets (that's the product layer, not the engine)
- Posting tweets (the engine never publishes)
- Platforms beyond X (phase 2)
- A user-facing UI (that's a product, not the technology)
- Historical/sandbox eval (live eval only — real data, real resolution, ungameable)

## Open questions

- **X API tier:** Which tier gives filtered stream + impression counts? Need to verify cost and rate limits.
- **Eval batch size:** How many tweets per batch for meaningful Brier score? (Numinous uses 100 events.)
- **Data point discovery:** How does the engine discover and test new data sources automatically vs. manually adding them?
- **Brier score vs. alternatives:** Brier score is the default for probability calibration. Is there a better metric for this domain?
- **Live eval latency:** Short timeframes (1h) give faster eval cycles but impressions may not have stabilized. What's the right minimum timeframe for eval?
