# PostProphet

## What this is

A simulated X for agents to test content before posting. Content agents can't post, delete, post, delete until they get it right — every bad post is real. PostProphet is the test harness that fills that gap: draft, predict, iterate in simulation, ship once when confidence is high.

The harness predicts the probability of a tweet hitting 2x the author's median impressions within 24 hours of posting, scored with Brier score against real resolved impressions.

## Who it's for

Agents that post on X. Whether it's an autonomous content agent posting on behalf of a protocol, or a human-in-the-loop using the craft loop to iterate — PostProphet lets you test before you post.

## How it works

**Inputs:** Who (author + their engagement baseline), What (the tweet text + attached media), When (planned post time)

**Output:** Probability (0.0–1.0), a point estimate, structured scoring, pattern analysis, and directive editorial feedback.

**The feedback loop:** Write a tweet → PostProphet scores it → if probability is low, feedback suggests specific changes → tweak → re-predict → post when it clears your threshold.

## Key decisions

- **2x median target.** The target is aspirational — "will this perform twice as well as their typical tweet?" Grounded in the author's real history, not a fixed number.
- **Structured scoring.** Content quality and algorithmic signals, grounded in the X algorithm's engagement weights. Dimensions evolve as the harness improves.
- **Optional ecosystem context.** If you set up `accounts.txt` and run `track`, the model sees cross-account baselines (normalized by follower count). Without it, predict works fine on the author's own history.
- **No engagement leakage.** The prediction tweet's own metrics are hidden. Ecosystem tweets posted after the prediction are filtered out. The model predicts as if the tweet hasn't been posted yet.
- **Media analysis.** If a tweet has attached media, gpt-4o-mini vision analyzes it (type, quality, relevance).
- **Directive feedback.** Not "add a personal touch" but "Open with 'We just shipped...' — match their best tweet's first-person announcement structure."
- **24h resolve at posted_at.** Predictions resolve at the tweet's actual 24h mark, not 24h from prediction time.
- **Model-agnostic.** Swap any LLM with one env var. Compare Brier scores.
