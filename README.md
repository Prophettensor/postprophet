# PostProphet

Predict reach before you post.

PostProphet is a prediction harness that forecasts the probability of a tweet hitting a target impression count within a given timeframe. Give it who's posting, what they're saying, when they're posting, and a target — it returns a probability, a point estimate, and reasoning that explains the prediction and what would improve it.

## Why

Most marketing tools tell you what happened after you posted. PostProphet tells you what will happen before. Agents can generate a tweet, check the probability, iterate on the content until it clears a threshold, then ship. No more posting blind.

## How it works

```
1. capture  — X API fetches tweets from tracked accounts, gathers context:
                - Author stats (followers, following, tweet count)
                - Author baseline (avg impressions, avg likes, avg retweets from recent tweets)
                - Author's top 3 recent tweets (text + impressions)
                - Trending topics
                - When it was posted (or will be posted)
2. predict  — LLM predicts probability + point estimate + reasoning + suggestions
3. wait     — timeframe hours pass (default 24h, based on impression plateau research)
4. resolve  — check actual impressions via X API
5. score    — Brier score measures prediction accuracy
6. report   — track accuracy over time
```

## Usage

### Eval mode (test the harness against real tweets)

```bash
python postprophet.py track            — Capture from tracked accounts (accounts.txt)
python postprophet.py resolve          — Resolve pending predictions and score
python postprophet.py report           — Show score history + reasoning
python postprophet.py capture          — Keyword search capture (alternative)
```

### Product mode (predict for an unpublished tweet)

```bash
python postprophet.py predict "your tweet text" \
  --author username \
  --target 10000 \
  --timeframe 24 \
  --post-time "2026-07-15T09:00:00Z"
```

The harness fetches the author's recent tweets to build an engagement baseline, gets trending topics, and predicts the probability of hitting your target within the timeframe.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `X_BEARER_TOKEN` | — | X API v2 bearer token |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `POSTPROPHET_MODEL` | `gpt-4o-mini` | LLM model |
| `POSTPROPHET_TIMEFRAME` | `24` | Hours before resolving |
| `POSTPROPHET_TARGET` | `10000` | Target impressions (ignored in track mode — uses 1.2x author baseline) |
| `POSTPROPHET_BATCH` | `20` | Tweets per keyword capture batch |

## The eval

PostProphet is evaluated on real tweets from real accounts. The harness predicts the probability of a tweet hitting 1.2x the author's average impressions within 24 hours. Impressions resolve naturally. Brier score measures how calibrated the predictions are.

### What the LLM sees

The LLM never sees engagement metrics for the tweet being predicted. No likes, no retweets, no impressions. It predicts from:

- **Who** — author identity, follower count, engagement baseline from their other tweets
- **What** — the tweet text
- **When** — when it was posted and how long ago
- **Where** — the platform (X for now)
- **What's trending** — current trending topics

This means the eval is ungameable. The LLM can't look up the answer because it never sees engagement data. Old tweets and new tweets eval identically.

### Resolve timing

Predictions resolve at 24 hours after the tweet was posted. Research shows that 95% of tweets receive no relevant new impressions after 24 hours, and the median half-life of a tweet is 80 minutes (Pfeffer et al., 2023).

> Pfeffer, J.; Matter, D.; Sargsyan, A. (2023). "The Half-Life of a Tweet." *Proceedings of the International AAAI Conference on Web and Social Media*, 17(1).

For faster iteration, the timeframe can be reduced — at 3 hours, roughly 80% of final impressions have accumulated.

### Scoring

**Brier score** — standard forecasting metric. `(probability - actual_outcome)²` averaged across a batch. 0 = perfect, 1 = worst, 0.25 = random guessing. Lower is better.

**Point estimate accuracy** — the LLM also outputs a point estimate (predicted impression count). Scored with `1 - |estimated - actual| / actual`. Useful as a secondary metric.

### Regression testing

Change the prediction logic, rerun against fresh tweets, compare Brier scores. If the score goes down, the change improved the harness. If it goes up, revert. Same principle as any ML benchmark.

## Architecture

PostProphet is a **harness** — code that controls the flow:

1. **Capture** (code): X API fetches tweets from tracked accounts, computes author baselines
2. **Predict** (agent): LLM reasons from context → probability + point estimate + reasoning
3. **Store** (code): JSONL prediction records
4. **Wait** (code): 24h timer
5. **Resolve** (code): X API impression check
6. **Score** (code): Brier score + point estimate accuracy
7. **Report** (code): score history with reasoning

The agent is one function: `predict(context) → {probability, point_estimate, reasoning, suggestions}`. Everything else is harness code. The LLM is model-agnostic — swap models with one env var and compare Brier scores.

## Tracked accounts

PostProphet tracks Bittensor ecosystem accounts by default (see `accounts.txt`). Add or remove handles to control your eval data pool. Each account gets a dynamic target of 1.2x their average impressions — so predictions range from 30-80% instead of all-hits or all-misses.

## Quickstart

```bash
git clone https://github.com/buckZz7/postprophet.git
cd postprophet
uv venv && source .venv/bin/activate
uv pip install httpx openai
cp .env.example .env  # fill in your keys
python postprophet.py track   # capture + predict
# wait 24h
python postprophet.py resolve  # score
python postprophet.py report   # see results
```

## License

MIT
