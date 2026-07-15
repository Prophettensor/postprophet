# PostProphet

Prediction engine for social media reach.

Given who is posting, what they're posting, where they're posting, and a target — PostProphet predicts the probability of hitting that target within a given timeframe. It's the technology layer that lets agents grade their own content before it goes live.

## Quickstart

```bash
git clone https://github.com/buckZz7/postprophet.git
cd postprophet
uv venv && source .venv/bin/activate
uv pip install httpx openai
cp .env.example .env  # fill in your keys
python postprophet.py run
```

## How it works

```
1. capture  — X API search captures real tweets, gathers context:
                - Author stats (followers, following, tweet count)
                - Author baseline (avg impressions, avg likes, avg retweets from recent tweets)
                - Author's top 3 recent tweets (text + impressions)
                - Early engagement signals (likes, retweets, replies at capture time)
                - Trending topics
                - When it was posted (or will be posted)
2. predict  — LLM predicts probability of hitting target impressions within timeframe
3. wait     — timeframe hours pass (default 1h)
4. resolve  — check actual impressions via X API
5. score    — Brier score measures prediction accuracy
6. report   — track accuracy over time
```

## Usage

### Eval mode (test the engine against real tweets)

```bash
python postprophet.py capture          # Capture real tweets and predict
python postprophet.py resolve          # Resolve pending predictions and score
python postprophet.py report           # Show score history
python postprophet.py run              # Capture → resolve → report
```

### Product mode (predict for an unpublished tweet)

```bash
python postprophet.py predict "your tweet text" \
  --author username \
  --target 10000 \
  --timeframe 24 \
  --post-time "2026-07-15T09:00:00Z"
```

The engine fetches the author's recent tweets to build an engagement baseline, gets trending topics, and predicts the probability of hitting your target within the timeframe.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `X_BEARER_TOKEN` | — | X API v2 bearer token |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `SOCIALQUANT_MODEL` | `gpt-4o-mini` | LLM model |
| `SOCIALQUANT_TIMEFRAME` | `1` | Hours before resolving |
| `SOCIALQUANT_TARGET` | `10000` | Target impressions |
| `SOCIALQUANT_BATCH` | `20` | Tweets per capture batch |

## The eval

Live, real-time, ungameable. The engine predicts on real tweets captured from the X API. Impressions resolve naturally over the timeframe. Brier score measures prediction accuracy. PRs to the engine are scored against the same tweet batch — does accuracy improve?

### What the engine considers

- **Author baseline** — not just follower count, but average impressions per tweet. An account with 1k followers that averages 5k impressions is very different from one that averages 200.
- **Recent top tweets** — the author's 3 best-performing recent tweets (text + impressions) give the engine a sense of what works for this account.
- **Time of day** — when the tweet was (or will be) posted. A tweet at 3am hits differently than 9am.
- **Trending topics** — what's hot right now and whether the tweet relates.
- **Early engagement** — in eval mode, likes/retweets/replies at capture time.
- **Planned post time** — in product mode, when the tweet will go live.

## Architecture

PostProphet is a **harness** — code that controls the flow:

1. **Capture** (code): X API search, context gathering, author baseline computation
2. **Predict** (agent): LLM reasons from context → probability + reasoning
3. **Store** (code): JSONL prediction records
4. **Wait** (code): timeframe timer
5. **Resolve** (code): X API impression check
6. **Score** (code): Brier score calculation
7. **Report** (code): score history

The agent is one function: `predict(context) → {probability, reasoning}`. Everything else is harness.

## License

MIT
