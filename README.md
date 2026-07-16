# PostProphet

Predict reach before you post.

PostProphet is a prediction harness that forecasts the probability of a tweet hitting a target impression count within a given timeframe. Write a tweet, set a target, get a probability — plus reasoning on what would improve it.

## How it works

```
1. write    — draft your tweet, set a target (impressions + timeframe)
2. predict  — PostProphet predicts probability + reasoning
3. iterate  — tweak the tweet, re-predict, repeat until it clears your threshold
4. post     — ship when the probability is high enough
```

## Usage

### Predict your own tweet

```bash
python postprophet.py predict "your tweet text" \
  --author username \
  --target 10000 \
  --timeframe 24 \
  --post-time "2026-07-15T09:00:00Z"
```

### Eval mode (test the harness)

```bash
python postprophet.py track            — Capture from tracked accounts (accounts.txt)
python postprophet.py resolve          — Resolve pending predictions and score
python postprophet.py report           — Show score history + reasoning
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `X_BEARER_TOKEN` | — | X API v2 bearer token |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `POSTPROPHET_MODEL` | `gpt-4o-mini` | LLM model (model-agnostic) |
| `POSTPROPHET_TIMEFRAME` | `24` | Hours before resolving |
| `POSTPROPHET_TARGET` | `10000` | Target impressions |
| `POSTPROPHET_BATCH` | `20` | Tweets per keyword capture batch |

## The eval

PostProphet is evaluated on real tweets. It predicts the probability of a tweet hitting 1.2x the author's average impressions within 24 hours. Impressions resolve naturally. Brier score measures how calibrated the predictions are. Change the harness, rerun on fresh tweets, compare scores.

### Resolve timing

Predictions resolve at 24 hours. Research shows 95% of tweets stop getting new impressions after 24 hours, with a median half-life of 80 minutes (Pfeffer et al., 2023).

> Pfeffer, J.; Matter, D.; Sargsyan, A. (2023). "The Half-Life of a Tweet." *Proceedings of the International AAAI Conference on Web and Social Media*, 17(1).

## Quickstart

```bash
git clone https://github.com/buckZz7/postprophet.git
cd postprophet
uv venv && source .venv/bin/activate
uv pip install httpx openai
cp .env.example .env  # fill in your keys
python postprophet.py predict "your tweet" --author your_handle --target 5000
```

## License

MIT
