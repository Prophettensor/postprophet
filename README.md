# PostProphet

Predict reach before you post.

PostProphet is a prediction harness that forecasts the probability of a tweet hitting 2x the author's median impressions within 24 hours of posting. Write a tweet, get a probability — plus structured scoring and editorial feedback on what would improve it.

## How it works

```
1. write    — draft your tweet
2. predict  — PostProphet scores it and predicts probability + feedback
3. craft    — iterate with the craft loop: draft → predict → revise → repeat
4. post     — ship when the probability is high enough
```

## Usage

### Predict a single tweet (product mode)

```bash
python postprophet.py predict "your tweet text" --author username
```

PostProphet fetches the author's real engagement history and predicts the probability of hitting 2x their median impressions in 24h. If the tweet has attached media, it analyzes it via vision. If you've set up ecosystem tracking (see below), it adds cross-account context.

### Iterative improvement (craft loop)

```bash
python craft.py --author username --topic "what the tweet should be about" \
  --target-confidence 0.80 --max-iterations 5
```

Drafts a tweet, gets PostProphet's prediction + feedback, revises based on the feedback, repeats until target confidence or max iterations. The writing agent sees the author's actual best/worst tweets and PostProphet's pattern analysis.

### Eval mode (test the harness)

```bash
python postprophet.py track            — Capture fresh tweets from tracked accounts (<12h old)
python postprophet.py resolve          — Resolve pending predictions and score with Brier
python postprophet.py report           — Show score history + detailed predictions
```

## Scoring

Each tweet is scored on multiple dimensions (0-10), grounded in the X algorithm's engagement weights. Scores cover content quality and algorithmic signals. The dimensions evolve as the harness improves.

## Ecosystem context (optional)

If you add accounts to `accounts.txt` and run `track` once, PostProphet caches their engagement data. Future `predict` calls then include ecosystem context — cross-account baselines normalized by follower count (impressions per 1K followers) so craft quality is compared, not account size. Zero extra API cost (uses cache only).

Without ecosystem setup, `predict` works fine — it just uses the author's own history without cross-account comparison.

## Data integrity

- **No engagement leakage:** The prediction tweet's own likes, replies, bookmarks, and quotes are hidden from the LLM. The model predicts as if the tweet hasn't been posted yet.
- **No time leakage:** Ecosystem tweets posted after the prediction tweet are filtered out. Age is calculated relative to the prediction tweet's post time.
- **Self-reply filtered:** Thread continuations are removed from both the prediction target and the baseline (they get artificially low impressions).
- **Fresh tweets only:** Track mode only predicts tweets under 12 hours old.
- **Resolve at posted_at + 24h:** Predictions resolve at the tweet's actual 24h mark, not 24h from when we predicted.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `X_BEARER_TOKEN` | — | X API v2 bearer token (pay-per-use) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `POSTPROPHET_MODEL` | `gpt-4o-mini` | LLM model (model-agnostic) |
| `POSTPROPHET_TIMEFRAME` | `24` | Hours before resolving |
| `POSTPROPHET_TARGET` | `10000` | Fallback target (overridden by 2x median) |
| `POSTPROPHET_BATCH` | `20` | Tweets per keyword capture batch |

## The eval

PostProphet predicts the probability of a tweet hitting 2x the author's median impressions within 24 hours of posting. Impressions resolve at the tweet's posted_at + 24h. Brier score measures calibration. Change the harness, rerun on fresh tweets, compare scores.

### Cost control

- Author baseline cache (6h TTL) — warm runs skip X API calls
- Baseline fetch limited to 10 tweets per account
- Ecosystem context from cache (zero API cost in predict mode)
- X API pay-per-use: ~$2.60 per cold track cycle, ~$0.21 per warm cycle

> Pfeffer, J.; Matter, D.; Sargsyan, A. (2023). "The Half-Life of a Tweet." *Proceedings of the International AAAI Conference on Web and Social Media*, 17(1).

## Quickstart

```bash
git clone https://github.com/buckZz7/postprophet.git
cd postprophet
uv venv && source .venv/bin/activate
uv pip install httpx openai
cp .env.example .env  # fill in your keys
python postprophet.py predict "your tweet" --author your_handle
```

### Optional: ecosystem context

```bash
echo "competitor_handle" >> accounts.txt  # add accounts for ecosystem tracking
python postprophet.py track  # fetch + cache author data (one-time, refreshes every 6h)
python postprophet.py predict "your tweet" --author your_handle  # now includes ecosystem context
```

## License

MIT
