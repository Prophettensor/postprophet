# PostProphet

Predict reach before you post.

PostProphet is a prediction harness that forecasts the probability of a tweet hitting 2x the author's median impressions within 24 hours of posting. Write a tweet, get a probability — plus structured scoring and editorial feedback on what would improve it.

## How it works

```
1. write    — draft your tweet
2. predict  — PostProphet scores it on 8 dimensions, predicts probability + feedback
3. craft    — iterate with the craft loop: draft → predict → revise → repeat
4. post     — ship when the probability is high enough
```

## Usage

### Predict a single tweet (product mode)

```bash
python postprophet.py predict "your tweet text" --author username
```

PostProphet fetches the author's real engagement history, builds an ecosystem baseline from tracked accounts, analyzes any attached media, and predicts the probability of hitting 2x their median impressions in 24h.

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

## Scoring dimensions

Each tweet is scored on 8 dimensions (0-10), grounded in the X algorithm's engagement weights:

| Dimension | What it measures | Algorithm basis |
|---|---|---|
| hook_strength | Scroll-stopping opener | — |
| specificity | Concrete vs vague | — |
| emotional_trigger | Makes you feel something | — |
| reply_inducement | Invites response | Replies = 27-150x a like |
| bookmark_worthiness | Reference-worthy content | Bookmarks = 10-12x a like |
| structure_readability | Line breaks, visual pacing | Dwell time = ~10x a like |
| clarity_density | Insight per character | — |
| link_penalty_risk | External URL present | Links = -30% to -94% reach |

## Ecosystem context

PostProphet uses all accounts in `accounts.txt` as an ecosystem baseline. Tweets are normalized by follower count (impressions per 1K followers) so craft quality is compared, not account size. The model sees:

- The author's 10 recent tweets ranked by impressions
- Best/worst tweets across the ecosystem (normalized)
- Recent industry pulse (what subnets are tweeting about, with engagement + age)
- Media analysis (image type, quality, relevance via gpt-4o-mini vision)

Ecosystem context works in both `track` and `predict` mode (from cache, zero extra API cost).

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
echo "your_handle" >> accounts.txt  # add accounts for ecosystem context
python postprophet.py track  # fetch + cache author data
python postprophet.py predict "your tweet" --author your_handle
```

## License

MIT
