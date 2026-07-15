# SocialQuant

Prediction engine for social media reach.

Given who is posting, what they're posting, where they're posting, and a target — SocialQuant predicts the probability of hitting that target within a given timeframe. It's the technology layer that lets agents grade their own content before it goes live.

## Quickstart

```bash
git clone https://github.com/buckZz7/socialquant.git
cd socialquant
uv venv && source .venv/bin/activate
uv pip install httpx openai
cp .env.example .env  # fill in your keys
python socialquant.py run
```

## How it works

```
1. capture  — X API search captures real tweets, gathers context (author stats, early metrics, trending)
2. predict  — LLM predicts probability of hitting target impressions within timeframe
3. wait     — timeframe hours pass (default 4h)
4. resolve  — check actual impressions via X API
5. score    — Brier score measures prediction accuracy
6. report   — track accuracy over time
```

## Usage

```bash
python socialquant.py capture     # Capture real tweets and predict
python socialquant.py resolve     # Resolve pending predictions and score
python socialquant.py report      # Show score history
python socialquant.py run         # Capture → resolve → report
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `X_BEARER_TOKEN` | — | X API v2 bearer token |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `SOCIALQUANT_MODEL` | `gpt-4o-mini` | LLM model |
| `SOCIALQUANT_TIMEFRAME` | `4` | Hours before resolving |
| `SOCIALQUANT_TARGET` | `10000` | Target impressions |
| `SOCIALQUANT_BATCH` | `20` | Tweets per capture batch |

## The eval

Live, real-time, ungameable. The engine predicts on real tweets captured from the X API. Impressions resolve naturally over the timeframe. Brier score measures prediction accuracy. PRs to the engine are scored against the same tweet batch — does accuracy improve?

## Architecture

SocialQuant is a **harness** — code that controls the flow:

1. **Capture** (code): X API search, context gathering
2. **Predict** (agent): LLM reasons from context → probability + reasoning
3. **Store** (code): JSONL prediction records
4. **Wait** (code): timeframe timer
5. **Resolve** (code): X API impression check
6. **Score** (code): Brier score calculation
7. **Report** (code): score history

The agent is one function: `predict(context) → {probability, reasoning}`. Everything else is harness.

## License

MIT
