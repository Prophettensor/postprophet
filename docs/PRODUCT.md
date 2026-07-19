# PostProphet — Product: The Content Engine

## What it is

A self-improving content coach for X. It watches what works, learns from it, and helps write better tweets. Not a prediction market. Not a mining subnet. A product that produces good content.

## The Loop

```
Market Research (what's the ecosystem talking about?)
    ↓
Content Agent (draft tweets based on what worked)
    ↓
PostProphet (evaluate: logistic regression + example-based feedback)
    ↓
Revise (until it clears the bar)
    ↓
Post to @PostProphet
    ↓
Measure (did it hit 2x median?)
    ↓
Learn (update coefficients, improve feedback)
    ↓
Repeat
```

## Architecture

### Market Research Agent (`market_research.py`)
- Monitors key voices, themes, competitors
- Identifies what's performing well across the ecosystem
- Produces a market brief: themes, hot topics, what's resonating

### Content Agent (`content_team.py`)
- Drafts tweets based on market brief + dev brief
- Uses examples from the author's best work
- Revision loop: 40% target, 2 plateau, 5 max iterations

### PostProphet Harness (`postprophet.py`)
- Scores 5 causal dimensions (hook, specificity, emotion, bookmark, structure)
- Logistic regression computes probability (Brier 0.20)
- Example-based feedback: "Their best tweet does X. Yours does Y. Fix: do Z."
- Timing data: day of week, hour, weekend flag

### Learner (new, to build)
- After each tweet resolves (24h), recompute:
  - Logistic regression coefficients (new data = better model)
  - Calibration curve (is the model under/overconfident?)
  - Example-based feedback patterns (what structural differences actually matter?)
- Weekly: retrain full model with all resolved predictions

## What we have

| Component | Status | Notes |
|-----------|--------|-------|
| Market research agent | Built | `market_research.py` |
| Content agent | Built | `content_team.py` |
| PostProphet harness | Built | Logistic regression, Brier 0.20 |
| Example-based feedback | Built | Prompt updated today |
| X write API | NOT built | Needed for auto-posting |
| Learner (self-improvement) | NOT built | The missing piece |
| Tracking (96 accounts) | Working | Daily cron, 16:00 UTC |
| Data (600+ resolved) | Working | ~28% hit rate |

## What we need to build

### 1. X Write API
- OAuth credentials for @PostProphet
- Post tweets from the content loop
- Track engagement after posting

### 2. Learner
- After each tweet resolves, update:
  - Logistic regression coefficients
  - Calibration data
  - Feedback patterns
- Weekly full retrain
- The model gets better every day

### 3. Content Loop Integration
- Connect market_research.py → content_team.py → postprophet.py → poster
- The loop runs daily at 16:00 UTC (after the tracking cron)
- Outputs: ranked draft tweets + predictions + feedback
- Posts the best one

## What we're NOT building

- Miner PR pipeline (premature, gameable)
- GitHub Actions for PR eval (not needed)
- Approaches directory (single product, not a platform)
- Causality eval gates (nice metric, not the product)
- Dual-track scoring (overengineering)

## The Product

The product is @PostProphet posting good tweets every day. The eval harness is how it improves. The metrics (Brier, causality) are internal diagnostics, not the product.

If a piece of it benefits from distributed competition later, we can add Gittensor then. But the product comes first. Prove it works. Then scale.

## Next steps

1. X write API (OAuth for @PostProphet)
2. Learner (auto-retrain after each resolution)
3. Content loop integration (connect the pieces)
4. Run daily at 16:00 UTC

That's it. Simple. Self-improving. Produces good content.
