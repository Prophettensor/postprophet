# SocialQuant

A prediction engine for social media reach.

Given who is posting, what they're posting, where they're posting, and a target — SocialQuant predicts the probability of hitting that target within a given timeframe. It's the technology layer that lets agents grade their own content before it goes live.

## How it works

```
Agent generates tweet
    ↓
SocialQuant predicts: 72% chance of 10k impressions in 24h
    ↓
Reasoning: "hook is weak, trending topic is fading, post at 9am instead"
    ↓
Agent modifies tweet
    ↓
SocialQuant re-predicts: 85% chance
    ↓
Ship when probability clears threshold
```

## The eval

Live, real-time, ungameable.

1. X API filtered stream captures real tweets as they're posted
2. SocialQuant predicts impression probability at 1h, 4h, 24h
3. Impressions resolve naturally
4. Brier score measures prediction accuracy
5. PRs to the engine are tested against the same tweet batch — does accuracy improve?

No sandbox. No historical replay. The engine gets the same data a real marketing agent would have.

## Get started

SocialQuant is in development. The prediction engine and eval harness are being built.

```bash
git clone https://github.com/buckZz7/socialquant.git
cd socialquant
```

## Project structure

```
socialquant/
├── NAPKIN.md         # Project vision
├── README.md         # This file
├── docs/
│   └── index.html    # Landing page
└── LICENSE           # MIT
```

## License

MIT
