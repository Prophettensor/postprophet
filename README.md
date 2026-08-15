# PostProphet

**The marketing layer for what you ship.**

Builders and coding agents ship all day. Nobody markets the output — the work
disappears into commits and merge requests. PostProphet takes the continuous
stream of what your agents produce and turns it into a publishing presence that
learns from real engagement.

It connects to your agent stack, understands the vision and market you're within,
surfaces ideas, generates content, posts it, and learns from performance.

## The idea in one line

An agent that watches what you actually build, turns it into content engineered
toward the actions the platform ranks highest, and gets smarter from real
measured results — not vibes.

## Build vs use (the design split)

- **The product (this repo)** is a stack-agnostic framework: connectors, an idea
  layer, a generation planner, a publisher, and a learner. It doesn't care whose
  stack or what you ship.
- **A builder instance** is ONE config file wiring your stack connectors, your
  vision/market profile, your account, and your voice. That's "using the
  product."

A builder adopting this never touches the framework — they copy
`example-config.yaml`, fill in their stack and vision, and run the loop.

## Why the timing is right

X open-sourced its For You feed ranking algorithm (`xai-org/x-algorithm`, Aug
2026). The action weights in `home-mixer/params/param.rs` tell us what the
platform values: **reply 5.0 (→20 mutual), quote 5.0, share 2.0, follow 4.0,
favorite 0.5** — plus structural levers (out-of-network discount 0.75,
author-diversity decay 0.5). That gives the learn-from-performance loop a real,
stable objective instead of a guess.

## The loop

```
YOUR AGENT STACK   (git/CI, research, customer success, ecosystem, webhook...)
        |  outputs & events
        v
HARVEST   connectors emit grounded "seeds" (facts, never invented)
        v
IDEAS     surface + rank ideas against your vision/market      [zero LLM]
        v
PLAN      pair winning strategies with best-fit ideas (bandit)  [zero LLM]
        v
DRAFT     agent turns idea+strategy into post text              [LLM, in-session]
        v
PUBLISH   post — manual (you post by hand), approve (you approve, then auto-post),
          or auto (agent posts on schedule)
        v
LEARN     real engagement reweights strategies                  [zero-LLM cron]
        v
repeat, favoring what actually works
```

Non-circular by construction: reward comes from real platform metrics, the
objective prior comes from the platform's shipped weights, and nothing scores
its own drafts with an LLM.

## Install

```bash
pip install -e .
```

## Use

```bash
postprophet init                       # write an example config
postprophet round --config your.yaml --top 5   # drafting brief (LLM step follows)
postprophet measure --config your.yaml         # read outcomes + retrain (cron)
postprophet ideas --config your.yaml           # see surfaced ideas only
```

- `round` is the interactive step (needs an LLM — run it when you want drafts).
- `measure` is the zero-LLM cron step — wire it to a schedule. It's silent when
  there's nothing new, so a no_agent cron works without burning tokens.

## Connectors (the portability layer)

The product connects to any stack through a small, plain "seed" contract:

```json
{ "source": "git", "kind": "ship", "title": "shipped X",
  "detail": "...", "date": "2026-08-15", "tags": [] }
```

Shipped connectors:

- **git** — emits `ship` seeds from recent commits (every builder has this).
- **github** — emits `ship`/`milestone` seeds from merged PRs and releases in a repo.
- **agent_output** — reads files any agent drops (research findings, customer
  logs, ecosystem monitors) as `finding`/`signal` seeds.
- **rss** — emits `signal`/`event` seeds from feeds — competitor blogs, ecosystem
  announcements, news. The "understand your market" input.
- **webhook** — a tiny local HTTP endpoint so any agent can POST a seed.

Builders write their own thin adapter for stack-specific sources — it's a small
callable returning a list of seed dicts. No framework changes needed. See
`docs/CONNECTORS.md` for the connector-authoring guide.

## Strategies (the playbook)

Each draft is tagged with the action it's engineered to elicit, from X's weights:

`reply_starter` · `quote_worthy` · `share_worthy` · `dwell_holder` · `follow_bait`
each × `mutual` / `out_of_network`

The learner (Thompson-sampling bandit) reweights these from real outcomes, so
the playbook is a prior, not a rule.

## Architecture

```
src/postprophet/
  context.py     seed contract + validation
  config.py      builder-instance config schema
  connectors/    git, github, agent_output, rss, webhook
  ideas.py       surface + rank ideas against vision/market
  generator.py   plan strategies x ideas, build draft prompts
  learner.py     bandit over strategies, rewarded by real outcomes
  reward.py      X action weights as the objective prior
  tracker.py     read real engagement via xurl (no_agent cron)
  pipeline.py    the public API / orchestration
  cli.py         round / measure / ideas / init
```

## License

MIT
