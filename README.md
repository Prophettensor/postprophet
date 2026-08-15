# Content Agent (coach mode)

A self-improving X content agent. Watches what you're *actually building*, turns
real work into content ideas, drafts posts engineered toward the actions X's
open-sourced FYP algorithm ranks highest, and learns which strategies work from
real measured engagement.

Built from the signal in `xai-org/x-algorithm` (the For You feed code X open-
sourced Aug 2026): the ranking action-weights in `home-mixer/params/param.rs`.

## The loop

```
harvester (real work -> grounded seeds)     [zero LLM]
   -> generator (pick strategy + levers)     [zero LLM]
   -> YOU draft / review / post              [LLM step, in-session]
   -> tracker (read real X engagement)       [zero LLM cron, every 12h]
   -> learner (bandit reweights strategies)  [zero LLM cron]
   -> repeat, favoring what actually works
```

## The insight

PostProphet tried to *predict* reach from text dimensions — plateaued at Brier
0.20 because impressions are controlled by the algorithm, not the text. A content
agent doesn't predict; it *generates* and controls what gets posted. So the best
feedback is real measured engagement, and X's open weights give it the reward
function:

- reply 5.0 (->20.0 mutual), quote 5.0, share 2.0, follow 4.0, favorite 0.5
- structural: out-of-network discount 0.75, author-diversity decay 0.5

## Files

- `xweights.py` — reward function encoded from X's shipped params
- `strategies.py` — the playbook (reply_starter, quote_worthy, share_worthy,
  dwell_holder, follow_bait) x (mutual, out_of_network)
- `harvester.py` — watches /opt/data repos+plans, emits grounded content seeds
- `generator.py` — picks which (strategy, lever) combos to generate next
- `learner.py` — Thompson-sampling bandit, reweights from real outcomes
- `tracker.py` — reads real X engagement via xurl (no_agent cron)
- `run.py` — interactive drafting round (LLM step)

## Commands

    python3 run.py --top 5     # start a drafting round (LLM step, in-session)
    python3 tracker.py         # read engagement + retrain (or via cron)
    python3 learner.py --update  # manual retrain
    python3 harvester.py       # see current grounded seeds

## Cron

`content-agent tracker + learner` (no_agent, every 12h, zero LLM tokens).
Runs `~/.hermes/scripts/content_agent_tracker.sh`, silent when nothing new.

## Setup needed

xurl auth for the posting account (one-time, manual):
    HOME=/opt/data/home xurl auth apps add <app> --client-id ... --client-secret ...
    HOME=/opt/data/home xurl auth oauth2 --app <app> <handle>
Until authed, tracker exits silently and nothing records.
