# PostProphet — PR Evaluation Pipeline

## Overview

Trustless, forward-looking merge process. PRs are evaluated on tweets that don't exist when the PR is submitted. No overfitting possible.

## Two-stage eval

### Stage 1: Smoke test (instant, gameable, just a gate)

When a PR opens, run the historical eval set (50 frozen tweets, temp=0).

Purpose: catch broken configs, regressions, and garbage before wasting 24h.

- If config is invalid JSON → auto-close
- If probability not 0.0-1.0 → auto-close
- If Brier regresses by >0.05 → auto-close with "regression on historical set"
- If Brier improves or is within noise → enters batch queue

This stage IS gameable (miners can see the historical eval set). That's fine — it's just a smoke test. The real decision is Stage 2.

### Stage 2: Batch eval (24h, forward-looking, ungameable)

PRs that pass Stage 1 enter a batch queue. At cron time (noon EST daily):

1. Collect all queued PRs
2. The cron has been predicting fresh tweets all day with the current (main) config
3. These tweets are now 0-24h old, resolving throughout the day
4. At next cron (24h later), all predictions have resolved
5. Run EACH queued PR config on the SAME fresh tweets (temp=0)
6. Compute Brier for: current config, each PR config
7. The PR with the lowest Brier that beats current by >0.01 → MERGE
8. Others → re-evaluate tomorrow against new baseline
9. PRs that don't merge after 3 batches → close with "insufficient improvement"

The fresh tweets didn't exist when the PR was submitted. No overfitting possible.

## Batch mechanics

- Batch window: 24h (noon EST to noon EST)
- Minimum tweets per batch: 10 (if fewer, skip batch, wait another day)
- All PRs in the same batch are evaluated on the same tweets → fair comparison
- Only ONE PR merges per batch (the best one)
- Complementary changes: if PR B still improves after PR A merges, it merges next batch

## Copy/tweak protection

If PR B is a minor tweak of PR A (detected by config diff similarity):

- Both enter the same batch
- Only the better one merges
- The copy gets nothing
- No explicit detection needed — the batch comparison handles it naturally

If two PRs are genuinely complementary (different dimensions changed):
- Best one merges first
- Other re-evaluates against new baseline next batch
- If it still improves → merges too
- Stack naturally

## Anti-gaming

| Attack | Defense |
|--------|---------|
| Overfit to historical eval set | Stage 2 uses fresh tweets that didn't exist at PR time |
| Copy/tweak another PR | Batch comparison: only best merges |
| Submit many PRs to brute-force | 1 PR per contributor per batch queue |
| Change the model secretly | Model field is checked, must match baseline |
| Inflate prompt tokens | Config diff reviewed, token cap (10K) enforced in Stage 1 |
| Modify eval set | Eval set hash checked, PRs touching eval_set.jsonl auto-closed |

## Cost

- Stage 1: $0.02 per PR (50 LLM calls, temp=0)
- Stage 2: $0.02 per PR per batch (fresh tweets, temp=0)
- Total per PR: $0.04 if it merges in one batch, $0.06-0.10 if it takes multiple batches
- Daily cron cost: $0.02 (baseline predictions already running)

## Timeline

```
Day 0, 2pm:  Miner submits PR
Day 0, 2pm:  Stage 1 smoke test runs (instant)
Day 0, 2pm:  Passes → enters batch queue
Day 1, noon: Cron collects batch, fresh tweets being predicted
Day 2, noon: Fresh tweets resolved, Stage 2 eval runs
Day 2, noon: Best PR merges (if delta > 0.01)
```

Total: ~46 hours from PR to merge decision. Forward-looking. Ungameable.

## Implementation

| Component | Status |
|-----------|--------|
| Historical eval set (Stage 1) | ✅ Built |
| run_duel.py (per-tweet results + hash) | ✅ Built |
| Temperature 0 | ✅ Built |
| Probability validation | ✅ Built |
| Dockerfile | ✅ Built |
| GitHub Action (Stage 1 smoke test) | 🔜 Building now |
| Batch queue (Stage 2) | 🔜 Building |
| Fresh tweet collection for Stage 2 | 🔜 Building (cron already tracks) |
| Auto-merge on delta > 0.01 | 🔜 Building |
| Duels dashboard update | 🔜 Building |

## What the GitHub Action does

### On PR open:

```yaml
1. Checkout PR branch
2. Validate config (JSON, structure, token cap)
3. Run Stage 1: historical eval (temp=0, Docker)
4. If regression > 0.05: close PR, comment results
5. If passes: comment "passed smoke test, entering batch queue"
6. Add label "batch-queued"
```

### Daily cron (noon EST):

```yaml
1. Collect all PRs with "batch-queued" label
2. Get fresh resolved predictions from last 24h (cron already has these)
3. For each PR: checkout branch, run eval on fresh tweets (temp=0, Docker)
4. Also run current (main) on same fresh tweets
5. Compare Brier scores
6. Best PR with delta > 0.01: merge, update duels dashboard, bump version
7. Others: comment results, keep in queue for next batch
8. PRs in queue >3 batches: close with "insufficient improvement"
```
