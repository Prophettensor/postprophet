# Contributing to PostProphet

PostProphet is a tweet reach prediction harness. It predicts the probability a tweet will hit 2x the author's median impressions within 24 hours. Contributions are evaluated automatically via Brier score.

## What you can change

**Only `environments/` files.** This is the scored surface — the scoring dimensions, their descriptions, the probability guide, and extra considerations.

You **cannot** change:
- `postprophet.py` (eval logic)
- `run_duel.py` (eval runner)
- `data/eval_set.jsonl` (frozen eval set)
- `scripts/` (CI scripts)
- `.github/` (workflow definitions)

PRs that touch protected files are automatically rejected.

## How to submit a PR

1. Fork the repo
2. Create a branch: `git checkout -b experiment/your-hypothesis`
3. Edit `environments/twitter.json`
4. Commit and push
5. Open a PR with a description of your hypothesis

## What happens when you open a PR

### Stage 1: Smoke test (instant)

Runs automatically on PR open. Uses the historical eval set (50 randomly sampled resolved tweets, temperature 0 for determinism).

- Validates your config (JSON structure, token cap <10K, no protected file changes)
- Runs eval: your config vs current config on the same 50 tweets
- Comments on your PR with full results: Brier scores, delta, per-tweet breakdown, results hash
- If regression > 0.05: PR auto-closed
- If passes: PR gets `batch-queued` label

### Stage 2: Batch eval (forward-looking, ~46 hours)

Runs daily at noon EST. Uses fresh tweets resolved in the last 24 hours — tweets that didn't exist when you submitted your PR.

- Collects all `batch-queued` PRs
- Runs each PR config + current config on the same fresh tweets
- The best PR with delta > 0.01 merges automatically
- Others re-evaluate next batch against the new baseline
- PRs that don't merge after 3 batches are closed

This is forward-looking eval. You can't overfit because the outcomes didn't exist when you submitted.

## What makes a good PR

- **Be specific.** "Make specificity description more forceful about requiring concrete numbers" is good. "Improve the prompt" is bad.
- **One change at a time.** Don't change 5 dimensions at once. You won't know which one helped.
- **Use 0.0-1.0 format.** The probability field expects floats, not percentages. Write `0.70` not `70%`.
- **Keep the JSON structure.** Same keys, same types. Add dimensions, don't remove them unless your hypothesis says to.
- **Test locally first.** Clone, set `POSTPROPHET_TEMPERATURE=0`, run `python run_duel.py`. Compare your hash to the PR comment.

## How to verify results

Every eval produces a SHA256 hash of the raw LLM responses. To verify:

```bash
git checkout <pr-branch>
POSTPROPHET_TEMPERATURE=0 python run_duel.py
# Compare the hash in the output to the hash in the PR comment
```

Temperature 0 makes outputs deterministic. Same config + same eval set + same model + temp 0 = same hash.

## Current stats

- Resolved predictions: 600+
- Brier score: ~0.21 (coin flip is 0.25)
- Eval set: 50 tweets, randomly sampled, rotated daily
- Model: gpt-4o-mini
- Cost per eval: ~$0.02

## Questions?

Open an issue or check the docs:
- `docs/PR_EVAL.md` — full eval pipeline design
- `docs/exploits.md` — known attack vectors and defenses
- `docs/trustless-maintainer.md` — trust model and anti-gaming measures
