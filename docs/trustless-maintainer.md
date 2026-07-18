# PostProphet — Trustless Autonomous Maintainer Design

## Goal

A system where PRs are evaluated, scored, and merged without human intervention. Miners submit config changes, the system runs the eval, and merges if the Brier improves. No one can game it because the merge decision is based on math, not politics.

## The Flow

```
Miner submits PR (config change to environments/twitter.json)
    ↓
GitHub Action triggers on PR open
    ↓
Step 1: Validate
    - Is the config valid JSON?
    - Are dimensions the same structure?
    - Did they change anything other than descriptions/probability_guide?
    - Token count of prompt < 10K?
    - If invalid: auto-close PR with reason
    ↓
Step 2: Smoke test
    - Run 1 prediction on a random eval tweet
    - If probability is not 0.0-1.0: auto-close PR
    - If crashes: auto-close PR
    ↓
Step 3: Baseline eval
    - Run current (main) config on frozen 50-tweet eval set
    - Record Brier + per-tweet results
    - (Cached if main hasn't changed since last PR)
    ↓
Step 4: Challenger eval
    - Run PR config on same frozen 50-tweet eval set
    - Record Brier + per-tweet results
    ↓
Step 5: Compare
    - Delta = baseline_brier - challenger_brier
    - If delta > 0.01: MERGE
    - If delta < -0.01: close PR with "regression" label
    - If |delta| <= 0.01: close PR with "no significant change" label
    ↓
Step 6: Post results
    - Comment on PR with full eval results
    - Update duels dashboard with new entry
    - If merged: bump version, update .version file
```

## Anti-gaming measures

### 1. Eval set integrity
- Eval set hash stored in `.version` file
- PRs that modify `data/eval_set.jsonl` are auto-rejected
- Eval set rotates every 30 days via a separate trusted process (not miner-submitted)

### 2. Holdout set
- Second eval set that miners never see
- PR eval runs on both public eval set AND holdout
- If public improves but holdout doesn't: overfitting detected, reject
- Holdout is refreshed quarterly

### 3. Token cap
- Prompt template token count is computed during validation
- If PR config causes prompt to exceed 10K tokens: auto-reject
- Prevents token inflation attacks

### 4. Model lock
- PR eval uses the same model as baseline (from `.version` file)
- PRs that change `POSTPROPHET_MODEL` are a different PR type
- Model changes require separate review (not auto-merged)

### 5. Rate limiting
- Max 1 PR per GitHub user per 7 days
- Max 5 open PRs at a time across all miners
- Prevents brute-force spam

### 6. Win rate tracking
- Track each miner's PR win/loss record
- Miners with <10% win rate after 5 PRs: flagged
- Flagged miners: PRs require manual review for next 30 days
- This is a soft disincentive, not a ban

## Implementation status

| Component | Status | Notes |
|-----------|--------|-------|
| Eval command | ✅ Built | `python postprophet.py eval` |
| Experiment runner | ✅ Built | `python postprophet.py experiment` |
| Version auto-hash | ✅ Built | `python postprophet.py version` |
| Probability validation | ✅ Built | Clamps to 0.0-1.0 |
| Per-tweet results | ✅ Built | `run_duel.py` saves all 50 |
| Raw LLM responses | ✅ Built | `data/duel_raw.json` with full predictions |
| Results hash (SHA256) | ✅ Built | Tamper-evident verification |
| Temperature 0 for eval | ✅ Built | `POSTPROPHET_TEMPERATURE=0` |
| GitHub PR creation | ✅ Built | Via API |
| Duels dashboard | ✅ Built | All 50 results + PR link + hash |
| Docker image | 🔜 Next | Reproducible environment |
| GitHub Action (auto-eval on PR) | 🔜 Next | Public CI logs, auto-merge |
| Eval set rotation | 🔜 Next | Daily refresh from fresh resolved predictions |
| Holdout eval set | ❌ Deferred | Need 200+ resolved predictions |
| TEE attestation | ❌ Deferred | Only needed for private holdout set |
| Token cap validation | ❌ Not built | Need prompt token counter |
| Rate limiting | ❌ Not built | Needs GitHub Action + state |
| Win rate tracking | ❌ Not built | Needs persistent state |
| Auto-merge | ❌ Not built | Needs GitHub Action with write access |

## Trust model

**Current (building now):** Docker + GitHub Actions + temp=0 + results hash
- Eval runs in public CI, logs are visible
- Same Docker image = same environment
- Temperature 0 = deterministic outputs
- SHA256 hash of raw results = tamper-evident
- Miner reproduces locally, compares hash

**Future (if needed):** TEE attestation
- Only necessary when we add a private holdout set miners can't see
- The enclave proves the eval ran honestly on hidden data
- Not needed now — eval set is public, transparency IS the trust

**Eval set rotation:** Daily refresh from fresh resolved predictions (after cron resolves at noon EST). Overfitting to today's eval set doesn't survive tomorrow's rotation. No need for a hidden holdout set yet.

## Next steps

1. **GitHub Action**: On PR open, run eval, comment with results, auto-merge if delta > 0.01
2. **Holdout set**: Once we have 200+ resolved predictions, split into public eval (50) + holdout (50)
3. **Contributing docs**: How to submit a PR, what changes are allowed, how eval works
4. **Contributing skill**: Hermes skill that walks a contributor through the process

## Philosophy

The merge decision is: "did this config change improve Brier on the frozen eval set by more than 0.01?" That's it. No opinions, no taste, no politics. Just math.

The system is trustless because:
- The eval set is frozen and hash-verified
- The Brier is computed from real tweet outcomes
- The model can't see the answers (no leakage)
- The config diff is public
- The per-tweet results are public
- The holdout set catches overfitting

A miner's only path to merge is: give the model better instructions that produce more accurate predictions on tweets it has never seen.
