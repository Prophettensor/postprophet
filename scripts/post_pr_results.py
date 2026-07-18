"""Post eval results as a PR comment + auto-close/label."""
import json, os, urllib.request, sys

with open('data/duel_results.json') as f:
    results = json.load(f)

delta = results['delta']
baseline = results['baseline_brier']
challenger = results['challenger_brier']
rhash = results.get('results_hash', 'unknown')

status = 'WITHIN NOISE'
if delta > 0.01:
    status = 'IMPROVEMENT'
elif delta < -0.05:
    status = 'REGRESSION'

per_tweet = results.get('per_tweet', [])
tweet_rows = '\n'.join(
    f'| @{t["author"]} | {t["target"]:,} | {t["actual_impressions"]:,} | {"YES" if t["hit"] else "NO"} | {round(t["baseline_prob"]*100)}% | {round(t["challenger_prob"]*100)}% |'
    for t in per_tweet
)

verdict = 'REGRESSION - PR auto-closed' if delta < -0.05 else 'Entered batch queue. Stage 2 runs at next cron.'

body = f"""## Stage 1: Smoke Test Results

| | Current | Challenger |
|---|---|---|
| Version | {results["baseline_version"]} | {results["challenger_hash"]} |
| Brier | {baseline} | {challenger} |

**Delta:** {delta}
**Status:** {status}
**Results hash:** `{rhash}`
**Temperature:** 0 (deterministic)
**Eval set:** {results["eval_size"]} tweets

{verdict}

<details>
<summary>Per-tweet breakdown</summary>

| Author | Target | Actual | Hit | Current | Challenger |
|---|---|---|---|---|---|
{tweet_rows}

</details>

To verify: checkout this branch, set `POSTPROPHET_TEMPERATURE=0`, run `python run_duel.py`, compare hash."""

token = os.environ['GITHUB_TOKEN']
pr_number = os.environ['PR_NUMBER']
repo = os.environ['GITHUB_REPOSITORY']

# Post comment
data = json.dumps({'body': body}).encode()
req = urllib.request.Request(
    f'https://api.github.com/repos/{repo}/issues/{pr_number}/comments',
    data=data,
    headers={
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json',
    },
)
resp = urllib.request.urlopen(req)
print(f'Comment posted: {resp.status}')

# Auto-close or label
if delta < -0.05:
    data = json.dumps({'state': 'closed'}).encode()
    req = urllib.request.Request(
        f'https://api.github.com/repos/{repo}/pulls/{pr_number}',
        data=data,
        headers={
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json',
        },
        method='PATCH',
    )
    urllib.request.urlopen(req)
    print(f'PR #{pr_number} closed (regression)')
else:
    data = json.dumps({'labels': ['batch-queued']}).encode()
    req = urllib.request.Request(
        f'https://api.github.com/repos/{repo}/issues/{pr_number}/labels',
        data=data,
        headers={
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    urllib.request.urlopen(req)
    print(f'PR #{pr_number} labeled batch-queued')
