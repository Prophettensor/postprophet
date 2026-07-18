# PostProphet — Future Enhancements

## Web of Trust / Trust Graph

**Idea:** Build an engagement-based trust graph across tracked accounts. Instead of just showing follower count + median impressions when someone replies/quotes, compute a centrality/trust score — "how many other tracked accounts engage with this person?"

**Why it matters:** A small account (500 followers) that's frequently engaged with by 8+ tracked accounts is more influential than their follower count suggests. Currently the model would discount them based on followers alone. A trust score would surface hidden influence.

**How it would work:**
- Track who replies to/quotes whom across the ecosystem over time
- Compute PageRank-like centrality score per account
- Show trust score alongside follower count + median impressions in the prompt
- A reply from a high-trust account boosts the prediction; from a low-trust account doesn't

**Why not now:**
- Requires accumulated engagement data over many track cycles
- Computing the graph adds complexity (engagement matrix, centrality algorithm, cache)
- We don't have enough resolved predictions yet to know if the current approach (follower count + median as proxy) works well enough
- Risk of becoming a Rube Goldberg machine before the basics are validated

**Trigger for building:** When we have 100+ resolved predictions AND the data shows the model is systematically misweighting small-but-influential accounts.

## Account Discovery from Tracked Replies

**Idea:** Automatically suggest new accounts to track based on who gets quoted by existing tracked accounts.

**Why it matters:** Manual curation doesn't scale. But pure keyword search pulls in too much noise. Tracked quotes are the highest-signal discovery method — if an account quotes someone, they're putting that person on their timeline. That's a trust signal.

**Status: BUILT.** `suggest` command works. Quoted accounts auto-logged during track/backfill. Manual review + add via `python postprophet.py add @handle`.

**Next step: automate the full lifecycle:**

### Auto-add (with cost guard)
- Cron runs `suggest` weekly
- Accounts quoted 3+ times by tracked accounts = auto-add candidates
- Before adding: fetch their recent tweets to verify they post at least 1x/week
- If they don't tweet enough, skip — not worth ongoing track cost
- Hard cap: 100 accounts max (cost ceiling)

### Auto-prune (dead accounts)
- Track last-tweet-date per account during each track run
- 14 days no tweets → flag in report
- 30 days no tweets → auto-remove from accounts.txt (log to data repo for audit)
- Dead accounts waste ~$0.06 per track run each

### Unuseful data detection
- Flag accounts where Brier is significantly worse than overall average
- Flag accounts with 0 variance (always hit or always miss)
- Flag accounts with no engagement (no quotes, no replies from other tracked accounts)
- Don't auto-remove these — just flag for manual review

### Cost budget
- Track run cost: ~$0.01 per account (user lookup + 20 tweet reads)
- 100 accounts = ~$1.00 per cold track run, ~$0.21 warm
- Daily cron = ~$30/month max
- Adding accounts increases cost linearly — pruning keeps it bounded

**Trigger for building:** After 1 week of cron data showing which accounts actually produce predictions.

