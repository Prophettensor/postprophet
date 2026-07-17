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

**Idea:** Automatically suggest new accounts to track based on who replies to and quotes tweets from existing tracked accounts.

**Why it matters:** Manual curation doesn't scale. But pure keyword search pulls in too much noise. Tracked replies are the highest-signal discovery method — if an account replies to multiple tracked accounts, they're clearly in the ecosystem.

**How it would work:**
- Scan recent tracked replies/quotes (we already fetch these)
- Count how many different tracked accounts each replier engages with
- Suggest accounts that engage with 3+ tracked accounts and have >500 followers
- Command: `python postprophet.py suggest` — prints a list of candidate accounts with engagement stats
- User reviews and approves: `python postprophet.py add @suggested_handle`
- Still curated (human approves) but discovery is automated

**Why not now:**
- 40 accounts is plenty for the current eval
- Need more resolved predictions before expanding the ecosystem matters
- Diluting the ecosystem with noise would hurt the Brier score

**Trigger for building:** When expanding beyond the initial Bittensor niche, or when the 40-account ecosystem feels too small for accurate predictions.

