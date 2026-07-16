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
