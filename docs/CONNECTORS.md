# Writing a connector

A connector is how you connect PostProphet to your agent stack. It turns
discrete facts from any source into "seeds" the idea layer can surface. If you
have a source PostProphet doesn't ship a connector for — a message bus, a
Slack/Teams feed, a customer-success database, an internal API — you write a
thin adapter. No framework changes needed.

## The seed contract

A connector emits a list of seeds. A seed is one verifiable fact, with a fixed
shape:

```json
{
  "source": "git:my-repo",      // where it came from (you pick the label)
  "kind": "ship",               // one of: ship | finding | signal | event | milestone
  "title": "merged: add auth",  // one-line summary (used as the idea anchor)
  "detail": "PR #42 merged...", // fuller context for grounding (optional)
  "date": "2026-08-15",         // ISO date when it happened (optional)
  "tags": ["ship", "my-repo"]   // free-form topical tags (optional)
}
```

The `kind` field drives which content angles the idea layer can use:

| kind       | suggested angles                                    |
|------------|-----------------------------------------------------|
| `ship`     | announcement, build-in-public, milestone            |
| `finding`  | pattern lesson, thought-leadership, insight         |
| `signal`   | customer insight, market take, contrarian           |
| `event`    | timely take, reaction                               |
| `milestone`| milestone, recap                                    |

## The rules (this is what keeps content honest)

1. **Seeds carry facts only.** Interpretation — vision fit, market angle, which
   action to target — happens upstream in the idea layer, never in a connector.
2. **No fabrication.** The drafting LLM is told these seeds are ground truth to
   be framed, not invented. If a connector invents data, the whole pipeline lies.
3. **Fail safe.** A connector that errors should be skipped, not crash the round.
4. **Zero LLM.** Connectors are deterministic. They must never call a model —
   the whole point is that harvesting is free.

## Minimum implementation

A connector is any object with a `collect()` method that returns a list of seed
dicts (or `Seed` objects):

```python
# my_connector.py
from postprophet.context import Seed

class MyConnector:
    def __init__(self, endpoint=None, token=None):
        self.endpoint = endpoint
        self.token = token

    def collect(self):
        # ... fetch facts from your source ...
        return [
            Seed(
                source="my_source",
                kind="finding",
                title="customer reported X",
                detail="in the last support batch",
                tags=["customers"],
            ).to_dict()
        ]
```

## Registering it

Two ways:

1. **Add to a builder config** — if you want it configurable per instance, add
   the type to `postprophet.connectors.build_connector` and reference it in the
   config:
   ```yaml
   connectors:
     - name: customers
       type: my_source
       options:
         endpoint: "https://..."
   ```
2. **Wire directly in code** — pass any object with `collect()` to the pipeline:
   ```python
   from postprophet.pipeline import Pipeline
   pipe = Pipeline(cfg, data_dir)
   seeds = MyConnector().collect()          # call it yourself
   ideas = pipe.ideas(seeds=seeds)          # feed into the idea layer
   ```

## Pattern to copy

Look at `postprophet/connectors/agent_output.py` for the simplest reference
implementation (reads files, no network). `git.py` is the pattern for shell-based
sources; `rss.py` for network parsing. All follow the same contract.

## Testing

Test that your connector returns well-formed seeds and handles failure. See
`tests/test_connectors.py` for examples — note the github/rss tests mock the
network so they run offline and fast.
