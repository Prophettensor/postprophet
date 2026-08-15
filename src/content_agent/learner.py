"""
Learner — multi-armed bandit over generation strategies, rewarded by REAL measured
engagement (not LLM self-scoring).

Non-circular by construction:
    - REWARD comes from real platform metrics of actually-posted tweets.
    - OBJECTIVE prior comes from X's shipped ranking weights (strategies.py).
    - Nothing scores its own drafts with an LLM.

Each outcome records: strategy, network_lever, metrics, and the derived reward
(reward_per_impression). We maintain per-combination Thompson-sampling stats
(alpha/beta over reward-above-median vs below) and expose the value of each
(strategy, lever) so generation can exploit winners + explore unknowns.

Data lives under the instance dir (data/outcomes.jsonl, data/learner_state.json).
Deterministic and cheap — safe to run as a no_agent cron every tick.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from .reward import reward_per_impression
from .strategies import STRATEGIES, NETWORK_LEVERS

# Gamma prior pseudo-counts. Weak prior: first outcomes dominate; not zero so a
# strategy is never fully written off.
ALPHA0, BETA0 = 1.0, 1.0


class Learner:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.outcomes_path = os.path.join(data_dir, "outcomes.jsonl")
        self.state_path = os.path.join(data_dir, "learner_state.json")

    # ---- reading outcomes ----
    def _outcomes(self):
        if not os.path.exists(self.outcomes_path):
            return
        with open(self.outcomes_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_state(self, state):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    # ---- bandit update ----
    def update(self) -> tuple[dict, int]:
        outcomes = list(self._outcomes())
        if not outcomes:
            return {}, 0
        rewards = [reward_per_impression(o.get("metrics", {})) for o in outcomes]
        rewards = [r for r in rewards if r > 0] or rewards
        median_reward = sorted(rewards)[len(rewards) // 2]

        counts = defaultdict(lambda: {"alpha": ALPHA0, "beta": BETA0, "n": 0, "sum": 0.0})
        for o in outcomes:
            key = (o.get("strategy", "unknown"), o.get("network_lever", "out_of_network"))
            reward = reward_per_impression(o.get("metrics", {}))
            c = counts[key]
            c["n"] += 1
            c["sum"] += reward
            if reward >= median_reward:
                c["alpha"] += 1
            else:
                c["beta"] += 1

        state = {
            "median_reward": median_reward,
            "strategies": {f"{s}|{l}": c for (s, l), c in counts.items()},
            "n_outcomes": len(outcomes),
        }
        self._save_state(state)
        return state, len(outcomes)

    # ---- value estimates ----
    def strategy_scores(self):
        """Return {strategy: {lever: {mean, n, prob_good}}} using current state.
        Falls back to the X-weight prior when no data exists."""
        state = self._load_state()
        strat = state.get("strategies", {})
        result = {}
        for sname, sdef in STRATEGIES.items():
            result[sname] = {}
            for lever in NETWORK_LEVERS:
                rec = strat.get(f"{sname}|{lever}")
                if rec and rec.get("n", 0) > 0:
                    mean = rec["sum"] / rec["n"]
                    prob_good = rec["alpha"] / (rec["alpha"] + rec["beta"])
                else:
                    mean = sdef["x_weight"] * 1e-4
                    prob_good = 0.5
                result[sname][lever] = {
                    "mean": mean,
                    "n": rec["n"] if rec else 0,
                    "prob_good": prob_good,
                }
        return result

    def _alpha_for(self, sname, lever):
        rec = self._load_state().get("strategies", {}).get(f"{sname}|{lever}")
        return rec["alpha"] if rec else ALPHA0

    def _beta_for(self, sname, lever):
        rec = self._load_state().get("strategies", {}).get(f"{sname}|{lever}")
        return rec["beta"] if rec else BETA0

    def _x_prior(self, sname, lever):
        base = STRATEGIES[sname]["x_weight"]
        return base * 1.0 if lever == "mutual" else base * 0.75

    def sample_next(self, rng=None):
        """Thompson-sample the next (strategy, lever) to generate."""
        rng = rng or random
        scores = self.strategy_scores()
        best, best_sample = None, -1.0
        for sname, levers in scores.items():
            for lever, stats in levers.items():
                a = self._alpha_for(sname, lever)
                b = self._beta_for(sname, lever)
                sample = rng.betavariate(a, b) if (a > 0 and b > 0) else 0.5
                sample += self._x_prior(sname, lever) * 1e-4
                if sample > best_sample:
                    best_sample = sample
                    best = (sname, lever)
        return best

    def expected_return(self, sname, lever="out_of_network"):
        return self.strategy_scores().get(sname, {}).get(lever, {}).get("mean", 0.0)

    def summary(self):
        scores = self.strategy_scores()
        n = len(list(self._outcomes()))
        lines = [f"resolved outcomes: {n}"]
        for sname, levers in scores.items():
            for lever, stats in levers.items():
                lines.append(
                    f"  {sname:16s} / {lever:14s} n={stats['n']:3d} "
                    f"mean_reward={stats['mean']:.5f} p_good={stats['prob_good']:.2f}"
                )
        return "\n".join(lines)
