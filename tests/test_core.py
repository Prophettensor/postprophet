"""Core tests: reward function, idea fit, learner bandit, pipeline pairing."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from postprophet.context import Seed, validate_seed
from postprophet.config import VisionConfig, InstanceConfig
from postprophet.ideas import surface_ideas, _fit_score
from postprophet.learner import Learner
from postprophet.reward import raw_reward, reward_per_impression


# ---------- reward ----------

def test_reward_weights_conversation_over_likes():
    # 1 quote (weight 5) should outrank 9 likes (weight 0.5 each = 4.5)
    a = {"quote_count": 1, "reply_count": 0, "bookmark_count": 0,
         "retweet_count": 0, "like_count": 0, "impression_count": 100}
    b = {"quote_count": 0, "reply_count": 0, "bookmark_count": 0,
         "retweet_count": 0, "like_count": 9, "impression_count": 100}
    assert raw_reward(a) > raw_reward(b)


def test_reward_per_impression_normalizes_by_reach():
    small = {"quote_count": 1, "like_count": 0, "impression_count": 100}
    big = {"quote_count": 2, "like_count": 0, "impression_count": 200}
    # same conversion ratio (1%) -> same per-impression reward
    assert reward_per_impression(small) == pytest.approx(reward_per_impression(big))
    assert reward_per_impression({}) == 0.0


# ---------- context / seeds ----------

def test_seed_contract_validates():
    good = {"source": "git", "kind": "ship", "title": "shipped X"}
    assert validate_seed(good).kind == "ship"
    with pytest.raises(ValueError):
        validate_seed({"source": "git", "kind": "bogus", "title": "x"})
    with pytest.raises(ValueError):
        validate_seed({"source": "git", "kind": "ship", "title": "   "})


# ---------- ideas ----------

def test_fit_source_affinity_and_avoid():
    vision = VisionConfig(
        positioning="postprophet content agents that learn from real engagement",
        problem="builders have no marketing arm",
        audience=["builders"],
        avoid=["gaming the algorithm"],
    )
    ship = Seed(source="git:postprophet", kind="ship",
                title="shipped postprophet framework")
    # source matches vision => high fit
    assert _fit_score(ship, vision)[0] >= 0.4
    # avoid topic (>=2 shared tokens with "gaming the algorithm") is floored
    avoided = Seed(source="git:foo", kind="finding",
                   title="we game the algorithm for reach")
    assert _fit_score(avoided, vision)[0] < 0.2


def test_surface_ideas_sorts_by_fit():
    vision = VisionConfig(positioning="content agents", problem="marketing for builders")
    seeds = [
        Seed(source="git:postprophet", kind="ship", title="shipped agent"),
        Seed(source="git:unrelated", kind="ship", title="totally different thing"),
    ]
    ideas = surface_ideas(seeds, vision, limit=5)
    assert ideas[0]["fit"] >= ideas[1]["fit"]
    assert "angles" in ideas[0] and ideas[0]["primary_angle"]


# ---------- learner bandit ----------

def test_learner_reweights_from_outcomes():
    tmp = tempfile.mkdtemp()
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "outcomes.jsonl"), "w") as f:
        for _ in range(3):
            f.write(json.dumps({"strategy": "reply_starter", "network_lever": "mutual",
                                "metrics": {"reply_count": 8, "quote_count": 1,
                                            "bookmark_count": 2, "retweet_count": 2,
                                            "like_count": 20, "impression_count": 1000}})
                       + "\n")
        for _ in range(3):
            f.write(json.dumps({"strategy": "share_worthy", "network_lever": "out_of_network",
                                "metrics": {"reply_count": 0, "quote_count": 0,
                                            "bookmark_count": 0, "retweet_count": 1,
                                            "like_count": 5, "impression_count": 1500}})
                       + "\n")
    l = Learner(tmp)
    state, n = l.update()
    assert n == 6
    scores = l.strategy_scores()
    assert scores["reply_starter"]["mutual"]["mean"] > scores["share_worthy"]["out_of_network"]["mean"]
    assert scores["reply_starter"]["mutual"]["n"] == 3


# ---------- pipeline pairing ----------

def test_plan_pairs_strategy_with_idea(tmp_path):
    from postprophet.config import LoopConfig
    cfg = InstanceConfig(name="t", loop=LoopConfig())
    from postprophet.generator import choose_generation_plan
    l = Learner(str(tmp_path))
    ideas = [
        {"seed": Seed(source="git:a", kind="finding", title="found X").to_dict(),
         "angles": ["pattern lesson"], "primary_angle": "pattern lesson", "fit": 0.9},
        {"seed": Seed(source="git:b", kind="ship", title="shipped Y").to_dict(),
         "angles": ["announcement"], "primary_angle": "announcement", "fit": 0.8},
    ]
    plan = choose_generation_plan(l, ideas, n=3)
    assert len(plan) >= 1
    for item in plan:
        assert "strategy" in item and "idea" in item


# ---------- publish modes ----------

def test_publish_modes_valid_and_invalid():
    from postprophet.config import AccountConfig
    for mode in ("manual", "approve", "auto"):
        assert AccountConfig(publish_mode=mode).publish_mode == mode
    import pytest as _p
    with _p.raises(ValueError):
        AccountConfig(publish_mode="coach")  # old name rejected
