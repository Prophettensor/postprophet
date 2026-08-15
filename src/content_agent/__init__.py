"""
Content Agent — marketing layer for what you ship.

A framework (not a single tool) that any builder plugs into their agent stack.
Watches the continuous stream of what your agents produce, understands your
vision and market, turns it into a publishing presence, and learns from real
engagement which content works.

Design split (the product vs using the product):
    - This package IS the product: stack-agnostic framework + connectors +
      the closed loop (harvest -> ideas -> generate -> publish -> learn).
    - A "builder instance" is the USE: one config file wiring their stack
      connectors, vision/market profile, account, and voice.

The reward function comes from xai-org/x-algorithm's shipped ranking weights
(home-mixer/params/param.rs), so the learn-from-performance loop has a real,
stable objective. Nothing here scores its own drafts with an LLM — reward comes
from real platform metrics, objective prior from X's shipped params.
"""

__version__ = "0.1.0"
