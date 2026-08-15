"""Tests for the derivation layer + agent_memory connector."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from postprophet import derive
from postprophet.config import VisionConfig, VoiceConfig
from postprophet.connectors.agent_memory import AgentMemoryConnector


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def test_extract_voice_from_memory(tmp_path):
    mem = "NO emojis.\nno hashtags ever\nkeep it dry, no fluff\n"
    v = derive.extract_voice(mem)
    assert "emojis" in v.do_not
    assert "hashtags" in v.do_not
    assert "no fluff" in v.tone


def test_explicit_voice_wins(tmp_path):
    mem = "NO emojis."
    explicit = VoiceConfig(tone="warm", do_not=["hashtags"], max_chars=200)
    v = derive.extract_voice(mem, explicit)
    assert v.tone == "warm"
    assert "emojis" in v.do_not  # memory adds it
    assert "hashtags" in v.do_not  # explicit keeps it
    assert v.max_chars == 200


def test_derive_vision_from_readme(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# My Product\n\nDoes X for Y.")
    vision = derive.derive_vision(repo_paths=[str(repo)])
    assert "My Product" in vision.positioning or "Does X" in vision.positioning


def test_agent_memory_connector_reads_lines(tmp_path):
    p = _write(tmp_path, "MEMORY.md", "NO emojis.\n§\nBuild the marketing layer.\n")
    c = AgentMemoryConnector(paths=[p])
    seeds = c.collect()
    titles = " ".join(s["title"] for s in seeds)
    assert "marketing layer" in titles
    assert all(s["kind"] == "finding" for s in seeds)


def test_pipeline_derive_mode_fills_gaps(tmp_path):
    from postprophet.pipeline import Pipeline
    from postprophet.config import InstanceConfig, LoopConfig, ConnectorConfig, AccountConfig

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# The Ship Engine\n\nMarkets what agents build.")
    mem = _write(tmp_path, "MEMORY.md", "NO emojis.\nbuilders ship all day, no marketing arm\n")

    cfg = InstanceConfig(
        name="t", loop=LoopConfig(derive=True),
        account=AccountConfig(publish_mode="manual"),
        connectors=[ConnectorConfig(name="git", type="git", options={"repos": [str(repo)]})],
    )
    pipe = Pipeline(cfg, str(tmp_path / "data"))
    # vision should have been filled from the README, voice from memory
    assert "Ship Engine" in pipe.config.vision.positioning or "Markets what agents" in pipe.config.vision.positioning
    assert "emojis" in pipe.config.voice.do_not
