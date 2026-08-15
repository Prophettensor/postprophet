"""Tests for multi-project scoping (for_project / projects config)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from postprophet.config import (InstanceConfig, ProjectConfig, LoopConfig,
                                ConnectorConfig, AccountConfig, VisionConfig,
                                VoiceConfig)
from postprophet.pipeline import Pipeline


def _make_config(tmp_path, repos_by_project):
    """Build a multi-project config with one or more projects."""
    projects = []
    for name, repos in repos_by_project.items():
        for rp in repos:
            os.makedirs(rp, exist_ok=True)
            readme = rp / "README.md"
            if not readme.exists():
                readme.write_text(f"# {name}\n\nBuilds {name} for people.")
        projects.append(ProjectConfig(name=name, repos=[str(r) for r in repos]))
    return InstanceConfig(
        name="multi",
        loop=LoopConfig(),
        account=AccountConfig(publish_mode="manual"),
        vision=VisionConfig(),
        projects=projects,
    )


def test_for_project_scopes_connectors_to_that_repo(tmp_path):
    proj_a = tmp_path / "alpha"
    proj_b = tmp_path / "beta"
    cfg = _make_config(tmp_path, {"alpha": [proj_a], "beta": [proj_b]})
    pipe = Pipeline(cfg, str(tmp_path / "data"))

    scoped = pipe.for_project("alpha")
    # alpha repo is in scope
    repo_paths = scoped._repo_paths()
    assert any(str(proj_a) in r for r in repo_paths)
    assert not any(str(proj_b) in r for r in repo_paths)


def test_for_project_derives_own_vision_from_its_repo(tmp_path):
    proj_a = tmp_path / "alpha"
    proj_b = tmp_path / "beta"
    (tmp_path / "alpha").mkdir(parents=True, exist_ok=True)
    (tmp_path / "beta").mkdir(parents=True, exist_ok=True)
    (proj_a / "README.md").write_text("# Alpha Engine\n\nMarkets agent ships.")
    (proj_b / "README.md").write_text("# Beta Data\n\nTracks token flows.")
    cfg = _make_config(tmp_path, {"alpha": [proj_a], "beta": [proj_b]})
    pipe = Pipeline(cfg, str(tmp_path / "data"))

    a = pipe.for_project("alpha")
    b = pipe.for_project("beta")
    assert "Alpha" in a.config.vision.positioning
    assert "Beta" in b.config.vision.positioning
    # vision is per-project, not shared
    assert a.config.vision.positioning != b.config.vision.positioning


def test_voice_stays_global_across_projects(tmp_path):
    proj_a = tmp_path / "alpha"
    proj_b = tmp_path / "beta"
    cfg = _make_config(tmp_path, {"alpha": [proj_a], "beta": [proj_b]})
    cfg.voice = VoiceConfig(tone="direct, no fluff", do_not=["emojis"], max_chars=280)
    pipe = Pipeline(cfg, str(tmp_path / "data"))

    a = pipe.for_project("alpha")
    b = pipe.for_project("beta")
    assert a.config.voice.do_not == b.config.voice.do_not == ["emojis"]
    assert a.config.voice.tone == b.config.voice.tone == "direct, no fluff"


def test_for_project_unknown_raises(tmp_path):
    proj_a = tmp_path / "alpha"
    cfg = _make_config(tmp_path, {"alpha": [proj_a]})
    pipe = Pipeline(cfg, str(tmp_path / "data"))
    with pytest.raises(KeyError):
        pipe.for_project("nope")


def test_project_scoped_round_produces_grounded_ideas(tmp_path):
    proj_a = tmp_path / "alpha"
    (tmp_path / "alpha").mkdir(parents=True, exist_ok=True)
    (proj_a / "README.md").write_text("# Alpha Engine\n\nMarkets what agents ship.")

    # a git repo so the project's git connector has a seed
    from subprocess import run
    for cmd in [["git", "init", "-q"],
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit",
                 "--allow-empty", "-m", "ship alpha v2"]]:
        run(cmd, cwd=str(proj_a), capture_output=True)

    cfg = _make_config(tmp_path, {"alpha": [proj_a]})
    pipe = Pipeline(cfg, str(tmp_path / "data"))
    scoped = pipe.for_project("alpha")
    seeds = scoped.harvest()
    # project's git seed is present and grounded
    assert any("ship alpha" in s.title for s in seeds)
    ideas = scoped.ideas(seeds)
    assert len(ideas) >= 1
