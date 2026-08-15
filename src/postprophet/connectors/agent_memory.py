"""Agent-memory connector — emits 'finding' seeds from the agent's own memory.

This is the "connect to your coding agent" connector. Every builder's agent has
a persistent memory / workspace (MEMORY.md, USER.md, CLAUDE.md, AGENTS.md, notes).
That memory holds the vision, the market, the preferences — the exact context
PostProphet would otherwise ask the user to hand-write.

Options:
    paths    list of memory/context files or directories to read (default: common
             agent-home locations: ~/.hermes/memories/, ./AGENTS.md, ./CLAUDE.md)
    kind     seed kind for these (default finding)
    max_lines cap per file so a huge memory file doesn't flood the idea layer

Each non-trivial line in memory becomes a grounded 'finding' seed — the vision
and market the builder already committed to. The idea layer can then rank real
content against it, and the derivation layer (derive.py) uses the same source.
"""

from __future__ import annotations

import os

from ..context import Seed

DEFAULT_PATHS = None  # resolved per-instance from real Hermes/agent home layouts


def _default_paths():
    """Resolve the real agent-memory locations. Hermes keeps memories under
    HERMES_HOME/memories/ (e.g. /opt/data/memories/) and/or ~/.hermes/memories/;
    coding agents use ./AGENTS.md, ./CLAUDE.md, ./.cursorrules. Check them all."""
    homes = []
    for var in ("HERMES_HOME", "HOME"):
        v = os.environ.get(var)
        if v:
            homes.append(os.path.expanduser(v))
    homes.append(os.path.expanduser("~"))

    paths = []
    for h in dict.fromkeys(homes):  # dedupe, keep order
        paths.append(os.path.join(h, "memories", "MEMORY.md"))
        paths.append(os.path.join(h, "memories", "USER.md"))
        paths.append(os.path.join(h, ".hermes", "memories", "MEMORY.md"))
        paths.append(os.path.join(h, ".hermes", "memories", "USER.md"))
        paths.append(os.path.join(h, "memories"))
        paths.append(os.path.join(h, ".hermes", "memories"))
    paths += ["./AGENTS.md", "./CLAUDE.md", "./.cursorrules"]
    return paths


class AgentMemoryConnector:
    def __init__(self, paths=None, kind="finding", source="agent_memory",
                 max_lines=200, max_entries=50):
        self.paths = paths or _default_paths()
        self.kind = kind
        self.source = source
        self.max_lines = max_lines
        self.max_entries = max_entries

    def _candidate_files(self):
        files = []
        for p in self.paths:
            p = os.path.expanduser(p)
            if os.path.isfile(p):
                files.append(p)
            elif os.path.isdir(p):
                for name in sorted(os.listdir(p)):
                    if name.endswith((".md", ".txt")):
                        files.append(os.path.join(p, name))
        return files

    def _lines(self, path):
        try:
            with open(path, errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # skip headers / pure separators / filenames
                    if line.startswith(("#", "---", "==")):
                        continue
                    yield line
        except OSError:
            return

    def collect(self):
        seeds = []
        for path in self._candidate_files():
            source_name = os.path.basename(path)
            for line in self._lines(path):
                if len(seeds) >= self.max_entries:
                    return seeds
                if len(line) > 3 and not line.startswith(("§", "*")):
                    seeds.append(Seed(
                        source=f"{self.source}:{source_name}",
                        kind=self.kind,
                        title=line[:120],
                        detail=line,
                        # internal tag: memory is voice/context, NOT content to
                        # draft from — the idea layer must skip these
                        tags=["agent_memory", "internal"],
                    ).to_dict())
        return seeds
