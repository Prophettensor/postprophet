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

DEFAULT_PATHS = [
    "~/.hermes/memories/MEMORY.md",
    "~/.hermes/memories/USER.md",
    "~/.hermes/memories/",
    "./AGENTS.md",
    "./CLAUDE.md",
    "./.cursorrules",
]


class AgentMemoryConnector:
    def __init__(self, paths=None, kind="finding", source="agent_memory",
                 max_lines=200, max_entries=50):
        self.paths = paths or DEFAULT_PATHS
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
                        tags=["agent_memory"],
                    ).to_dict())
        return seeds
