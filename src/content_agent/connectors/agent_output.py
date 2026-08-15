"""Agent-output connector — emits seeds from any agent that writes files.

This is the "connect to your agent stack" connector for non-git agents: research
agents, customer-success logs, ecosystem monitors, etc. Any agent that writes
structured or text output to a directory can feed the content agent by having its
output land here.

Options:
    path      directory to watch (or single file)
    exts      file extensions to read (default: .json, .md, .txt, .log)
    kind      what these outputs represent: finding | signal | event (default finding)
    source    source label used in seeds (default "agent_output")

Seeds are built from file content:
    - .json files: uses keys "title", "detail", "date", "tags" if present,
      else the first non-empty string value as title.
    - text files: first non-empty line = title, remainder (first 300 chars) = detail.

Zero LLM. Deterministic (sorted by mtime, newest first).
"""

from __future__ import annotations

import json
import os

from ..context import Seed

DEFAULT_EXTS = (".json", ".md", ".txt", ".log")


class AgentOutputConnector:
    def __init__(self, path, exts=None, kind="finding", source="agent_output",
                 limit=50):
        self.path = os.path.expanduser(path)
        self.exts = tuple(exts) if exts else DEFAULT_EXTS
        self.kind = kind
        self.source = source
        self.limit = limit

    def _targets(self):
        if os.path.isfile(self.path):
            return [self.path]
        if os.path.isdir(self.path):
            files = []
            for root, _, names in os.walk(self.path):
                for n in names:
                    if n.endswith(self.exts):
                        files.append(os.path.join(root, n))
            # newest first
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return files
        return []

    def _read(self, path):
        try:
            with open(path, errors="ignore") as f:
                return f.read(2000)
        except OSError:
            return ""

    def _parse(self, path):
        content = self._read(path)
        if not content.strip():
            return None
        name = os.path.basename(path)
        if path.endswith(".json"):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    title = str(data.get("title") or data.get("name") or name)
                    detail = str(data.get("detail") or data.get("summary") or "")
                    date = data.get("date")
                    tags = list(data.get("tags", []))
                    return {"title": title, "detail": detail, "date": date, "tags": tags}
            except json.JSONDecodeError:
                pass
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        title = lines[0] if lines else name
        detail = " ".join(lines[1:])[:300]
        return {"title": title, "detail": detail, "date": None, "tags": []}

    def collect(self):
        seeds = []
        for path in self._targets()[:self.limit]:
            parsed = self._parse(path)
            if not parsed:
                continue
            seeds.append(Seed(
                source=self.source,
                kind=self.kind,
                title=f"[{self.source}] {parsed['title']}",
                detail=parsed["detail"],
                date=parsed.get("date"),
                tags=parsed.get("tags", []),
            ).to_dict())
        return seeds
