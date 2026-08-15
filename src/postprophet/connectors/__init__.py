"""Reference connectors. Each implements collect() -> list[seed dicts].

These are the portable "connect to your agent stack" adapters. Builders use the
shipped ones or write a thin adapter of their own; the core never depends on any
specific connector.
"""

from .git import GitConnector
from .agent_output import AgentOutputConnector
from .webhook import WebhookConnector
from .github import GitHubConnector
from .rss import RssConnector
from .agent_memory import AgentMemoryConnector

__all__ = [
    "GitConnector", "AgentOutputConnector", "WebhookConnector",
    "GitHubConnector", "RssConnector", "AgentMemoryConnector",
]


def build_connector(cfg):
    """Instantiate a connector from a ConnectorConfig. Returns None if the type is
    unknown (so a partial config still runs). Unknown = warning, not crash."""
    if not cfg.enabled:
        return None
    try:
        if cfg.type == "git":
            return GitConnector(**cfg.options)
        if cfg.type == "agent_output":
            return AgentOutputConnector(**cfg.options)
        if cfg.type == "webhook":
            return WebhookConnector(**cfg.options)
        if cfg.type == "github":
            return GitHubConnector(**cfg.options)
        if cfg.type == "rss":
            return RssConnector(**cfg.options)
        if cfg.type == "agent_memory":
            return AgentMemoryConnector(**cfg.options)
    except TypeError:
        pass
    return None
