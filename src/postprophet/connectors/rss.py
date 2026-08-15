"""RSS/feed connector — emits 'signal'/'event' seeds from feeds and market
sources. This is the "understands the market you're within" input: competitor
blogs, ecosystem announcements, news feeds.

Options:
    feeds       list of feed URLs (required)
    kind        what these represent: signal | event (default signal)
    since_hours only emit items newer than this (default 48)

Zero LLM. Uses stdlib xml.etree to parse RSS 2.0 / Atom items. Deterministic
(newest first). Non-blocking: a single failed feed is skipped, not fatal.
"""

from __future__ import annotations

import email.utils
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from ..context import Seed

_ATOM = "{http://www.w3.org/2005/Atom}"


def _parse_date(value):
    if not value:
        return None
    try:
        # RFC 822 (RSS pubDate)
        return email.utils.parsedate_to_datetime(value).astimezone(timezone.utc)
    except Exception:
        pass
    try:
        # ISO 8601 (Atom updated/published)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


class RssConnector:
    def __init__(self, feeds, kind="signal", since_hours: int = 48, limit_per_feed: int = 5):
        self.feeds = feeds
        self.kind = kind
        self.since_hours = since_hours
        self.limit_per_feed = limit_per_feed

    def _fetch(self, url):
        req = urllib.request.Request(url, headers={"User-Agent": "postprophet/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read(2_000_000)  # cap at ~2MB
        except Exception:
            return None

    def _items(self, url, data):
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            return []
        items = []
        # Atom
        for entry in root.findall(f"{_ATOM}entry"):
            title = entry.findtext(f"{_ATOM}title") or ""
            link_el = entry.find(f"{_ATOM}link")
            link = (link_el.get("href") or "") if link_el is not None else ""
            date = _parse_date(entry.findtext(f"{_ATOM}updated")
                               or entry.findtext(f"{_ATOM}published"))
            items.append({"title": title, "link": link, "date": date,
                          "summary": entry.findtext(f"{_ATOM}summary") or ""})
        # RSS 2.0
        for item in root.iter("item"):
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            date = _parse_date(item.findtext("pubDate"))
            items.append({"title": title, "link": link, "date": date,
                          "summary": item.findtext("description") or ""})
        return items

    def collect(self):
        cutoff = datetime.now(timezone.utc).timestamp() - self.since_hours * 3600
        seeds = []
        for feed in self.feeds:
            data = self._fetch(feed)
            if data is None:
                continue
            for it in self._items(feed, data):
                ts = it["date"].timestamp() if it["date"] else 0
                if ts < cutoff:
                    continue
                seeds.append(Seed(
                    source=f"rss:{feed.split('/')[2]}",
                    kind=self.kind,
                    title=it["title"],
                    detail=f"{it.get('summary','')[:200]} {it.get('link','')}".strip(),
                    date=datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else None,
                    tags=["feed"],
                ).to_dict())
            if len(seeds) >= self.limit_per_feed * len(self.feeds):
                break
        return seeds[: self.limit_per_feed * len(self.feeds)]
