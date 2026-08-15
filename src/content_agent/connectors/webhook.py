"""Webhook connector — emits seeds from an inbound HTTP POST (seed-shaped JSON).

For builders who want a live "inject an idea" path: any agent in their stack can
POST a seed-shaped payload to a local endpoint and it lands in the content
pipeline. Uses the stdlib http.server so no framework dependency.

Options:
    host   bind host (default 127.0.0.1)
    port   bind port (default 8737)
    path   URL path to listen on (default "/seed")

NOTE on infrastructure: inbound ports are only reachable from inside the
container / builder's own machine. This is for local stack wiring. For remote
agents, the builder runs this behind their own ingress. This connector is
intentionally a thin server; builders with a message bus (Kafka, NATS, etc.)
should write their own collector instead.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..context import validate_seed


class _Handler(BaseHTTPRequestHandler):
    server: "WebhookConnector"  # set on the server instance

    def _ok(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if self.path != self.server.connector.path:
            self.send_response(404); self.end_headers(); return
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body or b"{}")
            seed = validate_seed(payload)  # raises if malformed
            self.server.connector.inbox.append(seed.to_dict())
            self._ok()
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def log_message(self, *a):
        pass  # quiet


class WebhookConnector:
    def __init__(self, host="127.0.0.1", port=8737, path="/seed", source="webhook"):
        self.host = host
        self.port = port
        self.path = path
        self.source = source
        self.inbox = []
        self._server = None
        self._thread = None

    def start(self):
        """Start the listener in a daemon thread. Idempotent."""
        if self._server:
            return
        handler = type("BoundHandler", (_Handler,), {})
        self._server = HTTPServer((self.host, self.port), handler)
        self._server.connector = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None

    def collect(self):
        """Return any seeds received since the last collect and drain the inbox."""
        out, self.inbox = self.inbox, []
        return out


if __name__ == "__main__":
    import time
    w = WebhookConnector()
    w.start()
    print(f"listening on {w.host}:{w.port}{w.path}  (ctrl-c to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        w.stop()
