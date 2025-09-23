"""Integration tests for the research pipeline."""
from __future__ import annotations

import contextlib
import http.server
import json
import socket
import threading
from pathlib import Path

from research_agent.pipeline import ResearchPipeline


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if "/search" not in self.path:
            self.send_response(404)
            self.end_headers()
            return
        payload = {
            "results": [
                {
                    "title": "Paper A",
                    "url": "https://example.com/a",
                    "content": "A summary",
                    "snippet": "A summary",
                },
                {
                    "title": "Paper B",
                    "url": "https://example.com/b",
                    "content": "B summary",
                    "snippet": "B summary",
                },
            ]
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs):  # type: ignore[override]
        return


@contextlib.contextmanager
def _searx_server() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        addr, port = sock.getsockname()
    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join()


def test_pipeline_runs(tmp_path: Path, monkeypatch) -> None:
    with _searx_server() as base_url:
        monkeypatch.setenv("RESEARCH_AGENT_ALLOW_LOOPBACK", "1")
        monkeypatch.setenv("SEARXNG_BASE_URL", base_url)
        pipeline = ResearchPipeline()
        result = pipeline.run("transformer safety")
        assert result.stats["count"] == 2
        md_path = tmp_path / "research.md"
        md_path.write_text(result.to_markdown(), encoding="utf-8")
        written = md_path.read_text(encoding="utf-8")
        assert "Paper A" in written
        assert "https://example.com/b" in written
