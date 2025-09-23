#!/usr/bin/env python3
"""Simple mock SearxNG server for testing."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


class MockHandler(BaseHTTPRequestHandler):
    """Handle SearxNG-like search requests."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path.startswith("/search"):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]

            # Mock response with relevant results
            results = [
                {
                    "title": "Emergent Behaviors in Multi-Agent RAG Systems",
                    "url": "https://example.com/research/1",
                    "content": "Analysis of evaluation metrics and governance patterns...",
                },
                {
                    "title": "Governance Patterns for RAG Systems",
                    "url": "https://example.com/research/2",
                    "content": "Best practices for multi-agent evaluation...",
                }
            ]

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"results": results}).encode())
            return

        self.send_response(404)
        self.end_headers()


def run(port=8080):
    """Run the mock server."""
    server = HTTPServer(("localhost", port), MockHandler)
    print(f"Mock SearxNG server running on port {port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
