"""SearxNG client with constitutional observability and resilience."""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error as ue
import urllib.parse as up
import urllib.request as ur
from typing import Any

from .egress_guard import is_allowed
from .metrics import Timer, log_event


class SearxClient:
    """Minimal HTTP client that honors CMA v5.3 resilience requirements."""

    def __init__(self, base_url: str | None = None, *, timeout: float = 10.0, attempts: int = 3) -> None:
        self.base_url = (base_url or os.environ.get("SEARXNG_BASE_URL") or "http://localhost:8080").rstrip("/")
        self.timeout = timeout
        self.attempts = attempts

    def search(self, query: str, *, categories: str = "science,web", max_results: int = 8) -> list[dict[str, Any]]:
        params = {"q": query, "format": "json", "categories": categories}
        url = f"{self.base_url}/search?{up.urlencode(params)}"
        allowed, reason = is_allowed(url)
        if not allowed:
            raise ValueError(f"egress blocked: {reason}")

        last_err: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            with Timer("research_query") as timer:
                try:
                    log_event(
                        "research.query.started",
                        attempt=attempt,
                        query=query,
                        url=url,
                    )
                    request = ur.Request(url, headers={"User-Agent": "SuperAlitaResearchAgent/0.1"})
                    with ur.urlopen(request, timeout=self.timeout) as resp:
                        payload = resp.read().decode("utf-8")
                    log_event(
                        "research.query.succeeded",
                        attempt=attempt,
                        elapsed=timer.elapsed,
                    )
                    data = json.loads(payload)
                    results = data.get("results", [])
                    if not isinstance(results, list):
                        raise ValueError("invalid response shape: results")
                    normalized = [
                        {
                            "title": str(item.get("title", "")),
                            "url": str(item.get("url", "")),
                            "snippet": str(item.get("content") or item.get("snippet", "")),
                        }
                        for item in results[:max_results]
                    ]
                    return normalized
                except (ue.URLError, ue.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                    last_err = exc
                    log_event(
                        "research.query.failed",
                        attempt=attempt,
                        error=str(exc),
                        elapsed=timer.elapsed,
                    )
                    if attempt >= self.attempts:
                        break
                    backoff = min(2 ** (attempt - 1), 8) + random.random()
                    time.sleep(backoff)
        assert last_err is not None  # pragma: no cover - defensive
        raise RuntimeError(f"query failed after {self.attempts} attempts") from last_err
