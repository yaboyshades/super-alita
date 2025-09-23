"""Lightweight observability primitives used by the research agent."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Timer:
    """Simple context manager measuring elapsed time."""

    name: str
    start: float | None = field(default=None, init=False)
    elapsed: float | None = field(default=None, init=False)

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        end = time.perf_counter()
        self.elapsed = end - (self.start or end)
        return False


def log_event(event: str, **kv: Any) -> None:
    """Emit a structured log entry (JSON) to stdout."""

    payload: dict[str, Any] = {
        "event": event,
        "ts": time.time(),
        **kv,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
