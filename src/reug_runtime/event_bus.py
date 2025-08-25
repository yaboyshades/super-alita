from __future__ import annotations

"""Async event bus implementations for the REUG runtime."""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class BaseEventBus(ABC):
    """Minimal interface for event bus implementations."""

    @abstractmethod
    async def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        """Emit an event and return the enriched payload."""


class FileEventBus(BaseEventBus):
    """Append events to a JSONL file asynchronously."""

    def __init__(self, log_dir: str | None):
        self.log_dir = Path(log_dir or "./logs/events")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.log_dir / "events.jsonl"

    async def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        event = {**event, "timestamp": time.time()}

        def _write() -> None:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            with self.file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        try:
            await asyncio.to_thread(_write)
        except Exception:
            logger.exception("failed to write event", extra={"event": event})
        return event


class RedisEventBus(BaseEventBus):
    """Publish events to a Redis channel asynchronously."""

    def __init__(self, url: str = "redis://localhost:6379/0", channel: str = "reug-events"):
        import redis  # type: ignore

        self._r = redis.Redis.from_url(url)
        self._ch = channel

    async def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        event = {**event, "timestamp": time.time()}
        try:
            await asyncio.to_thread(self._r.publish, self._ch, json.dumps(event))
        except Exception:
            logger.exception("failed to publish event", extra={"event": event})
        return event


def make_event_bus() -> BaseEventBus:
    """Factory selecting File or Redis bus based on environment."""

    backend = os.getenv("REUG_EVENTBUS", "").strip().lower()
    if backend == "redis":
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        channel = os.getenv("REUG_REDIS_CHANNEL", "reug-events")
        try:
            return RedisEventBus(url=url, channel=channel)
        except Exception as e:  # pragma: no cover
            logger.warning(
                "Redis event bus unavailable (%s); falling back to file", e,
                extra={"error": str(e)},
            )
    return FileEventBus(os.getenv("REUG_EVENT_LOG_DIR"))
