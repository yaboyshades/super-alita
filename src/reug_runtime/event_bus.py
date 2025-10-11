from __future__ import annotations

"""Async event bus implementations for the REUG runtime."""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class BaseEventBus(ABC):
    """Minimal interface for event bus implementations."""

    @abstractmethod
    async def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        """Emit an event and return the enriched payload."""

    # Optional pub/sub API used by plugin system. Default implementations are no-ops.
    async def subscribe(
        self, event_type: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:  # pragma: no cover - optional
        return None

    async def publish(
        self, event: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover - optional
        # Fallback to emit-only buses
        return await self.emit(event)


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


class InMemoryPubSubEventBus(FileEventBus):
    """In-memory pub/sub with JSONL logging via FileEventBus.

    - subscribe(event_type, handler): register an async handler
    - publish(event): dispatch to handlers and log via FileEventBus
    - emit(event): alias to publish for compatibility
    """

    def __init__(self, log_dir: str | None):
        super().__init__(log_dir)
        self._subs: dict[str, list[Callable[[dict[str, Any]], Awaitable[None]]]] = {}
        self._state_cache: dict[str, dict[str, Any]] = {}

    async def subscribe(
        self, event_type: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        self._subs.setdefault(event_type, []).append(handler)

    async def publish(self, event: dict[str, Any]) -> dict[str, Any]:
        # Log to file first
        await super().emit(event)
        # Dispatch to any subscribers (best-effort, non-blocking)
        handlers = list(self._subs.get(event.get("event_type", ""), []))
        for h in handlers:
            try:
                # Schedule without awaiting to avoid blocking
                asyncio.create_task(h(event))
            except Exception:
                logger.exception("failed to dispatch event", extra={"event": event})
        self._update_agent_state_cache(event)
        return event

    async def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        return await self.publish(event)

    # ---- Agent state cache helpers -------------------------------------------------

    def _update_agent_state_cache(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind", "")).lower()
        event_type = str(event.get("event_type", "")).lower()
        if kind != "agent_state" and event_type != "agentstateevent":
            return

        change = event.get("change")
        if not isinstance(change, dict):
            return

        agent_id = change.get("agent_id") or event.get("agent_id")
        state_key = change.get("state_key")
        if not agent_id or not state_key:
            return

        entry = self._state_cache.setdefault(str(agent_id), {})
        entry[str(state_key)] = {
            "value": change.get("value"),
            "previous_value": change.get("previous_value"),
            "metadata": deepcopy(change.get("metadata", {})),
            "change_id": change.get("change_id"),
            "timestamp": event.get("timestamp"),
        }

    def get_agent_state(self, agent_id: str) -> dict[str, Any]:
        """Return a copy of the cached state for the provided agent."""

        return deepcopy(self._state_cache.get(str(agent_id), {}))

    def snapshot_agent_states(self) -> dict[str, dict[str, Any]]:
        """Return a copy of the entire cached agent state registry."""

        return deepcopy(self._state_cache)


class EventMetricsCollector:
    """Track success/failure counts for event publication health."""

    def __init__(self) -> None:
        self._success = 0
        self._failure = 0
        self._dropped = 0
        self._lock = Lock()

    def record_publish_success(self) -> None:
        with self._lock:
            self._success += 1

    def record_publish_failure(self) -> None:
        with self._lock:
            self._failure += 1

    def record_publish_dropped(self) -> None:
        with self._lock:
            self._dropped += 1

    @property
    def error_rate(self) -> float:
        with self._lock:
            total = self._success + self._failure
            if total == 0:
                return 0.0
            return self._failure / total

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            total = self._success + self._failure
            error_rate = self._failure / total if total else 0.0
            return {
                "success": self._success,
                "failure": self._failure,
                "dropped": self._dropped,
                "error_rate": error_rate,
            }


class ProductionEventBus(InMemoryPubSubEventBus):
    """Resilient event bus with priority queues and circuit breaker logic."""

    def __init__(
        self,
        log_dir: str | None,
        queue_max_sizes: Mapping[str, int] | None = None,
        degraded_error_rate: float = 0.1,
        fallback_bus: BaseEventBus | None = None,
    ) -> None:
        super().__init__(log_dir)
        self.metrics = EventMetricsCollector()
        max_sizes = queue_max_sizes or {"high": 1000, "medium": 5000, "low": 10000}
        self.queues: dict[str, asyncio.Queue[dict[str, Any]]] = {
            key: asyncio.Queue(maxsize=value) for key, value in max_sizes.items()
        }
        self.dead_letter_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._workers: dict[str, asyncio.Task[None] | None] = {
            key: None for key in self.queues
        }
        self._fallback_bus = fallback_bus or FileEventBus(log_dir)
        self._degraded_error_rate = degraded_error_rate

    async def publish(
        self, event: dict[str, Any], priority: str | None = None
    ) -> dict[str, Any]:
        resolved_priority = self._resolve_priority(priority, event)
        if self.metrics.error_rate > self._degraded_error_rate:
            return await self.degraded_publish(event)
        queue = self.queues.get(resolved_priority)
        if queue is None:
            queue = self.queues["medium"]
            resolved_priority = "medium"
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            await self.handle_backpressure(event)
            self.metrics.record_publish_dropped()
            return event
        self._ensure_worker(resolved_priority)
        return event

    async def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        return await self.publish(event)

    async def degraded_publish(self, event: dict[str, Any]) -> dict[str, Any]:
        """Fallback path when error rate exceeds threshold."""

        try:
            await self._fallback_bus.emit(event)
            self.metrics.record_publish_success()
        except Exception:
            self.metrics.record_publish_failure()
            logger.exception("degraded publish failed", extra={"event": event})
        return event

    async def handle_backpressure(self, event: dict[str, Any]) -> None:
        """Handle queue saturation with importance-aware logic."""

        importance = float(event.get("importance", 0.0) or 0.0)
        if importance > 0.8:
            await self.force_process_important(event)
            return
        logger.warning("dropping event due to backpressure", extra={"event": event})
        await self.dead_letter_queue.put({**event, "dead_letter_reason": "backpressure"})

    async def force_process_important(self, event: dict[str, Any]) -> None:
        await self._process_event(event)

    def _ensure_worker(self, priority: str) -> None:
        worker = self._workers.get(priority)
        if worker is None or worker.done():
            self._workers[priority] = asyncio.create_task(self._drain_queue(priority))

    async def _drain_queue(self, priority: str) -> None:
        queue = self.queues[priority]
        while True:
            try:
                event = queue.get_nowait()
            except asyncio.QueueEmpty:
                self._workers[priority] = None
                break
            try:
                await self._process_event(event)
            finally:
                queue.task_done()

    async def _process_event(self, event: dict[str, Any]) -> None:
        try:
            await super().publish(event)
        except Exception:
            self.metrics.record_publish_failure()
            logger.exception("failed to publish event", extra={"event": event})
            await self.dead_letter_queue.put({**event, "dead_letter_reason": "failure"})
        else:
            self.metrics.record_publish_success()

    def _resolve_priority(self, priority: str | None, event: dict[str, Any]) -> str:
        requested = priority or str(event.get("priority", "medium")).lower()
        if requested.endswith("_priority"):
            requested = requested.rsplit("_priority", 1)[0]
        if requested not in self.queues:
            return "medium"
        return requested

class RedisEventBus(BaseEventBus):
    """Publish events to a Redis channel asynchronously."""

    def __init__(
        self, url: str = "redis://localhost:6379/0", channel: str = "reug-events"
    ):
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
                "Redis event bus unavailable (%s); falling back to file",
                e,
                extra={"error": str(e)},
            )
    if backend in {"production", "resilient"}:
        return ProductionEventBus(os.getenv("REUG_EVENT_LOG_DIR"))
    # Default to in-memory pub/sub with file logging to support plugins
    return ProductionEventBus(os.getenv("REUG_EVENT_LOG_DIR"))


# ---- Typed helper emitters (optional) --------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def evt_task_started(
    correlation_id: str, goal: str, session_id: str | None = None
) -> dict[str, Any]:
    return {
        "type": "TaskStarted",
        "correlation_id": correlation_id,
        "goal": goal,
        "session_id": session_id,
        "ts": _now_iso(),
    }


def evt_llm_chunk(
    correlation_id: str, text: str, session_id: str | None = None
) -> dict[str, Any]:
    return {
        "type": "LLMChunk",
        "correlation_id": correlation_id,
        "data": {"text": text},
        "session_id": session_id,
        "ts": _now_iso(),
    }


def evt_ability_called(
    correlation_id: str,
    span_id: str,
    tool: str,
    args: Any | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "AbilityCalled",
        "correlation_id": correlation_id,
        "span_id": span_id,
        "tool": tool,
        "session_id": session_id,
        "ts": _now_iso(),
    }
    if args is not None:
        payload["args"] = args
    return payload


def evt_ability_succeeded(
    correlation_id: str,
    span_id: str,
    tool: str,
    result: Any,
    session_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "AbilitySucceeded",
        "correlation_id": correlation_id,
        "span_id": span_id,
        "tool": tool,
        "result": result,
        "session_id": session_id,
        "ts": _now_iso(),
    }


def evt_ability_failed(
    correlation_id: str,
    span_id: str,
    tool: str,
    error: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "AbilityFailed",
        "correlation_id": correlation_id,
        "span_id": span_id,
        "tool": tool,
        "error": error,
        "session_id": session_id,
        "ts": _now_iso(),
    }


def evt_task_succeeded(
    correlation_id: str, data: dict[str, Any], session_id: str | None = None
) -> dict[str, Any]:
    return {
        "type": "TaskSucceeded",
        "correlation_id": correlation_id,
        "data": data,
        "session_id": session_id,
        "ts": _now_iso(),
    }


__all__ = [
    "BaseEventBus",
    "FileEventBus",
    "InMemoryPubSubEventBus",
    "RedisEventBus",
    "EventMetricsCollector",
    "ProductionEventBus",
    "make_event_bus",
    # helpers
    "evt_task_started",
    "evt_llm_chunk",
    "evt_ability_called",
    "evt_ability_succeeded",
    "evt_ability_failed",
    "evt_task_succeeded",
]
