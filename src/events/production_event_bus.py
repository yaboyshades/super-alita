from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

import redis.asyncio as redis


@dataclass(slots=True)
class Event:
    """Structured event payload persisted on the bus."""

    id: str
    type: str
    correlation_id: str
    timestamp: str
    source: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]


class DeadLetterQueue:
    """Captures events that exhausted retry attempts."""

    def __init__(self, redis_client: redis.Redis) -> None:
        self.redis = redis_client
        self.dlq_stream = "event_dlq"

    async def add_failed_event(self, event: Dict[str, Any], error: str, retry_count: int) -> None:
        payload = {
            "original_event": event,
            "error": error,
            "retry_count": retry_count,
            "failed_at": datetime.now(UTC).isoformat(),
            "dlq_id": str(uuid.uuid4()),
        }
        await self.redis.xadd(self.dlq_stream, {"data": json.dumps(payload)}, maxlen=10_000)

    async def get_failed_events(self, count: int = 100) -> List[Dict[str, Any]]:
        entries = await self.redis.xrevrange(self.dlq_stream, count=count)
        failed: List[Dict[str, Any]] = []
        for stream_id, raw in entries:
            payload = json.loads(raw[b"data"])
            payload["stream_id"] = stream_id
            failed.append(payload)
        return failed


class EventMetrics:
    """Tracks lifecycle metrics for emitted events."""

    def __init__(self) -> None:
        self.emitted_count = 0
        self.processed_count = 0
        self.failed_count = 0
        self.dlq_count = 0

    def record_emission(self) -> None:
        self.emitted_count += 1

    def record_processing(self, success: bool) -> None:
        self.processed_count += 1
        if not success:
            self.failed_count += 1

    def record_dlq(self) -> None:
        self.dlq_count += 1

    def snapshot(self) -> Dict[str, Any]:
        success_rate = 0.0
        if self.emitted_count:
            success_rate = self.processed_count / self.emitted_count
        return {
            "emitted_count": self.emitted_count,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "dlq_count": self.dlq_count,
            "success_rate": success_rate,
        }


class ProductionEventBus:
    """Redis Streams backed event bus with retry and replay support."""

    def __init__(
        self,
        redis_url: str,
        *,
        backup_file: Optional[str] = None,
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.redis_url = redis_url
        self.redis = redis_client or redis.from_url(redis_url)
        self.backup_file = backup_file
        self.dead_letter_queue = DeadLetterQueue(self.redis)
        self.metrics = EventMetrics()
        self.subscribers: Dict[str, List[tuple[str, Callable[[Dict[str, Any]], Awaitable[None]]]]] = {}
        self.event_schemas = self._load_event_schemas()

    def _load_event_schemas(self) -> Dict[str, Dict[str, Any]]:
        return {
            "user.action": {"required_fields": ["user_id", "action_type", "timestamp"]},
            "system.alert": {"required_fields": ["alert_type", "severity", "message"]},
            "llm.request": {"required_fields": ["prompt", "provider", "context"]},
            "constitutional.evaluation": {"required_fields": ["action", "result", "score"]},
        }

    async def emit(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        validation = self._validate_event_schema(event_data)
        if not validation["valid"]:
            raise ValueError(f"Event validation failed: {validation['errors']}")

        event_id = event_data.get("id", f"event_{uuid.uuid4().hex}")
        correlation_id = event_data.get("correlation_id", str(uuid.uuid4()))
        event = Event(
            id=event_id,
            type=event_data["type"],
            correlation_id=correlation_id,
            timestamp=event_data.get("timestamp", datetime.now(UTC).isoformat()),
            source=event_data.get("source", "super_alita"),
            data=event_data.get("data", {}),
            metadata=event_data.get(
                "metadata",
                {
                    "schema_version": "1.0",
                    "retry_count": 0,
                    "priority": event_data.get("priority", "normal"),
                },
            ),
        )

        stream_id = await self.redis.xadd("events", {"event": json.dumps(asdict(event))}, maxlen=100_000)
        if self.backup_file:
            await self._backup_event(event)

        self.metrics.record_emission()
        self.logger.info("Emitted event %s (%s)", event.type, event.id)
        return {
            "event_id": event.id,
            "stream_id": stream_id.decode() if isinstance(stream_id, bytes) else stream_id,
            "correlation_id": correlation_id,
            "timestamp": event.timestamp,
        }

    def _validate_event_schema(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        schema = self.event_schemas.get(event_data.get("type"))
        if not schema:
            return {"valid": True, "errors": []}
        errors: List[str] = []
        for field in schema["required_fields"]:
            if field not in event_data:
                errors.append(f"Missing required field: {field}")
        if "data" in event_data and not isinstance(event_data["data"], dict):
            errors.append("Data field must be a dictionary")
        return {"valid": not errors, "errors": errors}

    async def _backup_event(self, event: Event) -> None:
        try:
            entry = {"timestamp": datetime.now(UTC).isoformat(), "event": asdict(event)}
            with open(self.backup_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Unable to persist event backup: %s", exc)

    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
        consumer_group: str = "default",
    ) -> None:
        try:
            await self.redis.xgroup_create("events", consumer_group, id="0", mkstream=True)
        except Exception as exc:  # pragma: no cover - defensive
            if "BUSYGROUP" not in str(exc):
                raise
        self.subscribers.setdefault(consumer_group, []).append((pattern, handler))
        self.logger.info("Subscribed handler to %s in group %s", pattern, consumer_group)

    async def start_processing(
        self,
        consumer_group: str = "default",
        *,
        consumer_name: Optional[str] = None,
        batch_size: int = 10,
    ) -> None:
        consumer = consumer_name or f"consumer_{uuid.uuid4().hex[:8]}"
        self.logger.info("Starting event processing for %s in group %s", consumer, consumer_group)
        while True:
            try:
                results = await self.redis.xreadgroup(
                    groupname=consumer_group,
                    consumername=consumer,
                    streams={"events": ">"},
                    count=batch_size,
                    block=5000,
                )
                if not results:
                    continue
                for _, messages in results:
                    for message_id, message_data in messages:
                        event_dict = json.loads(message_data[b"event"])
                        handlers = self._find_matching_handlers(event_dict["type"], consumer_group)
                        for handler in handlers:
                            success = await self._process_event_with_retry(
                                handler,
                                event_dict,
                                message_id,
                                consumer_group,
                                consumer,
                            )
                            if success:
                                await self.redis.xack("events", consumer_group, message_id)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error("Event processing error: %s", exc)
                await asyncio.sleep(1)

    def _find_matching_handlers(
        self,
        event_type: str,
        consumer_group: str,
    ) -> List[Callable[[Dict[str, Any]], Awaitable[None]]]:
        handlers: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
        for pattern, handler in self.subscribers.get(consumer_group, []):
            if self._pattern_matches(event_type, pattern):
                handlers.append(handler)
        return handlers

    def _pattern_matches(self, event_type: str, pattern: str) -> bool:
        if pattern == event_type:
            return True
        if pattern.endswith(".*"):
            return event_type.startswith(pattern[:-2])
        if pattern.startswith("*."):
            return event_type.endswith(pattern[2:])
        if "*" in pattern:
            from fnmatch import fnmatch

            return fnmatch(event_type, pattern)
        return False

    async def _process_event_with_retry(
        self,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
        event: Dict[str, Any],
        message_id: str,
        consumer_group: str,
        consumer_name: str,
        *,
        max_retries: int = 3,
    ) -> bool:
        attempt = 0
        while attempt <= max_retries:
            try:
                await handler(event)
                self.metrics.record_processing(True)
                return True
            except Exception as exc:  # pragma: no cover - handler failures expected
                attempt += 1
                self.logger.warning(
                    "Handler failed for %s (attempt %s/%s): %s",
                    event.get("id"),
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt <= max_retries:
                    wait_time = min((2**attempt) + random.uniform(0, 0.1), 30)
                    await asyncio.sleep(wait_time)
                    continue
                self.metrics.record_processing(False)
                self.metrics.record_dlq()
                await self.dead_letter_queue.add_failed_event(event, str(exc), attempt)
                return False
        return False

    async def replay(
        self,
        from_timestamp: str,
        *,
        to_timestamp: Optional[str] = None,
        event_filter: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        from_id = await self._timestamp_to_stream_id(from_timestamp)
        to_id = await self._timestamp_to_stream_id(to_timestamp) if to_timestamp else "+"
        results = await self.redis.xrange("events", min=from_id, max=to_id)
        for _, data in results:
            event_dict = json.loads(data[b"event"])
            if event_filter and not self._event_matches_filter(event_dict, event_filter):
                continue
            yield event_dict

    async def _timestamp_to_stream_id(self, timestamp: Optional[str]) -> str:
        if not timestamp:
            return "0-0"
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            millis = int(dt.timestamp() * 1000)
            return f"{millis}-0"
        except Exception:  # pragma: no cover - fallback
            return "0-0"

    def _event_matches_filter(self, event: Dict[str, Any], event_filter: Dict[str, Any]) -> bool:
        for key, value in event_filter.items():
            if key == "type" and event.get("type") != value:
                return False
            if key == "source" and event.get("source") != value:
                return False
            if key == "correlation_id" and event.get("correlation_id") != value:
                return False
        return True

    async def get_metrics(self) -> Dict[str, Any]:
        metrics = self.metrics.snapshot()
        metrics.update(
            {
                "events_in_stream": await self.redis.xlen("events"),
                "events_in_dlq": await self.redis.xlen("event_dlq"),
                "active_subscriptions": sum(len(v) for v in self.subscribers.values()),
            }
        )
        return metrics


async def create_event_bus(redis_url: str, backup_file: Optional[str] = None) -> ProductionEventBus:
    """Factory helper for parity with runtime initialisers."""

    return ProductionEventBus(redis_url, backup_file=backup_file)
