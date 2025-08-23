import hashlib
import json
import logging
import os
import time
import uuid
from functools import lru_cache
from typing import Any, Dict

logger = logging.getLogger(__name__)


def new_id() -> str:
    return str(uuid.uuid4())


def new_span_id() -> str:
    return new_id()


def hash_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf8")).hexdigest()


def emit_jsonl(event: Dict[str, Any], path: str | None = None) -> None:
    """Persist event to local JSONL file."""
    if path is None:
        base = os.getenv("REUG_EVENT_LOG_DIR", "logs")
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, "events.jsonl")
    event["timestamp"] = time.time()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


@lru_cache(maxsize=1)
def _get_redis_client(url: str):
    try:
        import redis  # type: ignore
    except ImportError:  # pragma: no cover - redis optional
        return None
    return redis.Redis.from_url(url)


def emit_redis(event: Dict[str, Any]) -> None:
    """Publish event to Redis channel with fallback."""
    client = _get_redis_client(os.getenv("REUG_REDIS_URL", "redis://localhost:6379/0"))
    if client is None:
        emit_jsonl(event)
        return

    channel = os.getenv("REUG_EVENTBUS_CHANNEL", "reug.events")
    event["timestamp"] = time.time()
    try:
        client.publish(channel, json.dumps(event))
    except Exception:  # pragma: no cover - best effort
        logger.exception("Failed to publish event to Redis; falling back to JSONL")
        emit_jsonl(event)


class EventEmitter:
    """Flexible event emitter with pluggable sinks."""

    def __init__(self, sink=None):
        if sink is None:
            mode = os.getenv("REUG_EVENTBUS", "file").lower()
            sink = emit_redis if mode == "redis" else emit_jsonl

        if isinstance(sink, str):
            if sink == "redis":
                self.sink = emit_redis
            else:
                path = sink
                self.sink = lambda e, p=path: emit_jsonl(e, p)
        else:
            self.sink = sink

    def emit(self, event: Dict[str, Any] | None = None, **kwargs) -> Dict[str, Any]:
        if event is None:
            event = kwargs
        else:
            event.update(kwargs)
        if "event_id" not in event:
            event["event_id"] = new_id()
        self.sink(event)
        return event
