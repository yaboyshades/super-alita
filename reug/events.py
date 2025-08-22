import uuid, json, time, os, hashlib
from typing import Any, Dict


def new_id() -> str:
    return str(uuid.uuid4())


def new_span_id() -> str:
    return new_id()


def hash_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf8")).hexdigest()


def emit_jsonl(event: Dict[str, Any], path: str | None = None):
    if path is None:
        base = os.getenv("REUG_EVENT_LOG_DIR", "logs")
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, "events.jsonl")
    event["timestamp"] = time.time()
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(event) + "\n")


class EventEmitter:
    def __init__(self, sink=emit_jsonl):
        if isinstance(sink, str):
            path = sink
            self.sink = lambda e: emit_jsonl(e, path)
        else:
            self.sink = sink

    def emit(self, event: Dict[str, Any] | None = None, **kwargs):
        if event is None:
            event = kwargs
        else:
            event.update(kwargs)
        if "event_id" not in event:
            event["event_id"] = new_id()
        self.sink(event)
