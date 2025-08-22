import uuid, json, time
from typing import Any, Dict

def new_id() -> str:
    return str(uuid.uuid4())

def emit_jsonl(event: Dict[str, Any], path="logs/events.jsonl"):
    event["timestamp"] = time.time()
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(event) + "\n")

class EventEmitter:
    def __init__(self, sink=emit_jsonl):
        self.sink = sink
    def emit(self, event: Dict[str, Any]):
        if "event_id" not in event:
            event["event_id"] = new_id()
        self.sink(event)

