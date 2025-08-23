import json

from reug.events import EventEmitter


def test_emit_redis_falls_back_to_jsonl(monkeypatch, tmp_path):
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(tmp_path))
    monkeypatch.setattr("reug.events._get_redis_client", lambda url: None)

    emitter = EventEmitter(sink="redis")
    event = emitter.emit(event_type="PING")

    path = tmp_path / "events.jsonl"
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event_type"] == "PING"
    assert event["event_id"] == data["event_id"]
