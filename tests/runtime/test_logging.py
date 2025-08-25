import asyncio
import json
import logging

from src.main import create_app


def test_logging_and_startup_events(tmp_path, monkeypatch) -> None:
    log_dir = tmp_path / "logs"
    event_dir = tmp_path / "events"
    monkeypatch.setenv("REUG_LOG_DIR", str(log_dir))
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(event_dir))
    app = create_app()
    asyncio.run(app.router.startup())
    logging.shutdown()

    log_file = log_dir / "runtime.log"
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert lines
    record = json.loads(lines[0])
    assert record["message"] == "runtime startup"

    events_file = event_dir / "events.jsonl"
    assert events_file.exists()
    events = [json.loads(l) for l in events_file.read_text().splitlines()]
    kinds = {e["type"] for e in events}
    assert {"STATE_TRANSITION", "TaskStarted"} <= kinds
