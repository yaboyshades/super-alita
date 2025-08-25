"""Smoke tests for the FastAPI app factory."""

from fastapi.testclient import TestClient

from src.main import FileEventBus, create_app


def test_create_app_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(tmp_path))
    app = create_app()
    client = TestClient(app)

    resp = client.get("/healthz")
    assert resp.status_code == 200

    resp = client.post("/v1/chat/stream", json={"message": "hello", "session_id": "s1"})
    assert resp.status_code == 200
    assert "<final_answer>" in resp.text

    # event bus writes to file
    assert isinstance(app.state.event_bus, FileEventBus)
    log = tmp_path / "events.jsonl"
    assert log.exists()
    assert log.read_text().strip()
