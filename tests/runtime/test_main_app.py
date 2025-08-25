"""Smoke tests for the FastAPI app factory."""

import importlib

from fastapi.testclient import TestClient

import src.main as main
from tests.runtime import prefix_path


def test_create_app_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(tmp_path))
    app = main.create_app()
    client = TestClient(app)

    resp = client.get("/healthz")
    assert resp.status_code == 200

    resp = client.post(
        prefix_path("/v1/chat/stream"), json={"message": "hello", "session_id": "s1"}
    )
    assert resp.status_code == 200
    assert "<final_answer>" in resp.text

    # event bus writes to file
    assert isinstance(app.state.event_bus, main.FileEventBus)
    log = tmp_path / "events.jsonl"
    assert log.exists()
    assert log.read_text().strip()


def test_create_app_with_api_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(tmp_path))
    import reug_runtime.config as rc

    old_prefix = rc.SETTINGS.api_prefix
    rc.SETTINGS.api_prefix = "/api"
    try:
        app = main.create_app()
        client = TestClient(app)
        resp = client.post("/api/v1/chat/stream", json={"message": "hello", "session_id": "s1"})
        assert resp.status_code == 200
    finally:
        rc.SETTINGS.api_prefix = old_prefix
