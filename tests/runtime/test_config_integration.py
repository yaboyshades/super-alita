"""Integration tests ensuring environment-driven configuration is applied."""

from fastapi.testclient import TestClient

from src.main import FileEventBus, create_app


def test_config_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REUG_EVENTBUS", "file")
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("REUG_TOOL_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.setenv("REUG_REGISTRY", "simple")
    monkeypatch.setenv("REUG_KG", "simple")
    monkeypatch.setenv("REUG_LLM_PROVIDER", "mock")
    monkeypatch.setenv("REUG_MAX_TOOL_CALLS", "7")

    app = create_app()
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        settings = app.state.settings
        assert settings.max_tool_calls == 7
        assert settings.event_bus_backend == "file"
        assert settings.event_log_dir == str(tmp_path / "logs")
        assert settings.tool_registry_dir == str(tmp_path / "registry")
        assert isinstance(app.state.event_bus, FileEventBus)
        assert app.state.llm_model._provider == "mock"
