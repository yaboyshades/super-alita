from fastapi.testclient import TestClient

from src.main import (
    FileEventBus,
    LLMClient,
    SimpleAbilityRegistry,
    SimpleKG,
    create_app,
)


def test_app_factory_configures_app(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv(
        "CORS_ALLOW_ORIGINS", "https://foo.com,https://bar.com"
    )

    app = create_app()

    # dependencies exist on app.state
    assert isinstance(app.state.event_bus, FileEventBus)
    assert isinstance(app.state.ability_registry, SimpleAbilityRegistry)
    assert isinstance(app.state.kg, SimpleKG)
    assert isinstance(app.state.llm_model, LLMClient)

    # routers registered
    routes = {r.path for r in app.routes}
    assert "/v1/chat/stream" in routes
    assert "/tools/fs_read" in routes

    # CORS config respects environment
    cors = next(
        (mw for mw in app.user_middleware if mw.cls.__name__ == "CORSMiddleware"),
        None,
    )
    assert cors is not None
    assert cors.kwargs["allow_origins"] == [
        "https://foo.com",
        "https://bar.com",
    ]

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "service": "super-alita"}
    resp = client.get("/healthz")
    assert resp.status_code == 200
