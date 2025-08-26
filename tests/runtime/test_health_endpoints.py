"""Health endpoint tests using fakes."""

from fastapi.testclient import TestClient

from src.main import create_app
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


def _app_with_deps(event_bus, registry, kg, llm):
    app = create_app()
    app.state.event_bus = event_bus
    app.state.ability_registry = registry
    app.state.kg = kg
    app.state.llm_model = llm
    return app


def test_health_endpoints_ok():
    app = _app_with_deps(
        FakeEventBus(), FakeAbilityRegistry(), FakeKG(), FakeLLM()
    )
    client = TestClient(app)
    for path in ["/health", "/healthz"]:
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["components"]["event_bus"]["status"] == "ok"


def test_health_endpoint_dependency_failure():
    class BrokenBus(FakeEventBus):
        async def emit(self, event):
            raise RuntimeError("bus down")

    app = _app_with_deps(
        BrokenBus(), FakeAbilityRegistry(), FakeKG(), FakeLLM()
    )
    client = TestClient(app)
    for path in ["/health", "/healthz"]:
        resp = client.get(path)
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["components"]["event_bus"]["status"] == "unhealthy"
        assert "bus down" in data["components"]["event_bus"]["error"]
