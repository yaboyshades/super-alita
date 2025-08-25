import pytest
from fastapi import FastAPI
from reug_runtime.router import router

from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


def _mk_app():
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FakeAbilityRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = FakeLLM()
    return app


@pytest.mark.skip("client disconnect simulation not supported in test environment")
def test_client_disconnect(monkeypatch):
    pass
