
import asyncio
import socket

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from reug_runtime.router import router

from tests.runtime import prefix_path
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


def _mk_app():
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FakeAbilityRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = FakeLLM()
    return app


@pytest.mark.asyncio
async def test_client_disconnect():
    app = _mk_app()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient(base_url=f"http://{host}:{port}") as client:
              async with client.stream(
                  "POST",
                  prefix_path("/v1/chat/stream"),
                  json={"message": "hi", "session_id": "s1"},
              ) as resp:
                chunk_iter = resp.aiter_text()
                first_chunk = await chunk_iter.__anext__()
                assert first_chunk
                await resp.aclose()
        await asyncio.sleep(0.1)
    finally:
        server.should_exit = True
        await server_task
    events = app.state.event_bus.events
    assert {"type": "TaskFailed", "reason": "client_disconnected"} in events
    assert not any(e["type"] == "TaskSucceeded" for e in events)
