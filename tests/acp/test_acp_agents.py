"""Tests for ACP agents."""
import asyncio
import contextlib

import pytest

acp_sdk = pytest.importorskip("acp_sdk")
try:  # pragma: no cover - optional dependency
    from acp_sdk import Client, Message, MessagePart
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("acp_sdk client not available", allow_module_level=True)

from src.acp_app.server import main as acp_main


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def _acp_server():
    task = asyncio.create_task(acp_main())
    await asyncio.sleep(0.6)
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.fixture
async def client():
    async with Client(base_url="http://localhost:8000") as c:
        yield c


@pytest.mark.asyncio
async def test_echo_roundtrip(client: Client):
    messages = [Message(parts=[MessagePart(text="Test echo")])]
    response = await client.run_sync("echo", messages)

    assert len(response) == 1
    assert "Echo: Test echo" in response[0].parts[0].text


@pytest.mark.asyncio
async def test_classify_agent(client: Client):
    messages = [
        Message(parts=[MessagePart(text="This is a test message for classification")])
    ]
    response = await client.run_sync("classify", messages)

    assert len(response) == 1
    result_text = response[0].parts[0].text
    assert "classification" in result_text
    assert "confidence" in result_text


@pytest.mark.asyncio
async def test_router_stream(client: Client):
    messages = [Message(parts=[MessagePart(text="Route this")])]

    chunks = []
    async for msg in client.run("router", messages):
        for part in msg.parts:
            if getattr(part, "text", None):
                chunks.append(part.text)

    assert len(chunks) >= 2
    assert "Routing:" in chunks[0]


@pytest.mark.asyncio
async def test_search_agent(client: Client):
    messages = [Message(parts=[MessagePart(text="what is RAG", metadata={"mode": "web"})])]
    response = await client.run_sync("search", messages)

    full_text = "".join(
        part.text
        for msg in response
        for part in msg.parts
        if getattr(part, "text", None)
    )
    assert "Search Results" in full_text or "summary" in full_text.lower()
