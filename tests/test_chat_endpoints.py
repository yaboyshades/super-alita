import re

import httpx
import pytest
from httpx import ASGITransport

from src.main import app  # Import the FastAPI app instance

transport = ASGITransport(app=app)


@pytest.mark.asyncio
async def test_chat_json_roundtrip() -> None:
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        r = await client.post(
            "/v1/chat", json={"q": "hello integration", "session": "test"}
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data.get("type") == "message"
        assert "content" in data
        assert (
            data["content"].lower().startswith("hello")
            or len(data["content"]) > 0
        )


@pytest.mark.asyncio
async def test_chat_sse_stream_basic() -> None:
    # Use a raw stream to capture initial few events
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        url = "/v1/chat/stream"
        params = {"q": "hi there", "session": "test"}
        # httpx can't stream SSE via GET easily with auto decode; use raw stream
        r = await client.get(url, params=params)
        assert r.status_code == 200
        text = r.text
        # Expect start, multiple content, and done markers
        assert "event: start" in text or "data:" in text
        content_events = re.findall(r"event: content", text)
        # Accept either explicit event lines or implicit data frames with content JSON
        if not content_events:
            content_payloads = re.findall(r"data: (\{.*?\})", text)
            token_count = sum(1 for s in content_payloads if '"content"' in s)
            assert (
                token_count >= 2
            ), f"expected >=2 content fragments, got {token_count}\n{text}"
        else:
            assert (
                len(content_events) >= 2
            ), f"expected >=2 content events, got {len(content_events)}"
        assert "done" in text
        # Basic well-formed JSON fragments containing at least one content key
        json_snippets: list[str] = list(re.findall(r"data: (\{.*?\})", text))
        assert any("content" in s for s in json_snippets)
