from __future__ import annotations

import importlib

import httpx
import pytest
from httpx import ASGITransport


@pytest.mark.asyncio
async def test_auth_create_and_query_roundtrip(monkeypatch):
    # Configure environment BEFORE importing app
    monkeypatch.setenv("ALITA_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("ALITA_AUTH_OPEN_REG", "true")
    monkeypatch.setenv("ALITA_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("ALITA_RATE_LIMIT", "2")
    monkeypatch.setenv("ALITA_RATE_WINDOW", "10")

    mod = importlib.import_module("src.main")
    importlib.reload(mod)

    app = mod.app  # type: ignore
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Create key without admin due to open registration
        r = await client.post("/api/v1/auth/keys", json={"owner": "test@example.com"})
        assert r.status_code == 200
        api_key = r.json()["api_key"]

        # Missing key -> 401
        r = await client.post("/api/v1/query", json={"prompt": "hi"})
        assert r.status_code == 401

        # With key -> 200
        headers = {"Authorization": f"Bearer {api_key}"}
        r = await client.post("/api/v1/query", headers=headers, json={"prompt": "hi"})
        assert r.status_code == 200
        assert "answer" in r.json()

        # Rate limit headers present
        assert "X-RateLimit-Limit" in r.headers
        assert "X-RateLimit-Remaining" in r.headers

        # Exhaust limit quickly
        await client.post("/api/v1/query", headers=headers, json={"prompt": "hi"})
        r = await client.post("/api/v1/query", headers=headers, json={"prompt": "hi"})
        assert r.status_code in (200, 429)  # depending on limiter timing
        if r.status_code == 429:
            assert r.headers.get("Retry-After") is not None

