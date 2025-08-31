from __future__ import annotations

import importlib

import httpx
import pytest
from httpx import ASGITransport


@pytest.mark.asyncio
async def test_ability_whitelist(monkeypatch):
    # Only allow 'echo'
    monkeypatch.setenv("ALITA_ABILITY_WHITELIST", "echo")
    # No auth required for this test
    monkeypatch.setenv("ALITA_REQUIRE_API_KEY", "false")

    mod = importlib.import_module("src.main")
    importlib.reload(mod)
    app = mod.app  # type: ignore
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Allowed
        r = await client.post("/ability/execute/echo", json={"payload": "hello"})
        assert r.status_code == 200, r.text
        # Blocked
        r = await client.post("/ability/execute/fetch_github_raw", json={"url": "https://x"})
        assert r.status_code == 403
        assert r.json().get("error") == "ability_not_allowed"


@pytest.mark.asyncio
async def test_ability_admin_only(monkeypatch):
    # Admin required for all abilities
    monkeypatch.setenv("ALITA_ABILITIES_ADMIN_ONLY", "true")
    monkeypatch.setenv("ALITA_ADMIN_KEY", "admin")
    monkeypatch.setenv("ALITA_REQUIRE_API_KEY", "false")

    mod = importlib.import_module("src.main")
    importlib.reload(mod)
    app = mod.app  # type: ignore
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Without admin -> 403
        r = await client.post("/ability/execute/echo", json={"payload": "hi"})
        assert r.status_code == 403
        # With admin key -> 200
        r = await client.post(
            "/ability/execute/echo",
            headers={"Authorization": "Bearer admin"},
            json={"payload": "hi"},
        )
        assert r.status_code == 200, r.text

