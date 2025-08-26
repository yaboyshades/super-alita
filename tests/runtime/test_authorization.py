import httpx
import pytest

from src.agents.authorization import AuthorizationManager, OAuthConfig


@pytest.mark.asyncio
async def test_exchange_token(monkeypatch):
    """AuthorizationManager exchanges code for token and stores scopes."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/token"
        data = {
            "access_token": "abc123",
            "scope": "model.read other",
        }
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        mgr = AuthorizationManager(http_client=client)
        cfg = OAuthConfig(token_url="https://auth.test/token", client_id="cid")
        token = await mgr.exchange_token(cfg, code="code123")

    assert token == "abc123"
    assert mgr.access_token == "abc123"
    assert mgr.scopes == {"model.read", "other"}


def test_model_access_gating():
    prefs = {"gpt-4": ["model.read"]}
    mgr = AuthorizationManager(model_preferences=prefs)
    mgr.scopes = {"model.read"}
    assert mgr.model_allowed("gpt-4")
    mgr.scopes = {"different"}
    assert not mgr.model_allowed("gpt-4")
