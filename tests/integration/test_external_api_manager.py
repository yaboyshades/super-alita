import pytest

from src.integrations.external_api_manager import (
    APIResponse,
    DeepCodeAPIClient,
    ExternalAPIManager,
)


@pytest.mark.asyncio
async def test_external_api_manager_github_search_delegates(monkeypatch):
    manager = ExternalAPIManager()

    async def fake_search(_query, **_kwargs):
        return APIResponse(success=True, data={"items": [{"name": "demo"}]})

    monkeypatch.setattr(manager.clients["github"], "search_code", fake_search)

    result = await manager.github_search_code("demo")
    assert result["items"][0]["name"] == "demo"


@pytest.mark.asyncio
async def test_external_api_manager_deepcode_handles_failure(monkeypatch):
    manager = ExternalAPIManager()

    async def failing_analyze(_code, _context=None):
        return APIResponse(success=False, data={}, error="offline")

    monkeypatch.setattr(manager.clients["deepcode"], "analyze_code", failing_analyze)

    result = await manager.deepcode_analyze("print('hi')")
    assert result["error"] == "offline"


@pytest.mark.asyncio
async def test_deepcode_local_fallback_detects_security_issue():
    client = DeepCodeAPIClient()
    response = await client._local_analysis_fallback("exec('rm -rf /')", {})

    assert response.success is True
    assert any(issue["description"] for issue in response.data["security_issues"])
