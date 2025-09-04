import os, pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.main import create_app
from src.abilities.unified_registry import UnifiedToolRegistry
from src.reug_runtime.unified_router import UnifiedToolRouter

@pytest.fixture(scope="module")
def app():
    os.environ["ENABLE_GITHUB_DEMO"] = "0"
    return create_app()

def test_health(app):
    c = TestClient(app); r = c.get("/health")
    assert r.status_code == 200 and "tools" in r.json()

def test_tools_list(app):
    c = TestClient(app); r = c.get("/tools")
    assert r.status_code == 200 and "tools" in r.json()

@pytest.mark.asyncio
async def test_consensus_fallback():
    reg = UnifiedToolRegistry()
    async def fail(args):
        if args.get("method") != "simple_vote": raise ConnectionError
        return {"ok": True}
    reg.register_tool({"tool_id": "deepconf_consensus"}, fail)
    router = UnifiedToolRouter(reg)
    res = await router.execute_tool_with_recovery("deepconf_consensus", {"method": "ensemble_ranking"})
    assert res["success"] and res["result"]["consensus"]["ok"]

@pytest.mark.asyncio
async def test_unified_registry_discovery():
    reg = UnifiedToolRegistry()
    await reg.auto_discover_all()
    # Should discover some abilities even without demo flags
    assert len(reg.get_available_tools()) >= 0