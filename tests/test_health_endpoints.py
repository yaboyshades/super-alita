import pytest

from src.main import create_app


@pytest.mark.asyncio
async def test_health_and_routes_endpoints():
    app = create_app()
    from fastapi.testclient import TestClient  # type: ignore

    client = TestClient(app)

    r_health = client.get("/health/simple")
    assert r_health.status_code == 200
    payload = r_health.json()
    assert payload.get("status") == "ok"
    assert "timestamp" in payload

    r_routes = client.get("/routes")
    assert r_routes.status_code == 200
    route_list = r_routes.json()
    assert isinstance(route_list, list)
    # basic sanity: deepcode endpoints present
    expected = {"/deepcode/request", "/deepcode/latest", "/deepcode/apply"}
    got = {r.get("path") for r in route_list if isinstance(r, dict)}
    assert expected <= got
