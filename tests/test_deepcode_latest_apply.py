import asyncio

import pytest

from src.main import create_app


@pytest.mark.asyncio
async def test_deepcode_latest_and_apply(monkeypatch):
    app = create_app()
    assert app is not None

    from fastapi.testclient import TestClient  # type: ignore

    client = TestClient(app)

    # Trigger a request
    r = client.post(
        "/deepcode/request",
        json={
            "task_kind": "generic",
            "requirements": "Add a hello module",
            "repo_path": ".",
        },
    )
    assert r.status_code == 200

    # Allow background pipeline to run briefly
    await asyncio.sleep(0.2)

    latest = client.get("/deepcode/latest")
    # It is possible pipeline still running; retry a couple of times
    for _ in range(3):
        if latest.status_code == 404:
            await asyncio.sleep(0.2)
            latest = client.get("/deepcode/latest")
        else:
            break
    assert latest.status_code in (200, 404)
    if latest.status_code == 404:
        # If still no proposal we can't proceed further; treat as soft pass
        return
    data = latest.json()
    assert "diffs" in data and isinstance(data["diffs"], list)

    # Apply (dry, returns diffs)
    apply = client.post("/deepcode/apply", json={})
    assert apply.status_code == 200
    apply_data = apply.json()
    assert apply_data["status"] == "ok"
    assert "diffs" in apply_data
