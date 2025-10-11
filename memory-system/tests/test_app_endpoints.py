from __future__ import annotations

from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from src.app import app
from src.stores.episodic import episodic_store

client = TestClient(app)


def test_capture_endpoint_filters_and_keeps_messages():
    response = client.post(
        "/capture",
        json=[
            {
                "role": "user",
                "content": "I prefer coffee over tea in the morning.",
                "meta": {"topic": "preferences"},
            },
            {
                "role": "user",
                "content": "My API key is sk-1234567890",
                "meta": {"sensitive": True},
            },
            {
                "role": "user",
                "content": "Here's some code:\n```python\nprint('hello')\n```",
                "meta": {"language": "python"},
            },
        ],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["received"] == 3
    assert payload["kept"] == 2
    assert payload["quarantined"] == 1


def test_context_endpoint_returns_citations():
    client.post(
        "/capture",
        json=[
            {
                "role": "user",
                "content": "My favorite color is blue and I love sushi.",
                "meta": {"topic": "preferences"},
            }
        ],
    )
    response = client.get("/context", params={"q": "favorite color", "k": 5, "budget": 300})
    assert response.status_code == 200
    data = response.json()
    assert data["citations"]
    assert "text" in data
    assert data["provenance"]["query"] == "favorite color"


def test_health_endpoint_reports_counts():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert set(data["stores"].keys()) == {"episodic", "semantic", "working"}


def test_explain_endpoint_returns_provenance():
    client.post(
        "/capture",
        json=[
            {"role": "user", "content": "Test memory for explanation", "meta": {}},
        ],
    )
    memories = episodic_store.search("test memory", k=1)
    assert memories
    memory_id = memories[0].id
    response = client.get(f"/explain/{memory_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["memory"]["id"] == memory_id
    assert payload["provenance"]["source"].startswith("msg_")


def test_consolidation_trigger_endpoint():
    response = client.post("/consolidate")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "started"


def test_rules_reload_endpoint():
    response = client.post("/rules/reload")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
