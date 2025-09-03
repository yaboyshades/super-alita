"""Smoke test for task planner ability."""

import pytest
from fastapi.testclient import TestClient

import src.main as main


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Create test app with temp event log directory."""
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(tmp_path))
    app = main.create_app()

    # Use context manager to ensure lifespan events run
    with TestClient(app) as client:
        yield client


def test_planner_ability_registration(app_client):
    """Verify task_planner tool is registered in tools catalog."""
    resp = app_client.get("/tools/catalog")
    assert resp.status_code == 200

    catalog = resp.json()
    assert isinstance(catalog, list)

    # Find task_planner in catalog
    planner_tool = None
    print(f"Catalog length: {len(catalog)}")
    if catalog:
        print(f"First tool structure: {catalog[0]}")

    tool_names = [tool.get("name") for tool in catalog]
    print(f"All tool names in catalog: {tool_names}")

    for tool in catalog:
        if tool.get("name") == "task_planner":
            planner_tool = tool
            break

    assert planner_tool is not None, f"task_planner not found: {tool_names}"
    assert (
        planner_tool["description"]
        == "Decompose objectives into atomic, tool-oriented steps"
    )
    assert "input_schema" in planner_tool
    assert "output_schema" in planner_tool


def test_planner_ability_execution(app_client):
    """Test basic execution of task_planner ability."""
    payload = {
        "prompt": "Create a Python web scraper for news articles",
        "max_steps": 4,
    }

    resp = app_client.post("/ability/execute/task_planner", json=payload)
    assert resp.status_code == 200

    response_data = resp.json()
    assert response_data["ok"] is True
    assert response_data["tool"] == "task_planner"

    # Extract the actual result from the API envelope
    result = response_data["result"]
    assert "steps" in result
    assert "summary" in result
    assert "source" in result

    # Validate steps structure
    steps = result["steps"]
    assert isinstance(steps, list)
    assert len(steps) >= 1
    assert len(steps) <= 4  # max_steps constraint

    for step in steps:
        assert "id" in step
        assert "action" in step
        assert "rationale" in step
        assert isinstance(step["id"], int)
        assert isinstance(step["action"], str)
        assert isinstance(step["rationale"], str)
        assert step["id"] >= 1
        assert len(step["action"]) > 0

    # Validate source is one of expected values
    assert result["source"] in ["llm", "heuristic", "fallback"]


def test_planner_ability_empty_prompt(app_client):
    """Test planner handles empty prompt gracefully."""
    payload = {"prompt": ""}

    resp = app_client.post("/ability/execute/task_planner", json=payload)
    assert resp.status_code == 200

    result = resp.json()
    assert result["source"] == "fallback"
    assert len(result["steps"]) == 1
    assert "clarify objective" in result["steps"][0]["action"]


def test_planner_ability_max_steps_constraint(app_client):
    """Test max_steps parameter is respected."""
    payload = {
        "prompt": "Build a complete e-commerce platform with user auth, product catalog, shopping cart, payment processing, order management, and admin dashboard",
        "max_steps": 3,
    }

    resp = app_client.post("/ability/execute/task_planner", json=payload)
    assert resp.status_code == 200

    result = resp.json()
    assert len(result["steps"]) <= 3


def test_planner_ability_missing_prompt(app_client):
    """Test planner requires prompt parameter."""
    payload = {"max_steps": 5}

    resp = app_client.post("/ability/execute/task_planner", json=payload)
    # Should return 400 for missing required field
    assert resp.status_code == 400


def test_planner_ability_default_max_steps(app_client):
    """Test planner uses default max_steps when not provided."""
    payload = {"prompt": "Analyze website performance"}

    resp = app_client.post("/ability/execute/task_planner", json=payload)
    assert resp.status_code == 200

    result = resp.json()
    # Should respect default max_steps (6) - won't exceed it
    assert len(result["steps"]) <= 6
