"""Integration test chaining /sdd/specify -> /sdd/plan -> /sdd/tasks.

This uses FastAPI TestClient against the SDD router only (unit-style integration).
We avoid external network and ensure minimal assumptions.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.sdd.router import create_sdd_router

# Suppress benign SyntaxWarning coming from docstrings or imported modules during this test run
pytestmark = pytest.mark.filterwarnings("ignore::SyntaxWarning")


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(create_sdd_router())
    return app


def test_sdd_chain_flow_smoke():
    app = create_app()
    client = TestClient(app)

    # 1) specify
    spec_payload = {
        "user_input": "Build a todo API with create/list/delete endpoints",
        "constitutional_gates": True,
    }
    r1 = client.post("/sdd/specify", json=spec_payload)
    assert r1.status_code == 200, r1.text
    spec = r1.json()
    assert spec.get("success") is True
    assert spec.get("feature_id")
    assert spec.get("specification")
    assert "overall_compliance_score" in spec

    # 2) plan
    plan_payload = {
        "feature_id": spec.get("feature_id"),
        "specification": spec.get("specification"),
        "constitutional_gates": True,
    }
    r2 = client.post("/sdd/plan", json=plan_payload)
    assert r2.status_code == 200, r2.text
    plan = r2.json()
    assert plan.get("success") is True
    assert plan.get("plan")
    assert "overall_compliance_score" in plan

    # 3) tasks
    tasks_payload = {
        "feature_id": spec.get("feature_id"),
        "plan": plan.get("plan"),
        "constitutional_gates": True,
    }
    r3 = client.post("/sdd/tasks", json=tasks_payload)
    assert r3.status_code == 200, r3.text
    tasks = r3.json()
    assert tasks.get("success") is True
    assert tasks.get("tasks")
    assert isinstance(tasks.get("tasks"), list)
    assert "overall_compliance_score" in tasks
