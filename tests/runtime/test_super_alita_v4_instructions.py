import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reug_runtime.router_tools import tools
from reug_runtime.tools.instructions import build_super_alita_v4_instruction_payload


@pytest.mark.asyncio
async def test_instruction_payload_structure() -> None:
    payload = await build_super_alita_v4_instruction_payload()

    assert payload["profile"]["version"] == "4.0.0"
    sections = payload["sections"]
    assert isinstance(sections, list)
    titles = {section["title"] for section in sections}
    assert {"Setup & Installation", "API Smoke Tests", "Success Criteria"}.issubset(titles)

    validation = payload["validation"]
    assert {"approved", "reasoning"}.issubset(validation.keys())
    assert isinstance(validation["approved"], bool)
    assert isinstance(validation["reasoning"], str)


def test_super_alita_instruction_endpoint_returns_payload() -> None:
    app = FastAPI()
    app.include_router(tools)

    client = TestClient(app)
    response = client.get("/tools/super_alita_v4/instructions")
    assert response.status_code == 200

    body = response.json()
    assert body["profile"]["name"] == "Super Alita Runtime"
    assert any(
        section["title"] == "Deployment Paths" for section in body.get("sections", [])
    )
    assert body["validation"].get("reasoning")

