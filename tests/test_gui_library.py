from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from src.main import app

transport = ASGITransport(app=app)


@pytest.mark.asyncio
async def test_gui_index() -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/gui")
        assert r.status_code == 200
        text = r.text.lower()
        assert "super alita gui" in text
        assert "components" in text


@pytest.mark.asyncio
async def test_gui_component_status_badge() -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/gui/components/status_badge")
        assert r.status_code == 200
        assert "badge" in r.text


@pytest.mark.asyncio
async def test_gui_component_panel_snapshot() -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(
            "/gui/components/panel",
            params={"props": '{"title":"T","body":"B"}'},
        )
        assert r.status_code == 200
        # Basic snapshot characteristics
        text = r.text.replace("\n", "").strip()
        # Look for header/body markers with title/body substituted
        assert "panel-header" in text and ">T<" in text
        assert "panel-body" in text and ">B<" in text


@pytest.mark.asyncio
async def test_gui_schema_form_mapping() -> None:
    schema = {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string", "title": "Full Name"},
            "age": {"type": "integer"},
            "flag": {"type": "boolean"},
        },
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(
            "/gui/components/schema_form",
            params={"props": json_dumps({"schema": schema})},
        )
        assert r.status_code == 200
        body = r.text
        # Ensure each field is represented
        assert "Full Name" in body
        assert "name'" in body
        assert "age'" in body
        assert "checkbox" in body  # boolean mapping


@pytest.mark.asyncio
async def test_gui_list_components() -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/gui/components")
        assert r.status_code == 200
        data = r.json()
        assert "status_badge" in data.get("components", [])


def json_dumps(obj: Any) -> str:  # helper to avoid importing json at top-level
    import json

    return json.dumps(obj)
