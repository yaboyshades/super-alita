from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mcp_server" / "src"))

from mcp_server.server import app
import mcp_server.tools as tools_pkg
import toolforge
from toolforge import PromptDefinition, ResourceDescriptor


@pytest.mark.asyncio
async def test_resources_and_prompts_registered() -> None:
    toolforge.RESOURCES.append(
        ResourceDescriptor(name="demo", mime_type="text/plain", content="hi")
    )
    toolforge.PROMPTS.append(
        PromptDefinition(name="wave", description="Say hi", content="hello")
    )

    importlib.reload(tools_pkg)

    resources = {r.name for r in await app.list_resources()}
    prompts = {p.name for p in await app.list_prompts()}

    assert "demo" in resources
    assert "wave" in prompts
