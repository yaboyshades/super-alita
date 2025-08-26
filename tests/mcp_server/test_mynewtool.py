"""Tests for the simple string_info MCP tool."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mcp_server" / "src"))

from mcp_server.tools.mynewtool import MyNewTool


@pytest.mark.asyncio
async def test_mynewtool_success() -> None:
    result = await MyNewTool("hello")
    assert result == {"original": "hello", "upper": "HELLO", "length": 5}


@pytest.mark.asyncio
async def test_mynewtool_validation() -> None:
    result_empty = await MyNewTool("")
    assert result_empty["error"] == "text must be a non-empty string"

    result_type = await MyNewTool(123)  # type: ignore[arg-type]
    assert result_type["error"] == "text must be a string"

