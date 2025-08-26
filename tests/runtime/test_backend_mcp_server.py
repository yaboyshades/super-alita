from backend.mcp_server import fetch, search
import pytest
import asyncio


@pytest.mark.asyncio
async def test_search_and_fetch_tools() -> None:
    results = await search("dark")
    assert results == ["2"]

    data = await fetch("2")
    assert data["title"] == "Add dark mode"

    with pytest.raises(KeyError):
        await fetch("999")
