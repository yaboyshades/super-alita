import pytest
from hypothesis import given
from hypothesis import strategies as st

from mcp_local.registry import ToolRegistry as LocalRegistry
from super_alita_mcp.registry import ToolRegistry as RemoteRegistry


class EchoTool:
    async def __call__(self, text: str) -> str:
        return text.upper()

    async def astream(self, text: str):
        for ch in text.upper():
            yield ch


class PlainTool:
    async def __call__(self, text: str) -> str:
        return text.lower()


@pytest.mark.asyncio
@given(st.text(max_size=20))
@pytest.mark.parametrize("Registry", [LocalRegistry, RemoteRegistry])
async def test_stream_matches_invoke(Registry, text: str) -> None:
    registry = Registry()
    registry.register("echo", EchoTool())
    result = await registry.invoke("echo", {"text": text})
    chunks: list[str] = []
    async for chunk in registry.invoke_stream("echo", {"text": text}):
        chunks.append(chunk)
    assert "".join(chunks) == result


@pytest.mark.asyncio
@given(st.text(max_size=20))
@pytest.mark.parametrize("Registry", [LocalRegistry, RemoteRegistry])
async def test_invoke_stream_falls_back(Registry, text: str) -> None:
    registry = Registry()
    registry.register("plain", PlainTool())
    result = await registry.invoke("plain", {"text": text})
    chunks: list[str] = []
    async for chunk in registry.invoke_stream("plain", {"text": text}):
        chunks.append(chunk)
    assert chunks == [result]
