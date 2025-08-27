import asyncio

import pytest

from src.adapters.langchain_adapter import LangChainAdapter


class SyncTool:
    def run(self, *, x: int, y: int) -> int:
        return x + y


class CoroutineTool:
    def run(self, *, x: int, y: int):
        async def inner() -> int:
            await asyncio.sleep(0)
            return x * y

        return inner()


@pytest.mark.asyncio
async def test_sync_tool_execution():
    adapter = LangChainAdapter(SyncTool(), timeout=1.0)
    result = await adapter.run(x=1, y=2)
    assert result == 3


@pytest.mark.asyncio
async def test_coroutine_tool_execution():
    adapter = LangChainAdapter(CoroutineTool(), timeout=1.0)
    result = await adapter.run(x=2, y=3)
    assert result == 6
