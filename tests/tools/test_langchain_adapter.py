import pytest

from tools.langchain_adapter import AlitaToolRunnable


class AsyncFailingTool:
    async def aexecute(self, **_: object) -> None:
        raise RuntimeError("async boom")


class SyncFailingTool:
    def run(self, **_: object) -> None:
        raise RuntimeError("sync boom")


@pytest.mark.asyncio
async def test_ainvoke_handles_async_failure() -> None:
    runnable = AlitaToolRunnable(AsyncFailingTool())
    result = await runnable.ainvoke({})
    assert result == {"error": "async boom", "success": False}


@pytest.mark.asyncio
async def test_ainvoke_handles_sync_failure() -> None:
    runnable = AlitaToolRunnable(SyncFailingTool())
    result = await runnable.ainvoke({})
    assert result == {"error": "sync boom", "success": False}
