import asyncio
import pytest

from src.adapters.langchain_adapter import LangChainAdapter


class EchoChain:
    async def ainvoke(self, prompt: str) -> str:
        await asyncio.sleep(0)
        return f"echo:{prompt}"


def test_invoke_without_running_loop() -> None:
    adapter = LangChainAdapter(EchoChain())
    assert adapter.invoke("hi") == "echo:hi"


@pytest.mark.asyncio
async def test_invoke_with_running_loop() -> None:
    adapter = LangChainAdapter(EchoChain())
    # Call invoke while this coroutine's event loop is running
    result = adapter.invoke("hello")
    assert result == "echo:hello"
