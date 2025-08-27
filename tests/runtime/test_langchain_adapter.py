import asyncio
import pytest

from src.adapters.langchain_adapter import AlitaToolRunnable


async def echo_tool(args: dict):
    return args["input"]


def test_ainvoke_validates_input_type():
    runnable = AlitaToolRunnable(echo_tool)

    with pytest.raises(TypeError):
        asyncio.run(runnable.ainvoke("not a dict"))

    result = asyncio.run(runnable.ainvoke({"input": "hello"}))
    assert result == "hello"
