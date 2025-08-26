import asyncio
import pytest

from reug_runtime.llm_client import (
    AnthropicClient,
    GeminiClient,
    MockLLMClient,
    OpenAIClient,
    get_llm_client,
)
from reug_runtime.config import SETTINGS


def test_provider_selection() -> None:
    assert isinstance(get_llm_client("gemini-1.5"), GeminiClient)
    assert isinstance(get_llm_client("gpt-4"), OpenAIClient)
    assert isinstance(get_llm_client("claude-3"), AnthropicClient)
    assert isinstance(get_llm_client("other"), MockLLMClient)


class SlowMock(MockLLMClient):
    async def stream_chat(self, messages, timeout: float | None = None):
        timeout = timeout or SETTINGS.model_stream_timeout_s
        async with asyncio.timeout(timeout):
            await asyncio.sleep(timeout + 0.1)
            yield {"content": "late"}


@pytest.mark.asyncio
async def test_timeout_enforced(monkeypatch) -> None:
    monkeypatch.setattr(SETTINGS, "model_stream_timeout_s", 0.01)
    client = SlowMock()
    with pytest.raises(TimeoutError):
        async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
            pass
