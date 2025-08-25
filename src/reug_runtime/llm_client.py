from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator

from .config import SETTINGS


class LLMClient:
    """Base class for streaming LLM providers."""

    model_name: str

    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float | None = None
    ) -> AsyncGenerator[dict[str, str], None]:
        """Stream chat completion chunks.

        Args:
            messages: Conversation history.
            timeout: Optional override for the streaming timeout.
        """
        raise NotImplementedError


class GeminiClient(LLMClient):
    """Client backed by google-generativeai."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            import google.generativeai as genai  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            genai = None
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if genai and key:
            genai.configure(api_key=key)
            self._model = genai.GenerativeModel(model_name)
        else:
            self._model = None

    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float | None = None
    ) -> AsyncGenerator[dict[str, str], None]:
        if self._model is None:  # pragma: no cover - requires SDK
            raise RuntimeError("Gemini SDK not available")
        timeout = timeout or SETTINGS.model_stream_timeout_s
        async with asyncio.timeout(timeout):
            req = [m["content"] for m in messages]
            resp = await self._model.generate_content_async(req, stream=True)
            async for chunk in resp:
                text = getattr(chunk, "text", "")
                if not text:
                    try:
                        text = chunk.candidates[0].content.parts[0].text
                    except Exception:
                        text = ""
                if text:
                    yield {"content": text}


class OpenAIClient(LLMClient):
    """Client using the OpenAI async API."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            AsyncOpenAI = None
        api_key = os.getenv("OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=api_key) if AsyncOpenAI and api_key else None

    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float | None = None
    ) -> AsyncGenerator[dict[str, str], None]:
        if self._client is None:  # pragma: no cover - requires SDK
            raise RuntimeError("OpenAI SDK not available")
        timeout = timeout or SETTINGS.model_stream_timeout_s
        async with asyncio.timeout(timeout):
            stream = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                text = chunk.choices[0].delta.get("content", "")
                if text:
                    yield {"content": text}


class AnthropicClient(LLMClient):
    """Client using the Anthropic async API."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            AsyncAnthropic = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=api_key) if AsyncAnthropic and api_key else None

    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float | None = None
    ) -> AsyncGenerator[dict[str, str], None]:
        if self._client is None:  # pragma: no cover - requires SDK
            raise RuntimeError("Anthropic SDK not available")
        timeout = timeout or SETTINGS.model_stream_timeout_s
        async with asyncio.timeout(timeout):
            stream = await self._client.messages.stream(
                model=self.model_name,
                messages=messages,
            )
            async for chunk in stream:
                text = getattr(getattr(chunk, "delta", None), "text", "")
                if not text and getattr(chunk, "message", None):
                    try:
                        text = chunk.message.content[0].text
                    except Exception:
                        text = ""
                if text:
                    yield {"content": text}


class MockLLMClient(LLMClient):
    """Deterministic mock used for development and tests."""

    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float | None = None
    ) -> AsyncGenerator[dict[str, str], None]:
        timeout = timeout or SETTINGS.model_stream_timeout_s
        async with asyncio.timeout(timeout):
            has_result = any(
                m["role"] == "assistant" and "<tool_result" in m["content"]
                for m in messages
            )
            if not has_result:
                await asyncio.sleep(0)
                yield {"content": "Thinking... "}
                await asyncio.sleep(0)
                yield {
                    "content": '<tool_call>{"tool":"echo","args":{"payload":"hi"}}</tool_call>'
                }
            else:
                await asyncio.sleep(0)
                yield {
                    "content": '<final_answer>{"content":"done: hi","citations":[]}</final_answer>'
                }


def get_llm_client(model_name: str | None) -> LLMClient:
    """Factory selecting an LLM client based on model name."""
    if not model_name:
        return MockLLMClient()
    m = model_name.lower()
    if m.startswith("gemini"):
        return GeminiClient(model_name)
    if m.startswith("gpt") or m.startswith("openai"):
        return OpenAIClient(model_name)
    if m.startswith("claude"):
        return AnthropicClient(model_name)
    return MockLLMClient()


__all__ = [
    "LLMClient",
    "GeminiClient",
    "OpenAIClient",
    "AnthropicClient",
    "MockLLMClient",
    "get_llm_client",
]
