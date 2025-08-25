from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncGenerator
from typing import Any  # noqa: F401 (used in type comments and optional telemetry)

try:  # lightweight optional import
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None

try:  # pragma: no cover - optional dependency
    from telemetry import EventTypes, broadcast_agent_event  # type: ignore
except Exception:  # pragma: no cover

    async def broadcast_agent_event(  # type: ignore
        *_, **__
    ) -> None:  # noqa: D401
        """No-op broadcast when telemetry not available."""
        return None

    class EventTypes:  # type: ignore
        PERFORMANCE_METRIC = "performance_metric"
        CONVERSATION = "conversation"


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
        self._client = (
            AsyncAnthropic(api_key=api_key) if AsyncAnthropic and api_key else None
        )

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
                    "content": (
                        '<tool_call>{"tool":"echo","args":{"payload":"hi"}}</tool_call>'
                    )
                }
            else:
                await asyncio.sleep(0)
                yield {
                    "content": (
                        '<final_answer>{"content":"done: hi","citations":[]}'
                        "</final_answer>"
                    )
                }


class SuperAlitaFallbackClient(LLMClient):
    """Client that proxies to a local Super Alita OpenAI-compatible adapter.

    Expected environment variables:
      - SUPER_ALITA_BASE_URL (default: http://127.0.0.1:8080)
      - SUPER_ALITA_API_KEY  (optional; added as Authorization: Bearer)
      - SUPER_ALITA_MODEL    (optional; model name to send downstream)

    This client attempts to stream via /v1/chat/completions or /v1/completions
    endpoints (OpenAI-compatible). It yields text deltas as {"content": str}.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "SUPER_ALITA_MODEL", "gpt-oss-20b-4bit"
        )
        self.base_url = os.getenv("SUPER_ALITA_BASE_URL", "http://127.0.0.1:8080")
        self.api_key = os.getenv("SUPER_ALITA_API_KEY")
        self._client = None
        if httpx:  # pragma: no branch - simple guard
            self._client = httpx.AsyncClient(timeout=None)
        # Schedule a lightweight telemetry that we instantiated fallback (best-effort)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                broadcast_agent_event(
                    event_type=getattr(EventTypes, "LLM_FALLBACK", "llm_fallback"),
                    source="llm_client_factory",
                    data={
                        "selected": "super_alita_fallback",
                        "model": self.model_name,
                        "base_url": self.base_url,
                    },
                )
            )
        except RuntimeError:  # no running loop at import/startup
            pass

    async def _stream_openai_style(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        if self._client is None:  # pragma: no cover
            raise RuntimeError("httpx not available for SuperAlitaFallbackClient")
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "temperature": 0.2,
        }
        url_primary = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        url_alt = f"{self.base_url.rstrip('/')}/v1/completions"
        # Try chat endpoint first
        try_endpoints = [url_primary, url_alt]
        last_error: Exception | None = None
        for endpoint in try_endpoints:
            try:
                async with self._client.stream(  # type: ignore[arg-type]
                    "POST", endpoint, headers=headers, json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        text = await resp.aread()
                        raise RuntimeError(
                            f"{endpoint} {resp.status_code}: {text[:200]!r}"
                        )
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[len("data: ") :].strip()
                        else:
                            data_str = line.strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                        except Exception:
                            continue
                        # OpenAI style streaming: choices[].delta.content
                        choice = None
                        if isinstance(obj, dict):
                            choice = obj.get("choices", [{}])[0]
                        if choice and isinstance(choice, dict):
                            delta = choice.get("delta") or choice.get("message") or {}
                            content = delta.get("content") or choice.get("text")
                            if content:
                                yield content
                    return
            except Exception as e:  # pragma: no cover - network errors
                last_error = e
                continue
        if last_error:
            raise last_error

    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float | None = None
    ) -> AsyncGenerator[dict[str, str], None]:
        # Wrap streaming with timeout + telemetry events
        effective_timeout = timeout or SETTINGS.model_stream_timeout_s
        start = asyncio.get_event_loop().time()
        try:
            async with asyncio.timeout(effective_timeout):
                async for text in self._stream_openai_style(messages):
                    yield {"content": text}
        finally:
            dur = asyncio.get_event_loop().time() - start
            with contextlib.suppress(Exception):  # pragma: no cover
                await broadcast_agent_event(
                    event_type=EventTypes.PERFORMANCE_METRIC,
                    source="super_alita_fallback_llm",
                    data={
                        "metric": "llm_stream_duration_s",
                        "value": dur,
                        "model": self.model_name,
                    },
                )


def get_llm_client(model_name: str | None) -> LLMClient:
    """Factory selecting an LLM client based on model name."""
    if not model_name:
        return MockLLMClient()
    m = model_name.lower()
    if m == "auto":
        # Prefer Gemini if key present, else fallback
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            try:
                return GeminiClient("gemini-1.5-flash")
            except Exception:  # pragma: no cover
                return SuperAlitaFallbackClient()
        return SuperAlitaFallbackClient()
    if m.startswith("gemini"):
        try:
            return GeminiClient(model_name)
        except Exception:  # pragma: no cover
            # Fallback to Super Alita local client if available
            return SuperAlitaFallbackClient()
    if m.startswith("gpt") or m.startswith("openai"):
        return OpenAIClient(model_name)
    if m.startswith("claude"):
        return AnthropicClient(model_name)
    if m in {"super-alita", "super_alita", "alita"}:
        return SuperAlitaFallbackClient()
    return MockLLMClient()


class LocalHFClient(LLMClient):
    """Local Hugging Face transformers client (non-streaming token slicing).

    Environment variables:
      - LOCAL_HF_MODEL_ID (e.g. gpt2)
      - LOCAL_HF_MAX_NEW_TOKENS (int, default 128)
      - LOCAL_HF_DEVICE (cuda|cpu|auto) default auto
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("LOCAL_HF_MODEL_ID", "gpt2")
        self.max_new = int(os.getenv("LOCAL_HF_MAX_NEW_TOKENS", "128"))
        self.device_pref = os.getenv("LOCAL_HF_DEVICE", "auto")
        try:  # lazy import
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("transformers/torch not available") from e
        self.torch = torch
        device = (
            0
            if (self.device_pref in {"auto", "cuda"} and torch.cuda.is_available())
            else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(device)
        self.device = device

    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float | None = None
    ) -> AsyncGenerator[dict[str, str], None]:
        # Simple implementation: generate then yield chunks of decoded text
        timeout = timeout or SETTINGS.model_stream_timeout_s
        async with asyncio.timeout(timeout):
            # Build a single prompt from user + last assistant maybe
            user_parts = [m["content"] for m in messages if m["role"] == "user"]
            prompt = user_parts[-1] if user_parts else ""
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.device
            )
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            gen_ids = output[0][input_ids.shape[-1] :]
            # Slice into pseudo streaming chunks
            text_full = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            # Emit in 40 char chunks
            for i in range(0, len(text_full), 40):
                await asyncio.sleep(0)
                chunk = text_full[i : i + 40]
                if chunk:
                    yield {"content": chunk}


# Update factory to include hf: prefix logic
old_get_llm_client = get_llm_client


def get_llm_client(model_name: str | None) -> LLMClient:  # type: ignore[override]
    if model_name and model_name.lower().startswith("hf:"):
        target = model_name.split(":", 1)[1] or None
        return LocalHFClient(target)
    return old_get_llm_client(model_name)


__all__ = [
    "LLMClient",
    "GeminiClient",
    "OpenAIClient",
    "AnthropicClient",
    "MockLLMClient",
    "SuperAlitaFallbackClient",
    "LocalHFClient",
    "get_llm_client",
]
