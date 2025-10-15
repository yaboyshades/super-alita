from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.clients.llm_orchestrator import LLMOrchestrator


class DeterministicClient:
    def __init__(self, response: str = "deterministic response") -> None:
        self.response = response
        self.calls = 0

    async def generate(self, prompt: str, preferences: dict[str, Any]) -> str:
        self.calls += 1
        return self.response

    async def stream_generate(self, prompt: str, preferences: dict[str, Any]):
        yield self.response


class FailingClient(DeterministicClient):
    async def generate(self, prompt: str, preferences: dict[str, Any]) -> str:
        self.calls += 1
        raise RuntimeError("failure")


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep)


@pytest.mark.asyncio()
async def test_orchestrator_caches_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = LLMOrchestrator()
    deterministic = DeterministicClient("cached response")
    providers = {name: DeterministicClient() for name in orchestrator.providers}
    providers["gemini"] = deterministic
    orchestrator.providers = providers

    response_one = await orchestrator.generate("hello", {"mode": "test"})
    assert response_one.cached is False
    assert deterministic.calls == 1

    response_two = await orchestrator.generate("hello", {"mode": "test"})
    assert response_two.cached is True
    assert deterministic.calls == 1
    assert response_two.content == "cached response"


@pytest.mark.asyncio()
async def test_orchestrator_fallbacks_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = LLMOrchestrator()
    failing = FailingClient()
    backup = DeterministicClient("fallback success")
    orchestrator.providers["gemini"] = failing
    orchestrator.providers["ollama"] = backup

    result = await orchestrator.generate("need response", {})
    assert result.provider == "ollama"
    telemetry = orchestrator.get_telemetry()
    assert telemetry["overall"]["requests_total"] >= 2
    assert telemetry["overall"]["requests_failed"] >= 1
