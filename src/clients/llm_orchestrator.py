"""Multi-provider LLM orchestration with caching and telemetry."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from cachetools import TTLCache


@dataclass(slots=True)
class LLMResponse:
    """Structured response payload from the orchestrator."""

    content: str
    provider: str
    latency: float
    cost: float
    cached: bool = False
    error: str | None = None


@dataclass(slots=True)
class ProviderConfig:
    """Static configuration for each provider."""

    name: str
    cost_per_token: float
    max_tokens: int
    priority: int
    enabled: bool = True


class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def can_request(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class TelemetryCollector:
    """Aggregates telemetry for orchestrated calls."""

    def __init__(self) -> None:
        self.metrics: dict[str, float | int] = {
            "requests_total": 0,
            "requests_failed": 0,
            "total_cost": 0.0,
            "total_latency": 0.0,
        }
        self.provider_metrics: dict[str, dict[str, float | int]] = {}

    def record_request(self, provider: str, latency: float, cost: float, success: bool) -> None:
        self.metrics["requests_total"] += 1
        self.metrics["total_latency"] += latency
        self.metrics["total_cost"] += cost
        if not success:
            self.metrics["requests_failed"] += 1

        provider_metric = self.provider_metrics.setdefault(
            provider,
            {"requests": 0, "failures": 0, "total_latency": 0.0, "total_cost": 0.0},
        )
        provider_metric["requests"] += 1
        provider_metric["total_latency"] += latency
        provider_metric["total_cost"] += cost
        if not success:
            provider_metric["failures"] += 1

    def get_provider_stats(self, provider: str) -> dict[str, float]:
        if provider not in self.provider_metrics:
            return {
                "success_rate": 0.0,
                "avg_latency": 0.0,
                "avg_cost": 0.0,
                "total_requests": 0,
            }
        metrics = self.provider_metrics[provider]
        requests = int(metrics["requests"])
        failures = int(metrics["failures"])
        success_rate = (requests - failures) / requests if requests else 0.0
        avg_latency = metrics["total_latency"] / requests if requests else 0.0
        avg_cost = metrics["total_cost"] / requests if requests else 0.0
        return {
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "avg_cost": avg_cost,
            "total_requests": requests,
        }


class LLMOrchestrator:
    """Selects the best available provider for LLM requests."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.providers = self._initialize_providers()
        self.circuit_breakers = {name: CircuitBreaker() for name in self.providers}
        self.response_cache: TTLCache[str, LLMResponse] = TTLCache(maxsize=1_000, ttl=300)
        self.telemetry = TelemetryCollector()
        self.provider_configs: dict[str, ProviderConfig] = {
            "gemini": ProviderConfig("gemini", 0.000001, 8_192, 1),
            "ollama": ProviderConfig("ollama", 0.0000001, 4_096, 2),
            "openai": ProviderConfig("openai", 0.000002, 4_096, 3),
            "super_alita": ProviderConfig("super_alita", 0.0000005, 2_048, 4),
            "mock": ProviderConfig("mock", 0.0, 1_024, 5),
        }
        self.fallback_chain = ["gemini", "ollama", "openai", "super_alita", "mock"]

    async def generate(self, prompt: str, preferences: dict[str, Any] | None = None) -> LLMResponse:
        preferences = preferences or {}
        cache_key = self._get_cache_key(prompt, preferences)
        cached = self.response_cache.get(cache_key)
        if cached:
            self.logger.info("Cache hit for prompt: %s", prompt[:50])
            return LLMResponse(
                content=cached.content,
                provider=cached.provider,
                latency=cached.latency,
                cost=cached.cost,
                cached=True,
            )

        provider_name = self._select_provider(prompt, preferences)
        if not provider_name:
            return LLMResponse(
                content="",
                provider="none",
                latency=0.0,
                cost=0.0,
                error="No LLM providers available",
            )

        start_time = time.time()
        last_error: Exception | None = None
        attempts = 0

        while attempts < len(self.fallback_chain):
            attempts += 1
            provider_client = self.providers[provider_name]
            breaker = self.circuit_breakers[provider_name]
            try:
                response_text = await provider_client.generate(prompt, preferences)
                latency = time.time() - start_time
                cost = self._calculate_cost(provider_name, len(prompt), len(response_text))
                breaker.record_success()
                self.telemetry.record_request(provider_name, latency, cost, True)
                response = LLMResponse(
                    content=response_text,
                    provider=provider_name,
                    latency=latency,
                    cost=cost,
                )
                self.response_cache[cache_key] = response
                self.logger.info("LLM response from %s in %.2fs", provider_name, latency)
                return response
            except Exception as exc:  # pragma: no cover - defensive guard
                last_error = exc
                breaker.record_failure()
                latency = time.time() - start_time
                self.telemetry.record_request(provider_name, latency, 0.0, False)
                self.logger.warning("Attempt failed for %s: %s", provider_name, exc)
                next_provider = self._next_provider(provider_name)
                if not next_provider:
                    break
                provider_name = next_provider
                await asyncio.sleep(min(2 ** (attempts - 1) + random.uniform(0, 1), 5))

        return LLMResponse(
            content="",
            provider="none",
            latency=time.time() - start_time,
            cost=0.0,
            error=f"All providers failed: {last_error}",
        )

    async def stream_generate(
        self, prompt: str, preferences: dict[str, Any] | None = None
    ) -> AsyncIterator[str]:
        preferences = preferences or {}
        provider_name = self._select_provider(prompt, preferences)
        if not provider_name:
            yield "Error: No LLM providers available"
            return

        provider_client = self.providers[provider_name]
        try:
            async for chunk in provider_client.stream_generate(prompt, preferences):
                yield chunk
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.error("Stream generation failed: %s", exc)
            yield f"Error: {exc}"

    def get_telemetry(self) -> dict[str, Any]:
        provider_stats = {
            provider: self.telemetry.get_provider_stats(provider) for provider in self.providers
        }
        return {
            "overall": self.telemetry.metrics,
            "providers": provider_stats,
            "cache_info": {
                "currsize": self.response_cache.currsize,
                "maxsize": self.response_cache.maxsize,
                "ttl": self.response_cache.ttl,
            },
        }

    def _initialize_providers(self) -> dict[str, Any]:
        providers: dict[str, Any] = {}
        for name in ["gemini", "ollama", "openai", "super_alita", "mock"]:
            providers[name] = MockLLMClient(name)
        return providers

    def _get_cache_key(self, prompt: str, preferences: dict[str, Any]) -> str:
        key_data = f"{prompt}:{json.dumps(preferences, sort_keys=True)}"
        return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()

    def _select_provider(self, prompt: str, preferences: dict[str, Any]) -> str | None:
        cache_key = self._get_cache_key(prompt, preferences)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key].provider

        for provider_name in self.fallback_chain:
            config = self.provider_configs.get(provider_name)
            breaker = self.circuit_breakers.get(provider_name)
            if not config or not config.enabled:
                continue
            if breaker and not breaker.can_request():
                self.logger.warning("Provider %s circuit breaker open", provider_name)
                continue
            return provider_name
        return None

    def _next_provider(self, current: str) -> str | None:
        try:
            index = self.fallback_chain.index(current)
        except ValueError:
            return None
        if index + 1 < len(self.fallback_chain):
            return self.fallback_chain[index + 1]
        return None

    def _calculate_cost(self, provider: str, input_tokens: int, output_tokens: int) -> float:
        config = self.provider_configs[provider]
        return (input_tokens + output_tokens) * config.cost_per_token


class MockLLMClient:
    """Mock LLM provider used for tests and local runs."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def generate(self, prompt: str, preferences: dict[str, Any]) -> str:
        await asyncio.sleep(0.05)
        if random.random() < 0.1:
            raise RuntimeError("Mock provider failure")
        return f"Mock response from {self.name} for: {prompt[:50]}..."

    async def stream_generate(self, prompt: str, preferences: dict[str, Any]) -> AsyncIterator[str]:
        words = f"Streaming response from {self.name}".split()
        for word in words:
            await asyncio.sleep(0.02)
            yield f"{word} "


async def create_llm_orchestrator() -> LLMOrchestrator:
    return LLMOrchestrator()


__all__ = [
    "LLMResponse",
    "ProviderConfig",
    "CircuitBreaker",
    "TelemetryCollector",
    "LLMOrchestrator",
    "MockLLMClient",
    "create_llm_orchestrator",
]
