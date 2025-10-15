from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp

from src.clients.llm_orchestrator import CircuitBreaker


@dataclass(slots=True)
class APIResponse:
    """Standardised response wrapper for outbound integrations."""

    success: bool
    data: Any
    cached: bool = False
    error: Optional[str] = None
    latency: float = 0.0


class RateLimiter:
    """Simple sliding window rate limiter for async integrations."""

    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self._requests: list[float] = []

    async def acquire(self) -> None:
        now = time.time()
        self._requests = [t for t in self._requests if now - t < 60]
        if len(self._requests) >= self.requests_per_minute:
            wait_time = 60 - (now - self._requests[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            now = time.time()
            self._requests = [t for t in self._requests if now - t < 60]
        self._requests.append(time.time())

    def get_remaining_requests(self) -> int:
        now = time.time()
        self._requests = [t for t in self._requests if now - t < 60]
        return max(0, self.requests_per_minute - len(self._requests))


class ResponseCache:
    """In-memory response cache with TTL support."""

    def __init__(self, ttl: int = 300) -> None:
        self._ttl = ttl
        self._cache: Dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        payload, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None
        return payload

    def set(self, key: str, data: Any) -> None:
        self._cache[key] = (data, time.time())

    def generate_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        serialised = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
        return hashlib.md5(serialised.encode("utf-8")).hexdigest()


class GitHubAPIClient:
    """GitHub API client with rate limiting and caching."""

    def __init__(self, access_token: Optional[str] = None) -> None:
        self.access_token = access_token
        self.base_url = "https://api.github.com"
        self.rate_limiter = RateLimiter(30)
        self.circuit_breaker = CircuitBreaker()
        self.cache = ResponseCache()
        self.logger = logging.getLogger(__name__)

    async def search_code(self, query: str, **kwargs: Any) -> APIResponse:
        cache_key = self.cache.generate_key("search_code", {"query": query, **kwargs})
        cached = self.cache.get(cache_key)
        if cached is not None:
            return APIResponse(success=True, data=cached, cached=True)

        if not self.circuit_breaker.can_request():
            return APIResponse(success=False, data={}, error="Circuit breaker open")

        start_time = time.time()
        await self.rate_limiter.acquire()

        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SuperAlita/1.0",
        }
        if self.access_token:
            headers["Authorization"] = f"token {self.access_token}"

        params = {"q": query, **kwargs}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/search/code",
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    latency = time.time() - start_time
                    if response.status == 403:
                        self.circuit_breaker.record_failure()
                        return APIResponse(success=False, data={}, error="Rate limit exceeded", latency=latency)
                    if response.status == 200:
                        payload = await response.json()
                        self.circuit_breaker.record_success()
                        self.cache.set(cache_key, payload)
                        return APIResponse(success=True, data=payload, latency=latency)
                    if response.status == 404:
                        self.circuit_breaker.record_success()
                        payload = {"items": []}
                        self.cache.set(cache_key, payload)
                        return APIResponse(success=True, data=payload, latency=latency)
                    self.circuit_breaker.record_failure()
                    return APIResponse(success=False, data={}, error=f"HTTP {response.status}", latency=latency)
        except asyncio.TimeoutError:
            self.circuit_breaker.record_failure()
            return APIResponse(success=False, data={}, error="Request timeout", latency=time.time() - start_time)
        except Exception as exc:  # pragma: no cover - defensive
            self.circuit_breaker.record_failure()
            return APIResponse(success=False, data={}, error=str(exc), latency=time.time() - start_time)


class DeepCodeAPIClient:
    """DeepCode API client with local analysis fallback."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key
        self.base_url = "https://api.deepcode.ai/v1"
        self.rate_limiter = RateLimiter(10)
        self.circuit_breaker = CircuitBreaker()
        self.cache = ResponseCache()
        self.logger = logging.getLogger(__name__)

    async def analyze_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> APIResponse:
        context = context or {}
        cache_key = self.cache.generate_key(
            "analyze_code",
            {"code_hash": hashlib.md5(code.encode("utf-8")).hexdigest(), "context": context},
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return APIResponse(success=True, data=cached, cached=True)

        if not self.circuit_breaker.can_request():
            return await self._local_analysis_fallback(code, context)

        start_time = time.time()
        await self.rate_limiter.acquire()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "code": code,
            "context": context,
            "options": {"languages": ["python", "javascript", "typescript"], "severity": ["high", "medium", "low"]},
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/analyze",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 202:
                        data = await response.json()
                        analysis_id = data.get("analysis_id")
                        return await self._poll_analysis_result(analysis_id, start_time, cache_key)
                    if response.status == 429:
                        self.circuit_breaker.record_failure()
                        await asyncio.sleep(10)
                        return await self.analyze_code(code, context)
                    if response.status >= 500:
                        self.circuit_breaker.record_failure()
                        return await self._local_analysis_fallback(code, context)
                    if response.status != 200:
                        self.circuit_breaker.record_failure()
                        return APIResponse(success=False, data={}, error=f"HTTP {response.status}", latency=time.time() - start_time)
                    payload = await response.json()
                    self.circuit_breaker.record_success()
                    self.cache.set(cache_key, payload)
                    return APIResponse(success=True, data=payload, latency=time.time() - start_time)
        except Exception as exc:  # pragma: no cover - defensive
            self.circuit_breaker.record_failure()
            self.logger.warning("DeepCode analysis failed (%s), using fallback", exc)
            return await self._local_analysis_fallback(code, context)

    async def _poll_analysis_result(
        self,
        analysis_id: Optional[str],
        start_time: float,
        cache_key: str,
        *,
        max_wait: int = 60,
    ) -> APIResponse:
        if not analysis_id:
            return APIResponse(success=False, data={}, error="Missing analysis id", latency=time.time() - start_time)

        poll_start = time.time()
        while time.time() - poll_start < max_wait:
            await asyncio.sleep(2)
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/analysis/{analysis_id}",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        if response.status == 202:
                            continue
                        if response.status != 200:
                            break
                        data = await response.json()
                        latency = time.time() - start_time
                        result = {
                            "analysis_id": analysis_id,
                            "confidence": data.get("confidence", 0.8),
                            "suggestions": data.get("suggestions", []),
                            "security_issues": data.get("security_issues", []),
                            "performance_hints": data.get("performance_hints", []),
                            "cached": False,
                        }
                        self.circuit_breaker.record_success()
                        self.cache.set(cache_key, result)
                        return APIResponse(success=True, data=result, latency=latency)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug("Polling failed for %s: %s", analysis_id, exc)
                break
        return APIResponse(success=False, data={}, error="Analysis timeout", latency=time.time() - start_time)

    async def _local_analysis_fallback(self, code: str, context: Dict[str, Any]) -> APIResponse:
        start_time = time.time()
        security_patterns = [
            (r"exec\(", "Use of exec() function"),
            (r"eval\(", "Use of eval() function"),
            (r"subprocess\.call", "Potential unsafe subprocess call"),
            (r"pickle\.loads", "Potential unsafe pickle.loads usage"),
            (r"password.*=.*['\"][^'\"]*['\"]", "Hardcoded password detected"),
        ]
        perf_patterns = [
            (r"for\s+.*in\s+range\(len", "Consider using enumerate()"),
            (r"==\s*None", "Use 'is None' for comparisons"),
            (r"!=\s*None", "Use 'is not None' for comparisons"),
        ]
        security_issues: list[Dict[str, Any]] = []
        for pattern, description in security_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                security_issues.append(
                    {
                        "severity": "high",
                        "description": description,
                        "line": 0,
                        "confidence": 0.7,
                    }
                )
        suggestions: list[Dict[str, Any]] = []
        for pattern, description in perf_patterns:
            if re.search(pattern, code):
                suggestions.append({"type": "performance", "suggestion": description, "confidence": 0.6})
        result = {
            "analysis_id": f"local_{hashlib.md5(code.encode('utf-8')).hexdigest()[:8]}",
            "confidence": 0.5,
            "suggestions": suggestions,
            "security_issues": security_issues,
            "performance_hints": [],
            "cached": False,
            "source": "local_fallback",
        }
        latency = time.time() - start_time
        return APIResponse(success=True, data=result, latency=latency)


class ExternalAPIManager:
    """Coordinator for outbound integrations with resilience primitives."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.clients = {
            "github": GitHubAPIClient(),
            "deepcode": DeepCodeAPIClient(),
        }
        self.logger.info("ExternalAPIManager initialised with %s clients", len(self.clients))

    async def github_search_code(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        response = await self.clients["github"].search_code(query, **kwargs)
        if response.success:
            return response.data
        self.logger.error("GitHub search failed: %s", response.error)
        return {"items": [], "error": response.error}

    async def deepcode_analyze(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = await self.clients["deepcode"].analyze_code(code, context)
        if response.success:
            return response.data
        self.logger.error("DeepCode analysis failed: %s", response.error)
        return {"error": response.error or "analysis_failed", "suggestions": [], "security_issues": []}

    async def get_service_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {}
        for name, client in self.clients.items():
            status[name] = {
                "circuit_state": client.circuit_breaker.state,
                "failures": client.circuit_breaker.failure_count,
                "can_request": client.circuit_breaker.can_request(),
            }
        return status


async def create_external_api_manager() -> ExternalAPIManager:
    return ExternalAPIManager()
