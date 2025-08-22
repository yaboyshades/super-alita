#!/usr/bin/env python3
"""
Network utilities and connection pooling for Super-Alita.
Provides shared HTTP clients, connection pools, and adaptive concurrency.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp

from .metrics import MetricsCollector, NoOpMetricsCollector


@dataclass
class PoolConfig:
    """Configuration for connection pools"""

    max_connections: int = 100
    max_connections_per_host: int = 30
    timeout_seconds: float = 30.0
    keepalive_timeout: float = 30.0
    ttl_dns_cache: int = 300
    limit_per_host: int = 30


class AdaptiveConcurrencyGate:
    """
    Adaptive concurrency gate that adjusts limits based on performance.
    Implements TCP Vegas-style additive increase / multiplicative decrease.
    """

    def __init__(
        self,
        initial_limit: int = 10,
        min_limit: int = 1,
        max_limit: int = 1000,
        rtt_threshold: float = 0.1,  # 100ms
        metrics: MetricsCollector | None = None,
    ):
        self.initial_limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.rtt_threshold = rtt_threshold
        self.metrics = metrics or NoOpMetricsCollector()

        self._current_limit = initial_limit
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(initial_limit)

        # Performance tracking
        self._rtt_samples: list[float] = []
        self._sample_window = 100

    async def _update_limit(self, rtt: float) -> None:
        """Update concurrency limit based on RTT"""
        async with self._lock:
            self._rtt_samples.append(rtt)
            if len(self._rtt_samples) > self._sample_window:
                self._rtt_samples.pop(0)

            if len(self._rtt_samples) < 10:
                return  # Need more samples

            avg_rtt = sum(self._rtt_samples) / len(self._rtt_samples)

            # Vegas-style adjustment
            if (
                avg_rtt < self.rtt_threshold
                and self._in_flight >= self._current_limit * 0.8
            ):
                # Good performance and high utilization -> increase
                new_limit = min(self._current_limit + 1, self.max_limit)
            elif avg_rtt > self.rtt_threshold * 2:
                # Poor performance -> decrease aggressively
                new_limit = max(int(self._current_limit * 0.8), self.min_limit)
            else:
                new_limit = self._current_limit

            if new_limit != self._current_limit:
                self._current_limit = new_limit
                # Update semaphore
                if new_limit > self._semaphore._value:
                    # Increase permits
                    for _ in range(new_limit - self._semaphore._value):
                        self._semaphore.release()

                self.metrics.gauge("concurrency_limit").set(self._current_limit)

    @asynccontextmanager
    async def acquire(self):
        """Acquire concurrency slot with adaptive limits"""
        start_time = time.time()

        await self._semaphore.acquire()
        async with self._lock:
            self._in_flight += 1

        try:
            yield
        finally:
            async with self._lock:
                self._in_flight -= 1

            # Update performance metrics
            rtt = time.time() - start_time
            await self._update_limit(rtt)

    def get_stats(self) -> dict[str, Any]:
        """Get current gate statistics"""
        return {
            "current_limit": self._current_limit,
            "in_flight": self._in_flight,
            "avg_rtt": sum(self._rtt_samples) / len(self._rtt_samples)
            if self._rtt_samples
            else 0,
            "sample_count": len(self._rtt_samples),
        }


class PooledHTTPClient:
    """HTTP client with connection pooling and adaptive concurrency"""

    def __init__(
        self,
        name: str,
        config: PoolConfig | None = None,
        metrics: MetricsCollector | None = None,
        adaptive_concurrency: bool = True,
    ):
        self.name = name
        self.config = config or PoolConfig()
        self.metrics = metrics or NoOpMetricsCollector()

        self._session: aiohttp.ClientSession | None = None
        self._concurrency_gate: AdaptiveConcurrencyGate | None = None

        if adaptive_concurrency:
            self._concurrency_gate = AdaptiveConcurrencyGate(
                initial_limit=20,
                metrics=metrics,
            )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session is created"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                limit_per_host=self.config.max_connections_per_host,
                ttl_dns_cache=self.config.ttl_dns_cache,
                keepalive_timeout=self.config.keepalive_timeout,
            )

            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )

        return self._session

    async def request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> aiohttp.ClientResponse:
        """Make HTTP request with pooling and metrics"""
        start_time = time.time()
        session = await self._ensure_session()

        # Use adaptive concurrency if enabled
        if self._concurrency_gate:
            async with self._concurrency_gate.acquire():
                response = await session.request(method, url, **kwargs)
        else:
            response = await session.request(method, url, **kwargs)

        # Record metrics
        duration = time.time() - start_time
        self.metrics.histogram(
            "http_request_duration_seconds",
            {
                "client_name": self.name,
                "method": method,
                "status": str(response.status),
            },
        ).observe(duration)

        self.metrics.counter(
            "http_requests_total",
            {
                "client_name": self.name,
                "method": method,
                "status": str(response.status),
            },
        ).inc()

        return response

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """GET request"""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """POST request"""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """PUT request"""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """DELETE request"""
        return await self.request("DELETE", url, **kwargs)

    async def close(self) -> None:
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics"""
        stats = {
            "name": self.name,
            "session_closed": self._session is None
            or (self._session and self._session.closed),
        }

        if self._concurrency_gate:
            stats["concurrency"] = self._concurrency_gate.get_stats()

        return stats


class HTTPClientManager:
    """Manages named HTTP clients with shared pools"""

    def __init__(self, metrics: MetricsCollector | None = None):
        self.metrics = metrics or NoOpMetricsCollector()
        self._clients: dict[str, PooledHTTPClient] = {}

    def get_client(
        self,
        name: str,
        config: PoolConfig | None = None,
        adaptive_concurrency: bool = True,
    ) -> PooledHTTPClient:
        """Get or create a named HTTP client"""
        if name not in self._clients:
            self._clients[name] = PooledHTTPClient(
                name=name,
                config=config,
                metrics=self.metrics,
                adaptive_concurrency=adaptive_concurrency,
            )
        return self._clients[name]

    async def close_all(self) -> None:
        """Close all HTTP clients"""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

    def get_global_stats(self) -> dict[str, Any]:
        """Get statistics for all clients"""
        return {name: client.get_stats() for name, client in self._clients.items()}


class CircuitBreakerHTTPClient:
    """HTTP client with circuit breaker integration"""

    def __init__(
        self,
        client: PooledHTTPClient,
        circuit_breaker,  # Import would be circular, so using duck typing
    ):
        self.client = client
        self.circuit_breaker = circuit_breaker

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make request through circuit breaker"""

        async def make_request():
            return await self.client.request(method, url, **kwargs)

        return await self.circuit_breaker.execute_with_breaker(make_request)

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """GET request through circuit breaker"""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """POST request through circuit breaker"""
        return await self.request("POST", url, **kwargs)


# Global HTTP client manager
http_client_manager = HTTPClientManager()


def get_http_client(
    name: str = "default",
    config: PoolConfig | None = None,
    adaptive_concurrency: bool = True,
) -> PooledHTTPClient:
    """Get or create a named HTTP client"""
    return http_client_manager.get_client(
        name=name,
        config=config,
        adaptive_concurrency=adaptive_concurrency,
    )
