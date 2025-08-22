#!/usr/bin/env python3
"""
Local caching layer for Super-Alita performance optimization.
Provides LRU cache with TTL, size limits, and metrics integration.
"""

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .metrics import MetricsCollector, NoOpMetricsCollector


@dataclass
class CacheEntry:
    """Single cache entry with TTL and access tracking"""

    value: Any
    expires_at: float
    access_count: int = 0
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access tracking"""
        self.access_count += 1


class LocalCache:
    """
    High-performance local cache with LRU eviction and TTL.
    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        default_ttl: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,  # 1 minute
        metrics: MetricsCollector | None = None,
    ):
        self.name = name
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.metrics = metrics or NoOpMetricsCollector()

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Start cleanup task
        self._start_cleanup()

    def _start_cleanup(self) -> None:
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Background task to clean expired entries"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue cleanup
                print(f"Cache cleanup error for {self.name}: {e}")

    async def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if entry.expires_at <= current_time
            ]

            for key in expired_keys:
                del self._cache[key]
                self._evictions += 1

            # Update metrics
            self._update_metrics()

    def _update_metrics(self) -> None:
        """Update cache metrics"""
        total_ops = self._hits + self._misses
        hit_ratio = self._hits / total_ops if total_ops > 0 else 0.0

        labels = {"cache_name": self.name}
        self.metrics.gauge("cache_hit_ratio", labels).set(hit_ratio)
        self.metrics.gauge("cache_size_items", labels).set(len(self._cache))

        # Estimate size in bytes (rough approximation)
        estimated_bytes = len(self._cache) * 100  # Rough estimate
        self.metrics.gauge("cache_size_bytes", labels).set(estimated_bytes)

    async def get(self, key: str) -> Any | None:
        """Get value from cache"""
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                self.metrics.counter(
                    "cache_operations_total",
                    {"cache_name": self.name, "operation": "get", "result": "miss"},
                ).inc()
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                self.metrics.counter(
                    "cache_operations_total",
                    {"cache_name": self.name, "operation": "get", "result": "expired"},
                ).inc()
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1

            self.metrics.counter(
                "cache_operations_total",
                {"cache_name": self.name, "operation": "get", "result": "hit"},
            ).inc()

            self._update_metrics()
            return entry.value

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in cache"""
        if ttl is None:
            ttl = self.default_ttl

        expires_at = time.time() + ttl
        entry = CacheEntry(value=value, expires_at=expires_at)

        async with self._lock:
            # Check if we need to evict
            if key not in self._cache and len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key, _ = self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[key] = entry
            self._cache.move_to_end(key)  # Mark as most recently used

            self.metrics.counter(
                "cache_operations_total",
                {"cache_name": self.name, "operation": "set", "result": "success"},
            ).inc()

            self._update_metrics()

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.metrics.counter(
                    "cache_operations_total",
                    {
                        "cache_name": self.name,
                        "operation": "delete",
                        "result": "success",
                    },
                ).inc()
                self._update_metrics()
                return True

            self.metrics.counter(
                "cache_operations_total",
                {"cache_name": self.name, "operation": "delete", "result": "miss"},
            ).inc()
            return False

    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._update_metrics()

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        result = await self.get(key)
        return result is not None

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_ops = self._hits + self._misses
        hit_ratio = self._hits / total_ops if total_ops > 0 else 0.0

        return {
            "name": self.name,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_ratio": hit_ratio,
            "total_operations": total_ops,
        }

    async def shutdown(self) -> None:
        """Shutdown cache and cleanup resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        await self.clear()


class CacheManager:
    """Manages multiple named caches"""

    def __init__(self, metrics: MetricsCollector | None = None):
        self.metrics = metrics or NoOpMetricsCollector()
        self._caches: dict[str, LocalCache] = {}

    def get_cache(
        self,
        name: str,
        max_size: int = 1000,
        default_ttl: float = 300.0,
        cleanup_interval: float = 60.0,
    ) -> LocalCache:
        """Get or create a named cache"""
        if name not in self._caches:
            self._caches[name] = LocalCache(
                name=name,
                max_size=max_size,
                default_ttl=default_ttl,
                cleanup_interval=cleanup_interval,
                metrics=self.metrics,
            )
        return self._caches[name]

    async def clear_all(self) -> None:
        """Clear all caches"""
        for cache in self._caches.values():
            await cache.clear()

    async def shutdown(self) -> None:
        """Shutdown all caches"""
        for cache in self._caches.values():
            await cache.shutdown()
        self._caches.clear()

    def get_global_stats(self) -> dict[str, Any]:
        """Get statistics for all caches"""
        return {name: cache.get_stats() for name, cache in self._caches.items()}


# Global cache manager instance
cache_manager = CacheManager()


def cached(
    cache_name: str = "default",
    ttl: float = 300.0,
    key_func: Callable | None = None,
):
    """
    Decorator for caching function results.

    Args:
        cache_name: Name of cache to use
        ttl: Time to live in seconds
        key_func: Function to generate cache key from args/kwargs
    """

    def decorator(func):
        cache = cache_manager.get_cache(cache_name)

        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = (
                    f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
                )

            # Try cache first
            result = await cache.get(cache_key)
            if result is not None:
                return result

            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Cache result
            await cache.set(cache_key, result, ttl)
            return result

        def sync_wrapper(*args, **kwargs):
            return asyncio.create_task(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
