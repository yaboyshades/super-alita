#!/usr/bin/env python3
"""
DTA 2.0 Cache Management Module

High-performance, production-grade caching system for the Deep Thinking
Architecture. Supports Redis-based distributed caching and in-memory
fallback with TTL, statistics, and health monitoring.
"""

import asyncio
import hashlib
import json
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

# Redis support with fallback
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class CacheBackend(Enum):
    """Supported cache backend types."""

    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


class CacheOperation(Enum):
    """Cache operation types for statistics."""

    GET = "get"
    SET = "set"
    DELETE = "delete"
    CLEAR = "clear"
    EXISTS = "exists"


@dataclass
class CacheStats:
    """Cache statistics and performance metrics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_operations: int = 0
    total_latency_ms: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_gets = self.hits + self.misses
        return self.hits / total_gets if total_gets > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average operation latency."""
        return (
            self.total_latency_ms / self.total_operations
            if self.total_operations > 0
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary format."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "total_operations": self.total_operations,
            "hit_rate": self.hit_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "last_reset": self.last_reset.isoformat(),
        }


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""

    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int | None = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl_seconds is None:
            return False

        return datetime.now(UTC) > (
            self.created_at + timedelta(seconds=self.ttl_seconds)
        )

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now(UTC) - self.created_at).total_seconds()

    def access(self):
        """Record access to this cache entry."""
        self.access_count += 1
        self.last_accessed = datetime.now(UTC)


class InMemoryCache:
    """
    High-performance in-memory cache implementation.

    Features:
    - TTL (Time-To-Live) support
    - LRU eviction policy
    - Statistics tracking
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 10000, default_ttl: int | None = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU tracking
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def _record_operation(self, operation: CacheOperation, latency_ms: float):
        """Record operation statistics."""
        with self._lock:
            self._stats.total_operations += 1
            self._stats.total_latency_ms += latency_ms

            if operation == CacheOperation.GET:
                # Hit/miss will be recorded separately
                pass
            elif operation == CacheOperation.SET:
                self._stats.sets += 1
            elif operation == CacheOperation.DELETE:
                self._stats.deletes += 1

    def _cleanup_expired(self):
        """Remove expired entries."""
        datetime.now(UTC)
        expired_keys = []

        for key, entry in self._cache.items():
            if entry.is_expired:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._stats.evictions += 1

    def _evict_lru(self):
        """Evict least recently used entry if cache is full."""
        if len(self._cache) >= self.max_size and self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats.evictions += 1

    def _update_access_order(self, key: str):
        """Update LRU access order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        start_time = time.time()

        try:
            with self._lock:
                # Cleanup expired entries periodically
                if len(self._cache) % 100 == 0:
                    self._cleanup_expired()

                entry = self._cache.get(key)

                if entry is None:
                    self._stats.misses += 1
                    return None

                if entry.is_expired:
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._stats.misses += 1
                    self._stats.evictions += 1
                    return None

                # Record hit and update access
                entry.access()
                self._update_access_order(key)
                self._stats.hits += 1

                return entry.value

        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(CacheOperation.GET, latency_ms)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache."""
        start_time = time.time()

        try:
            with self._lock:
                # Use default TTL if not specified
                if ttl is None:
                    ttl = self.default_ttl

                # Evict LRU if needed
                self._evict_lru()

                # Create cache entry
                entry = CacheEntry(
                    key=key, value=value, created_at=datetime.now(UTC), ttl_seconds=ttl
                )

                self._cache[key] = entry
                self._update_access_order(key)

                return True

        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(CacheOperation.SET, latency_ms)

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        start_time = time.time()

        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
                    return True
                return False

        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(CacheOperation.DELETE, latency_ms)

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired

    async def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            return True

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                sets=self._stats.sets,
                deletes=self._stats.deletes,
                evictions=self._stats.evictions,
                total_operations=self._stats.total_operations,
                total_latency_ms=self._stats.total_latency_ms,
                last_reset=self._stats.last_reset,
            )

    async def get_info(self) -> dict[str, Any]:
        """Get detailed cache information."""
        with self._lock:
            stats = await self.get_stats()
            return {
                "backend": "memory",
                "max_size": self.max_size,
                "current_size": len(self._cache),
                "default_ttl": self.default_ttl,
                "stats": stats.to_dict(),
            }


class RedisCache:
    """
    Redis-based distributed cache implementation.

    Features:
    - Distributed caching across multiple instances
    - Automatic serialization/deserialization
    - TTL support via Redis
    - Connection pooling and resilience
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "dta:",
        default_ttl: int | None = 3600,
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self._redis: redis.Redis | None = None
        self._stats = CacheStats()
        self._lock = threading.RLock()

    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection, creating if needed."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def _make_key(self, key: str) -> str:
        """Create prefixed Redis key."""
        return f"{self.key_prefix}{key}"

    def _serialize_value(self, value: Any) -> str:
        """Serialize value for Redis storage."""
        return json.dumps(value, default=str, separators=(",", ":"))

    def _deserialize_value(self, data: str) -> Any:
        """Deserialize value from Redis."""
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data

    def _record_operation(self, operation: CacheOperation, latency_ms: float):
        """Record operation statistics."""
        with self._lock:
            self._stats.total_operations += 1
            self._stats.total_latency_ms += latency_ms

            if operation == CacheOperation.SET:
                self._stats.sets += 1
            elif operation == CacheOperation.DELETE:
                self._stats.deletes += 1

    async def get(self, key: str) -> Any | None:
        """Get value from Redis cache."""
        start_time = time.time()

        try:
            if not REDIS_AVAILABLE:
                return None

            redis_client = await self._get_redis()
            redis_key = self._make_key(key)

            data = await redis_client.get(redis_key)

            if data is None:
                with self._lock:
                    self._stats.misses += 1
                return None

            with self._lock:
                self._stats.hits += 1

            return self._deserialize_value(data)

        except Exception:
            with self._lock:
                self._stats.misses += 1
            return None

        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(CacheOperation.GET, latency_ms)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in Redis cache."""
        start_time = time.time()

        try:
            if not REDIS_AVAILABLE:
                return False

            redis_client = await self._get_redis()
            redis_key = self._make_key(key)
            serialized_value = self._serialize_value(value)

            if ttl is None:
                ttl = self.default_ttl

            if ttl:
                await redis_client.setex(redis_key, ttl, serialized_value)
            else:
                await redis_client.set(redis_key, serialized_value)

            return True

        except Exception:
            return False

        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(CacheOperation.SET, latency_ms)

    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        start_time = time.time()

        try:
            if not REDIS_AVAILABLE:
                return False

            redis_client = await self._get_redis()
            redis_key = self._make_key(key)

            result = await redis_client.delete(redis_key)
            return result > 0

        except Exception:
            return False

        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(CacheOperation.DELETE, latency_ms)

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            if not REDIS_AVAILABLE:
                return False

            redis_client = await self._get_redis()
            redis_key = self._make_key(key)

            result = await redis_client.exists(redis_key)
            return result > 0

        except Exception:
            return False

    async def clear(self) -> bool:
        """Clear all cache entries with the prefix."""
        try:
            if not REDIS_AVAILABLE:
                return False

            redis_client = await self._get_redis()
            pattern = f"{self.key_prefix}*"

            # Use scan to avoid blocking Redis
            keys = []
            async for key in redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await redis_client.delete(*keys)

            return True

        except Exception:
            return False

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                sets=self._stats.sets,
                deletes=self._stats.deletes,
                evictions=self._stats.evictions,
                total_operations=self._stats.total_operations,
                total_latency_ms=self._stats.total_latency_ms,
                last_reset=self._stats.last_reset,
            )

    async def get_info(self) -> dict[str, Any]:
        """Get detailed cache information."""
        stats = await self.get_stats()
        info = {
            "backend": "redis",
            "redis_url": self.redis_url,
            "key_prefix": self.key_prefix,
            "default_ttl": self.default_ttl,
            "redis_available": REDIS_AVAILABLE,
            "stats": stats.to_dict(),
        }

        # Try to get Redis info
        try:
            if REDIS_AVAILABLE and self._redis:
                redis_info = await self._redis.info()
                info["redis_info"] = {
                    "used_memory": redis_info.get("used_memory"),
                    "connected_clients": redis_info.get("connected_clients"),
                    "uptime_in_seconds": redis_info.get("uptime_in_seconds"),
                }
        except Exception:
            pass

        return info

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()


class DTACache:
    """
    High-level cache abstraction for DTA 2.0.

    Provides unified interface for multiple cache backends with
    intelligent fallback, health monitoring, and performance optimization.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.backend_type = CacheBackend(self.config.get("backend", "memory"))
        self.enabled = self.config.get("enabled", True)

        # Initialize backend
        if self.backend_type == CacheBackend.REDIS:
            self.primary_cache = RedisCache(
                redis_url=self.config.get("redis_url", "redis://localhost:6379"),
                key_prefix=self.config.get("key_prefix", "dta:"),
                default_ttl=self.config.get("default_ttl", 3600),
            )
            # Always have memory fallback
            self.fallback_cache = InMemoryCache(
                max_size=self.config.get("fallback_max_size", 1000),
                default_ttl=self.config.get("default_ttl", 3600),
            )
        else:
            self.primary_cache = InMemoryCache(
                max_size=self.config.get("max_size", 10000),
                default_ttl=self.config.get("default_ttl", 3600),
            )
            self.fallback_cache = None

        # Cache key generation settings
        self.hash_long_keys = self.config.get("hash_long_keys", True)
        self.max_key_length = self.config.get("max_key_length", 250)

        # Health monitoring
        self._health_status = "healthy"
        self._last_health_check = datetime.now(UTC)

    def _generate_cache_key(
        self, namespace: str, key_data: str | dict[str, Any]
    ) -> str:
        """Generate optimized cache key."""
        if isinstance(key_data, dict):
            # Create deterministic key from dict
            key_str = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
        else:
            key_str = str(key_data)

        full_key = f"{namespace}:{key_str}"

        # Hash long keys to avoid Redis key length limits
        if self.hash_long_keys and len(full_key) > self.max_key_length:
            key_hash = hashlib.sha256(full_key.encode()).hexdigest()[:16]
            return f"{namespace}:h:{key_hash}"

        return full_key

    async def get(self, namespace: str, key_data: str | dict[str, Any]) -> Any | None:
        """Get value from cache with fallback support."""
        if not self.enabled:
            return None

        cache_key = self._generate_cache_key(namespace, key_data)

        # Try primary cache first
        try:
            value = await self.primary_cache.get(cache_key)
            if value is not None:
                return value
        except Exception:
            # Primary cache failed, will try fallback
            pass

        # Try fallback cache if available
        if self.fallback_cache:
            try:
                return await self.fallback_cache.get(cache_key)
            except Exception:
                pass

        return None

    async def set(
        self,
        namespace: str,
        key_data: str | dict[str, Any],
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set value in cache with fallback support."""
        if not self.enabled:
            return False

        cache_key = self._generate_cache_key(namespace, key_data)
        success = False

        # Try primary cache
        try:
            if await self.primary_cache.set(cache_key, value, ttl):
                success = True
        except Exception:
            pass

        # Also set in fallback cache for redundancy
        if self.fallback_cache:
            try:
                await self.fallback_cache.set(cache_key, value, ttl)
            except Exception:
                pass

        return success

    async def delete(self, namespace: str, key_data: str | dict[str, Any]) -> bool:
        """Delete value from cache."""
        if not self.enabled:
            return False

        cache_key = self._generate_cache_key(namespace, key_data)
        success = False

        # Delete from primary cache
        try:
            if await self.primary_cache.delete(cache_key):
                success = True
        except Exception:
            pass

        # Delete from fallback cache
        if self.fallback_cache:
            try:
                await self.fallback_cache.delete(cache_key)
            except Exception:
                pass

        return success

    async def exists(self, namespace: str, key_data: str | dict[str, Any]) -> bool:
        """Check if key exists in cache."""
        if not self.enabled:
            return False

        cache_key = self._generate_cache_key(namespace, key_data)

        # Check primary cache
        try:
            if await self.primary_cache.exists(cache_key):
                return True
        except Exception:
            pass

        # Check fallback cache
        if self.fallback_cache:
            try:
                return await self.fallback_cache.exists(cache_key)
            except Exception:
                pass

        return False

    async def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in a namespace (best effort)."""
        if not self.enabled:
            return False

        # This is a simplified implementation
        # In production, you might want more sophisticated namespace clearing
        success = False

        try:
            # For now, just clear all caches
            await self.primary_cache.clear()
            success = True
        except Exception:
            pass

        if self.fallback_cache:
            try:
                await self.fallback_cache.clear()
            except Exception:
                pass

        return success

    @asynccontextmanager
    async def cached_operation(
        self,
        namespace: str,
        key_data: str | dict[str, Any],
        ttl: int | None = None,
    ):
        """
        Context manager for cached operations.

        Usage:
            async with cache.cached_operation("responses", {"query": "test"}) as cached:
                if cached.value is not None:
                    return cached.value

                result = await expensive_operation()
                await cached.store(result)
                return result
        """

        class CachedOperationContext:
            def __init__(self, cache_instance, cache_key, ttl):
                self.cache_instance = cache_instance
                self.cache_key = cache_key
                self.ttl = ttl
                self.value = None

            async def store(self, value: Any) -> bool:
                return await self.cache_instance.set(
                    namespace, key_data, value, self.ttl
                )

        context = CachedOperationContext(
            self, self._generate_cache_key(namespace, key_data), ttl
        )
        context.value = await self.get(namespace, key_data)

        yield context

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "enabled": self.enabled,
            "backend_type": self.backend_type.value,
            "health_status": self._health_status,
            "last_health_check": self._last_health_check.isoformat(),
        }

        # Get primary cache stats
        try:
            primary_stats = await self.primary_cache.get_stats()
            stats["primary_cache"] = primary_stats.to_dict()
        except Exception:
            stats["primary_cache"] = {"error": "Failed to get stats"}

        # Get fallback cache stats
        if self.fallback_cache:
            try:
                fallback_stats = await self.fallback_cache.get_stats()
                stats["fallback_cache"] = fallback_stats.to_dict()
            except Exception:
                stats["fallback_cache"] = {"error": "Failed to get stats"}

        return stats

    async def health_check(self) -> dict[str, Any]:
        """Perform cache health check."""
        self._last_health_check = datetime.now(UTC)

        health_info = {
            "timestamp": self._last_health_check.isoformat(),
            "enabled": self.enabled,
            "backend_type": self.backend_type.value,
            "primary_cache_healthy": False,
            "fallback_cache_healthy": False,
            "overall_status": "unhealthy",
        }

        # Test primary cache
        try:
            test_key = "health_check_test"
            test_value = {"timestamp": time.time()}

            # Test set/get/delete cycle
            await self.primary_cache.set(test_key, test_value, ttl=60)
            retrieved = await self.primary_cache.get(test_key)
            await self.primary_cache.delete(test_key)

            if retrieved == test_value:
                health_info["primary_cache_healthy"] = True
        except Exception as e:
            health_info["primary_cache_error"] = str(e)

        # Test fallback cache if available
        if self.fallback_cache:
            try:
                test_key = "health_check_test_fallback"
                test_value = {"timestamp": time.time()}

                await self.fallback_cache.set(test_key, test_value, ttl=60)
                retrieved = await self.fallback_cache.get(test_key)
                await self.fallback_cache.delete(test_key)

                if retrieved == test_value:
                    health_info["fallback_cache_healthy"] = True
            except Exception as e:
                health_info["fallback_cache_error"] = str(e)
        else:
            health_info["fallback_cache_healthy"] = True  # No fallback needed

        # Determine overall status
        if health_info["primary_cache_healthy"]:
            self._health_status = "healthy"
            health_info["overall_status"] = "healthy"
        elif health_info["fallback_cache_healthy"]:
            self._health_status = "degraded"
            health_info["overall_status"] = "degraded"
        else:
            self._health_status = "unhealthy"
            health_info["overall_status"] = "unhealthy"

        return health_info

    async def close(self):
        """Close cache connections."""
        if hasattr(self.primary_cache, "close"):
            await self.primary_cache.close()

        if self.fallback_cache and hasattr(self.fallback_cache, "close"):
            await self.fallback_cache.close()


# Utility functions for easy cache setup
def create_cache(config: dict[str, Any] | None = None) -> DTACache:
    """Create a DTACache instance with default configuration."""
    default_config = {
        "enabled": True,
        "backend": "memory",  # or 'redis'
        "default_ttl": 3600,  # 1 hour
        "max_size": 10000,
        "hash_long_keys": True,
        "max_key_length": 250,
    }

    if config:
        if hasattr(config, "__dict__"):
            # Handle dataclass or object - convert to dict
            if hasattr(config, "to_dict"):
                config_dict = config.to_dict()
            else:
                config_dict = {
                    k: v for k, v in config.__dict__.items() if not k.startswith("_")
                }
            default_config.update(config_dict)
        else:
            # Handle dict
            default_config.update(config)

    return DTACache(default_config)


# Example usage and testing
async def example_cache_usage():
    """Example of how to use DTA cache."""

    # Create cache instance
    cache = create_cache(
        {
            "backend": "memory",  # or 'redis' for distributed caching
            "default_ttl": 300,  # 5 minutes
            "max_size": 5000,
        }
    )

    # Basic operations
    await cache.set("responses", "test_query", {"result": "cached data"})

    cached_result = await cache.get("responses", "test_query")
    print(f"Cached result: {cached_result}")

    # Using the cached operation context manager
    async with cache.cached_operation(
        "expensive_ops", {"param": "value"}, ttl=600
    ) as cached:
        if cached.value is not None:
            print(f"Found cached result: {cached.value}")
            result = cached.value
        else:
            print("Computing expensive operation...")
            # Simulate expensive operation
            await asyncio.sleep(0.1)
            result = {"computed": "expensive result", "timestamp": time.time()}
            await cached.store(result)

    # Health check
    health = await cache.health_check()
    print(f"Cache health: {health}")

    # Statistics
    stats = await cache.get_stats()
    print(f"Cache stats: {stats}")

    # Cleanup
    await cache.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_cache_usage())
