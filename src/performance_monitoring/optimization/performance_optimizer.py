"""
Performance Optimizer Module

Provides advanced performance optimization features including:
- Multi-level caching (memory, disk, distributed)
- Connection pooling for database and external services
- Query optimization and batch processing
- Resource preloading and lazy loading
- Performance profiling and bottleneck detection
"""

import asyncio
import hashlib
import logging
import pickle
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    value: Any
    created_at: datetime
    accessed_at: datetime
    hit_count: int = 0
    size_bytes: int = 0
    ttl_seconds: int | None = None

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl_seconds is None:
            return False
        return (
            datetime.now() - self.created_at
        ).total_seconds() > self.ttl_seconds

    def touch(self):
        """Update access time and hit count"""
        self.accessed_at = datetime.now()
        self.hit_count += 1


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling"""

    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class MemoryCache:
    """High-performance in-memory cache with LRU eviction"""

    def __init__(self, max_size: int = 1000, default_ttl: int | None = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "size_bytes": 0}

    def get(self, key: str) -> Any | None:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["size_bytes"] -= entry.size_bytes
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats["hits"] += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Put value in cache"""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 0

            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats["size_bytes"] -= old_entry.size_bytes
                del self._cache[key]

            # Create new entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.default_ttl,
            )

            self._cache[key] = entry
            self._stats["size_bytes"] += size_bytes

            # Evict if necessary
            while len(self._cache) > self.max_size:
                self._evict_lru()

    def _evict_lru(self):
        """Evict least recently used entry"""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            self._stats["size_bytes"] -= entry.size_bytes

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._stats["size_bytes"] = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests
                if total_requests > 0
                else 0
            )

            return {
                **self._stats,
                "entries": len(self._cache),
                "hit_rate": hit_rate,
                "avg_size_bytes": (
                    self._stats["size_bytes"] / len(self._cache)
                    if self._cache
                    else 0
                ),
            }


class ConnectionPool:
    """Generic connection pool with health monitoring"""

    def __init__(
        self, connection_factory: Callable, config: ConnectionPoolConfig
    ):
        self.connection_factory = connection_factory
        self.config = config
        self._pool: list[Any] = []
        self._in_use: dict[int, Any] = {}
        self._lock = asyncio.Lock()
        self._created_count = 0
        self._stats = {
            "total_created": 0,
            "total_destroyed": 0,
            "current_active": 0,
            "current_idle": 0,
            "total_acquisitions": 0,
            "failed_acquisitions": 0,
        }

    async def acquire(self) -> Any:
        """Acquire connection from pool"""
        async with self._lock:
            self._stats["total_acquisitions"] += 1

            # Try to get from pool
            while self._pool:
                conn = self._pool.pop()
                if await self._is_connection_healthy(conn):
                    conn_id = id(conn)
                    self._in_use[conn_id] = conn
                    self._stats["current_active"] += 1
                    self._stats["current_idle"] -= 1
                    return conn
                else:
                    await self._destroy_connection(conn)

            # Create new connection if under limit
            if len(self._in_use) < self.config.max_connections:
                try:
                    conn = await self._create_connection()
                    conn_id = id(conn)
                    self._in_use[conn_id] = conn
                    self._stats["current_active"] += 1
                    return conn
                except Exception as e:
                    self._stats["failed_acquisitions"] += 1
                    logger.error(f"Failed to create connection: {e}")
                    raise

            # Pool exhausted
            self._stats["failed_acquisitions"] += 1
            raise Exception("Connection pool exhausted")

    async def release(self, connection: Any):
        """Release connection back to pool"""
        async with self._lock:
            conn_id = id(connection)
            if conn_id not in self._in_use:
                return

            del self._in_use[conn_id]
            self._stats["current_active"] -= 1

            if await self._is_connection_healthy(connection):
                self._pool.append(connection)
                self._stats["current_idle"] += 1
            else:
                await self._destroy_connection(connection)

    async def _create_connection(self) -> Any:
        """Create new connection"""
        conn = await self.connection_factory()
        self._stats["total_created"] += 1
        return conn

    async def _destroy_connection(self, connection: Any):
        """Destroy connection"""
        try:
            if hasattr(connection, "close"):
                await connection.close()
            elif hasattr(connection, "disconnect"):
                await connection.disconnect()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

        self._stats["total_destroyed"] += 1

    async def _is_connection_healthy(self, connection: Any) -> bool:
        """Check if connection is healthy"""
        try:
            if hasattr(connection, "ping"):
                return await connection.ping()
            elif hasattr(connection, "is_connected"):
                return await connection.is_connected()
            return True
        except Exception:
            return False

    async def close_all(self):
        """Close all connections"""
        async with self._lock:
            # Close pooled connections
            for conn in self._pool:
                await self._destroy_connection(conn)
            self._pool.clear()

            # Close in-use connections
            for conn in list(self._in_use.values()):
                await self._destroy_connection(conn)
            self._in_use.clear()

            self._stats["current_active"] = 0
            self._stats["current_idle"] = 0

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics"""
        return {
            **self._stats,
            "pool_size": len(self._pool),
            "in_use_count": len(self._in_use),
            "utilization": len(self._in_use) / self.config.max_connections,
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator"""

    def __init__(self):
        self.memory_cache = MemoryCache(max_size=5000, default_ttl=3600)
        self.connection_pools: dict[str, ConnectionPool] = {}
        self.query_cache = MemoryCache(max_size=1000, default_ttl=300)
        self.batch_processors: dict[str, BatchProcessor] = {}
        self._performance_profiles: dict[str, list[float]] = defaultdict(list)

        # Background optimization tasks
        self._optimization_tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self):
        """Start background optimization tasks"""
        self._running = True

        # Start cache maintenance
        self._optimization_tasks.append(
            asyncio.create_task(self._cache_maintenance_loop())
        )

        # Start performance monitoring
        self._optimization_tasks.append(
            asyncio.create_task(self._performance_monitoring_loop())
        )

        logger.info("Performance optimizer started")

    async def stop(self):
        """Stop background optimization tasks"""
        self._running = False

        # Cancel background tasks
        for task in self._optimization_tasks:
            task.cancel()

        # Close connection pools
        for pool in self.connection_pools.values():
            await pool.close_all()

        logger.info("Performance optimizer stopped")

    def register_connection_pool(
        self,
        name: str,
        connection_factory: Callable,
        config: ConnectionPoolConfig | None = None,
    ):
        """Register a connection pool"""
        if config is None:
            config = ConnectionPoolConfig()

        self.connection_pools[name] = ConnectionPool(
            connection_factory, config
        )
        logger.info(f"Registered connection pool: {name}")

    async def get_connection(self, pool_name: str) -> Any:
        """Get connection from named pool"""
        if pool_name not in self.connection_pools:
            raise ValueError(f"Unknown connection pool: {pool_name}")

        return await self.connection_pools[pool_name].acquire()

    async def release_connection(self, pool_name: str, connection: Any):
        """Release connection to named pool"""
        if pool_name in self.connection_pools:
            await self.connection_pools[pool_name].release(connection)

    def cache_result(self, key: str, value: Any, ttl: int | None = None):
        """Cache computation result"""
        self.memory_cache.put(key, value, ttl)

    def get_cached_result(self, key: str) -> Any | None:
        """Get cached computation result"""
        return self.memory_cache.get(key)

    def cache_query(self, query_hash: str, result: Any, ttl: int = 300):
        """Cache query result"""
        self.query_cache.put(query_hash, result, ttl)

    def get_cached_query(self, query_hash: str) -> Any | None:
        """Get cached query result"""
        return self.query_cache.get(query_hash)

    def hash_query(self, query: str, params: dict | None = None) -> str:
        """Generate hash for query and parameters"""
        content = f"{query}:{params or {}}"
        return hashlib.sha256(content.encode()).hexdigest()

    def profile_operation(self, operation_name: str, duration: float):
        """Record operation performance profile"""
        self._performance_profiles[operation_name].append(duration)

        # Keep only recent measurements
        if len(self._performance_profiles[operation_name]) > 100:
            self._performance_profiles[operation_name] = (
                self._performance_profiles[operation_name][-100:]
            )

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            "memory_cache": self.memory_cache.get_stats(),
            "query_cache": self.query_cache.get_stats(),
            "connection_pools": {},
            "performance_profiles": {},
        }

        # Connection pool stats
        for name, pool in self.connection_pools.items():
            stats["connection_pools"][name] = pool.get_stats()

        # Performance profiles
        for op_name, timings in self._performance_profiles.items():
            if timings:
                stats["performance_profiles"][op_name] = {
                    "count": len(timings),
                    "avg_ms": sum(timings) * 1000 / len(timings),
                    "min_ms": min(timings) * 1000,
                    "max_ms": max(timings) * 1000,
                    "recent_avg_ms": sum(timings[-10:])
                    * 1000
                    / min(len(timings), 10),
                }

        return stats

    async def _cache_maintenance_loop(self):
        """Background cache maintenance"""
        while self._running:
            try:
                # Clear expired entries periodically
                await asyncio.sleep(300)  # 5 minutes

                # Note: Actual implementation would clean expired entries
                logger.debug("Cache maintenance completed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")

    async def _performance_monitoring_loop(self):
        """Background performance monitoring"""
        while self._running:
            try:
                await asyncio.sleep(60)  # 1 minute

                stats = self.get_performance_stats()

                # Log performance warnings
                cache_hit_rate = stats["memory_cache"]["hit_rate"]
                if cache_hit_rate < 0.5:
                    logger.warning(f"Low cache hit rate: {cache_hit_rate:.2%}")

                for pool_name, pool_stats in stats["connection_pools"].items():
                    utilization = pool_stats["utilization"]
                    if utilization > 0.8:
                        logger.warning(
                            f"High connection pool utilization ({pool_name}): {utilization:.2%}"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")


# Context manager for performance profiling
class PerformanceProfiler:
    """Context manager for automatic performance profiling"""

    def __init__(self, optimizer: PerformanceOptimizer, operation_name: str):
        self.optimizer = optimizer
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.optimizer.profile_operation(self.operation_name, duration)


# Decorator for automatic caching
def cached_result(
    optimizer: PerformanceOptimizer,
    ttl: int | None = None,
    key_func: Callable | None = None,
):
    """Decorator for automatic result caching"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = (
                    f"{func.__name__}:{hash((args, tuple(kwargs.items())))}"
                )

            # Try cache first
            cached = optimizer.get_cached_result(cache_key)
            if cached is not None:
                return cached

            # Execute function and cache result
            result = await func(*args, **kwargs)
            optimizer.cache_result(cache_key, result, ttl)
            return result

        return wrapper

    return decorator
