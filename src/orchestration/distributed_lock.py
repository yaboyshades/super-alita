"""Distributed Lock Manager using Redis.

Redis-based distributed locks with:
- Atomic lock/unlock operations using Lua scripts
- TTL-based expiration
- Token-based ownership verification
- Context manager support for easy use
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

try:
    # Use redis.asyncio for Python 3.11+ compatibility
    import redis.asyncio as aioredis
except ImportError:
    try:
        import aioredis  # Fallback to legacy aioredis
    except ImportError:
        aioredis = None  # type: ignore[assignment]


class LockTimeout(Exception):
    """Raised when lock acquisition times out."""

    pass


class DistributedLock:
    """Redis-based distributed lock with atomic operations.

    Example:
        async with DistributedLock(redis, "my-resource", ttl=30):
            # Critical section - lock held
            await do_work()
        # Lock automatically released
    """

    # Lua script for atomic unlock (only unlock if token matches)
    UNLOCK_SCRIPT = """
    if redis.call('GET', KEYS[1]) == ARGV[1] then
        return redis.call('DEL', KEYS[1])
    else
        return 0
    end
    """

    def __init__(
        self,
        redis: Any,
        key: str,
        ttl: int = 30,
        timeout: float = 10.0,
        retry_delay: float = 0.1,
    ):
        """Initialize distributed lock.

        Args:
            redis: aioredis Redis connection
            key: Lock key (will be prefixed with "lock:")
            ttl: Lock TTL in seconds (default: 30)
            timeout: Max seconds to wait for lock acquisition (default: 10)
            retry_delay: Seconds to wait between retry attempts (default: 0.1)
        """
        if aioredis is None:
            raise RuntimeError("aioredis not installed")

        self.redis = redis
        self.key = f"lock:{key}"
        self.ttl = ttl
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.token = str(uuid4())
        self._acquired = False

    async def acquire(self) -> bool:
        """Acquire the lock.

        Returns:
            True if lock acquired, False if timeout

        Raises:
            LockTimeout: If lock cannot be acquired within timeout
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            # Try to set lock with NX (only if not exists) and EX (expiration)
            acquired = await self.redis.set(
                self.key, self.token, nx=True, ex=self.ttl
            )

            if acquired:
                self._acquired = True
                return True

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= self.timeout:
                raise LockTimeout(
                    f"Could not acquire lock {self.key} within {self.timeout}s"
                )

            # Wait before retry
            await asyncio.sleep(self.retry_delay)

    async def release(self) -> bool:
        """Release the lock using atomic Lua script.

        Returns:
            True if lock was released, False if lock not held or expired
        """
        if not self._acquired:
            return False

        # Execute Lua script for atomic unlock
        result = await self.redis.eval(
            self.UNLOCK_SCRIPT, keys=[self.key], args=[self.token]
        )

        self._acquired = False
        return bool(result)

    async def extend(self, additional_ttl: int | None = None) -> bool:
        """Extend the lock TTL.

        Args:
            additional_ttl: Additional seconds to extend (default: original TTL)

        Returns:
            True if extended, False if lock not held
        """
        if not self._acquired:
            return False

        ttl = additional_ttl or self.ttl

        # Only extend if token matches (atomic check-and-set)
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('EXPIRE', KEYS[1], ARGV[2])
        else
            return 0
        end
        """

        result = await self.redis.eval(
            script, keys=[self.key], args=[self.token, ttl]
        )

        return bool(result)

    async def __aenter__(self) -> DistributedLock:
        """Context manager entry - acquire lock."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - release lock."""
        await self.release()


class LockManager:
    """Manages multiple distributed locks.

    Provides high-level lock management with:
    - Named lock pools
    - Automatic cleanup
    - Lock statistics
    """

    def __init__(self, redis: Any):
        """Initialize lock manager.

        Args:
            redis: aioredis Redis connection
        """
        self.redis = redis
        self.active_locks: dict[str, DistributedLock] = {}
        self.lock_stats: dict[str, dict[str, Any]] = {}

    async def acquire_lock(
        self,
        key: str,
        ttl: int = 30,
        timeout: float = 10.0,
    ) -> DistributedLock:
        """Acquire a named lock.

        Args:
            key: Lock key
            ttl: Lock TTL in seconds
            timeout: Acquisition timeout in seconds

        Returns:
            Acquired DistributedLock instance

        Raises:
            LockTimeout: If lock cannot be acquired
        """
        lock = DistributedLock(self.redis, key, ttl=ttl, timeout=timeout)
        await lock.acquire()

        self.active_locks[key] = lock

        # Update stats
        if key not in self.lock_stats:
            self.lock_stats[key] = {"acquisitions": 0, "timeouts": 0}
        self.lock_stats[key]["acquisitions"] += 1

        return lock

    async def release_lock(self, key: str) -> bool:
        """Release a named lock.

        Args:
            key: Lock key

        Returns:
            True if released, False if lock not held
        """
        lock = self.active_locks.get(key)
        if not lock:
            return False

        result = await lock.release()
        if result:
            del self.active_locks[key]

        return result

    async def release_all(self) -> int:
        """Release all active locks.

        Returns:
            Number of locks released
        """
        count = 0
        for key in list(self.active_locks.keys()):
            if await self.release_lock(key):
                count += 1
        return count

    def get_active_locks(self) -> list[str]:
        """Get list of currently held lock keys.

        Returns:
            List of lock keys
        """
        return list(self.active_locks.keys())

    def get_lock_stats(self) -> dict[str, dict[str, Any]]:
        """Get lock statistics.

        Returns:
            Dict of lock key -> stats
        """
        return self.lock_stats.copy()


# Example usage patterns


async def safe_modify_resource(redis: Any, resource_id: str) -> None:
    """Example: Safely modify a shared resource."""
    async with DistributedLock(redis, f"resource:{resource_id}", ttl=30):
        # Critical section - only one process can be here
        # Read resource
        data = await redis.get(f"data:{resource_id}")

        # Modify
        if data:
            modified = data + "_modified"
            await redis.set(f"data:{resource_id}", modified)


async def coordinated_operation(
    manager: LockManager, operation_id: str
) -> None:
    """Example: Coordinate complex multi-step operation."""
    # Acquire multiple locks in order to prevent deadlock
    locks_needed = [
        f"op:{operation_id}",
        f"state:{operation_id}",
        f"config:{operation_id}",
    ]

    acquired = []
    try:
        for lock_key in locks_needed:
            await manager.acquire_lock(lock_key, ttl=60)
            acquired.append(lock_key)

        # All locks acquired - perform operation
        # ... complex operation ...
        pass

    finally:
        # Release in reverse order
        for lock_key in reversed(acquired):
            await manager.release_lock(lock_key)
