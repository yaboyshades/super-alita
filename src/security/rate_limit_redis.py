from __future__ import annotations

import time
from typing import Any


class RedisRateLimiter:
    """Redis-backed sliding window rate limiter using ZSET.

    Key schema: rl:{identifier}
    Stores timestamps (seconds) as scores and members.
    """

    def __init__(self, redis_client: Any):
        self.redis = redis_client

    async def is_allowed(self, identifier: str, limit: int, window: int):
        now = int(time.time())
        key = f"rl:{identifier}"
        min_score = now - window
        # Use pipeline to approximate atomic window maintenance
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, min_score)
        pipe.zcard(key)
        pipe.expire(key, window)
        res = await pipe.execute()
        count = int(res[1]) if res and len(res) >= 2 else 0
        allowed = count < limit
        if allowed:
            # Add current timestamp; keep small member uniqueness by appending count
            member = f"{now}-{count}"
            await self.redis.zadd(key, {member: now})
            await self.redis.expire(key, window)
            remaining = max(0, limit - (count + 1))
        else:
            remaining = 0
        reset_in = max(1, window)
        return allowed, {"remaining": remaining, "reset_in": reset_in}

