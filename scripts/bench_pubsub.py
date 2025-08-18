"""Publish N messages to measure Redis throughput."""
import os
import time

import redis

try:
    import orjson as jsonlib
except Exception:  # pragma: no cover
    import json as jsonlib

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
N = int(os.getenv("N", "2000"))


def main() -> None:
    r = redis.from_url(REDIS_URL, decode_responses=False)
    ps = r.pubsub()
    ps.subscribe("bench")

    payload = {"type": "bench", "i": 0}
    msg = jsonlib.dumps(payload)

    for _ in range(200):
        r.publish("bench", msg)

    start = time.perf_counter()
    for i in range(N):
        payload["i"] = i
        r.publish("bench", jsonlib.dumps(payload))
    elapsed = time.perf_counter() - start
    print(f"Published {N} msgs in {elapsed:.3f}s → {N/elapsed:.1f} msg/s")


if __name__ == "__main__":
    main()
