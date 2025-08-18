"""Publish (and optionally round-trip) N messages to measure Redis throughput."""
import os
import time

import redis

try:  # pragma: no cover
    import orjson as jsonlib
except ImportError:  # pragma: no cover
    import json as jsonlib

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
N = int(os.getenv("N", "2000"))
ROUNDTRIP = os.getenv("ROUNDTRIP", "0") == "1"
CHANNEL = os.getenv("CHANNEL", "bench")


def main() -> None:
    r = redis.from_url(REDIS_URL, decode_responses=False)
    ps = r.pubsub(ignore_subscribe_messages=True)
    if ROUNDTRIP:
        ps.subscribe(CHANNEL)

    payload = {"type": "bench", "i": 0}
    msg = jsonlib.dumps(payload)

    # Warm-up
    for _ in range(200):
        r.publish(CHANNEL, msg)

    if not ROUNDTRIP:
        start = time.perf_counter()
        for i in range(N):
            payload["i"] = i
            r.publish(CHANNEL, jsonlib.dumps(payload))
        elapsed = time.perf_counter() - start
        print(f"[publish-only] {N} msgs in {elapsed:.3f}s → {N/elapsed:.1f} msg/s")
        return

    received = 0
    start = time.perf_counter()
    for i in range(N):
        payload["i"] = i
        r.publish(CHANNEL, jsonlib.dumps(payload))
        while True:
            m = ps.get_message(timeout=1.0)
            if m and m.get("type") == "message":
                received += 1
                break
    elapsed = time.perf_counter() - start
    print(
        f"[roundtrip] {received}/{N} msgs in {elapsed:.3f}s → {received/elapsed:.1f} msg/s"
    )


if __name__ == "__main__":
    main()
