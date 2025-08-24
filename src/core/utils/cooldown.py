from __future__ import annotations

import time
from collections import OrderedDict
from typing import Optional


class CooldownLRU:
    """
    Small LRU-style cooldown cache.
    Tracks last-hit timestamp per key; returns True if within cooldown.
    """

    def __init__(self, maxsize: int = 1024, ttl_seconds: float = 300.0):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._store: OrderedDict[str, float] = OrderedDict()

    def _prune(self) -> None:
        now = time.time()
        to_del = [k for k, ts in self._store.items() if (now - ts) > self.ttl]
        for k in to_del:
            self._store.pop(k, None)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def hit(self, key: str) -> bool:
        """
        Record a hit and return True if it's within cooldown (i.e., should skip).
        """
        now = time.time()
        prev = self._store.get(key)
        self._store[key] = now
        self._store.move_to_end(key, last=True)
        self._prune()
        if prev is None:
            return False
        return (now - prev) < self.ttl
