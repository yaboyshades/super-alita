"""Simple in-memory working memory buffer."""
from __future__ import annotations

import collections
import threading
from typing import Deque, Iterable, List

from src.models import Memory, MemoryType


class WorkingBuffer:
    """Ring buffer keeping the last *capacity* memories in working set."""

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self._lock = threading.RLock()
        self._buffer: Deque[Memory] = collections.deque(maxlen=capacity)

    def add(self, memory: Memory) -> None:
        with self._lock:
            memory.kind = MemoryType.WORKING
            self._buffer.append(memory)

    def extend(self, memories: Iterable[Memory]) -> None:
        with self._lock:
            for memory in memories:
                self.add(memory)

    def latest(self, k: int = 5) -> List[Memory]:
        with self._lock:
            return list(list(self._buffer)[-k:])

    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


working_buffer = WorkingBuffer()
