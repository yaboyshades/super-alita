"""Performance marker shim used in test environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PerformanceMarker:
    name: str
    metadata: dict[str, Any] | None = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
