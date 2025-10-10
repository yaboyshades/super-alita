"""Event base classes for test environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaseEvent:
    type: str
    payload: dict[str, Any] | None = None
