from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAbility(ABC):
    """Constitutional base interface for abilities.

    Implementations should provide:
    - Metadata: name (snake_case), description, version (semver), author (optional)
    - JSON schemas (optional): input_schema, output_schema
    - Methods: initialize, validate_input, execute, health_check, shutdown
    """

    # Required metadata
    name: str
    description: str
    version: str
    author: str | None = None

    # Optional schemas (advisory only)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.event_bus: Any = None

    @abstractmethod
    async def initialize(
        self, event_bus: Any
    ) -> bool:  # pragma: no cover - interface
        """Wire subscriptions and emit initialization events."""

    @abstractmethod
    def validate_input(
        self, input_data: Any
    ) -> bool:  # pragma: no cover - interface
        """Validate input according to the ability contract."""

    @abstractmethod
    async def execute(
        self, input_data: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover - interface
        """Execute the ability and return a standardized result structure."""

    @abstractmethod
    async def health_check(
        self,
    ) -> dict[str, Any]:  # pragma: no cover - interface
        """Return a health status for monitoring and debugging."""

    @abstractmethod
    async def shutdown(self) -> None:  # pragma: no cover - interface
        """Gracefully shut down the ability."""

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"{getattr(self, 'name', 'ability')} v{getattr(self, 'version', '0.0.0')}"
