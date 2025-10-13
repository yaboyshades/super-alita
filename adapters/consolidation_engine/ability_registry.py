"""Ability registry adapter for consolidation follow-up actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from domain.consolidation_engine.service import AbilityRegistryAdapter


@runtime_checkable
class AbilityRegistryLike(Protocol):
    """Contract subset exposed by the runtime ability registry."""

    async def execute(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        correlation_id: str,
    ) -> Mapping[str, Any]:
        """Execute an ability and return structured output."""


class AbilityRegistryAdapterImpl(AbilityRegistryAdapter):
    """Concrete adapter delegating to the runtime registry implementation."""

    def __init__(self, registry: AbilityRegistryLike) -> None:
        self._registry = registry

    async def execute(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        correlation_id: str,
    ) -> Mapping[str, Any]:
        return await self._registry.execute(
            name,
            payload,
            correlation_id=correlation_id,
        )
