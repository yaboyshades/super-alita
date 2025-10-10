"""Component discovery utilities for unified orchestration (placeholder)."""

from __future__ import annotations


class ComponentRegistry:
    """Discovers available integrations and adapters."""

    def list_components(self) -> list[str]:
        raise NotImplementedError(
            "ComponentRegistry.list_components pending implementation"
        )
