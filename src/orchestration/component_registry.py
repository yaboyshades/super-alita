"""Hot-swappable component registry with dependency injection.

Components register capabilities and dependencies dynamically. Registry
manages lifecycle, dependency resolution, and graceful restarts without downtime.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from src.contracts import HealthStatus


@dataclass
class ComponentDescriptor:
    """Descriptor for a registerable component.

    Attributes:
        name: Unique component identifier
        provides: Set of capability identifiers this component provides
        requires: Set of capability identifiers this component depends on
        health_check: Async function returning HealthStatus
        shutdown_timeout: Max seconds to wait for graceful shutdown
        priority: Registration priority (higher = starts first)
    """

    name: str
    provides: set[str] = field(default_factory=set)
    requires: set[str] = field(default_factory=set)
    health_check: Callable[[], Coroutine[Any, Any, HealthStatus]] | None = None
    shutdown_timeout: float = 30.0
    priority: int = 0


class ComponentRegistry:
    """Registry for hot-swappable components with dependency management.

    Manages component lifecycle:
    1. Validates dependencies are met before registration
    2. Notifies dependents when new capabilities become available
    3. Supports hot-swapping with graceful shutdown
    4. Tracks component health and capabilities
    """

    def __init__(self):
        """Initialize empty registry."""
        self.components: dict[str, Any] = {}
        self.descriptors: dict[str, ComponentDescriptor] = {}
        # capability -> component names
        self.capability_providers: dict[str, set[str]] = {}
        # capability -> dependent component names
        self.dependents: dict[str, set[str]] = {}
        self._startup_order: list[str] = []

    async def register(self, desc: ComponentDescriptor, instance: Any) -> None:
        """Register a component instance.

        Args:
            desc: Component descriptor
            instance: Component instance

        Raises:
            ValueError: If dependencies not met or component already registered
        """
        if desc.name in self.components:
            raise ValueError(f"Component {desc.name} already registered")

        # Check dependencies
        missing = await self._check_dependencies(desc.requires)
        if missing:
            raise ValueError(
                f"Component {desc.name} missing dependencies: {missing}"
            )

        # Register component
        self.components[desc.name] = instance
        self.descriptors[desc.name] = desc

        # Update capability mappings
        for capability in desc.provides:
            if capability not in self.capability_providers:
                self.capability_providers[capability] = set()
            self.capability_providers[capability].add(desc.name)

        # Track dependents
        for capability in desc.requires:
            if capability not in self.dependents:
                self.dependents[capability] = set()
            self.dependents[capability].add(desc.name)

        # Track startup order
        self._startup_order.append(desc.name)

        # Notify components waiting for these capabilities
        await self._notify_dependents(desc.provides)

    async def unregister(self, name: str) -> None:
        """Unregister a component.

        Args:
            name: Component name

        Raises:
            ValueError: If component has active dependents
        """
        if name not in self.components:
            return

        desc = self.descriptors[name]

        # Check if any components depend on this one's capabilities
        active_dependents = set()
        for capability in desc.provides:
            if capability in self.dependents:
                active_dependents.update(self.dependents[capability])

        # Remove self from dependents
        active_dependents.discard(name)

        if active_dependents:
            raise ValueError(
                f"Cannot unregister {name}: active dependents {active_dependents}"
            )

        # Graceful shutdown
        instance = self.components[name]
        if hasattr(instance, "graceful_shutdown"):
            await asyncio.wait_for(
                instance.graceful_shutdown(),
                timeout=desc.shutdown_timeout,
            )

        # Remove from registry
        del self.components[name]
        del self.descriptors[name]

        # Update capability mappings
        for capability in desc.provides:
            if capability in self.capability_providers:
                self.capability_providers[capability].discard(name)
                if not self.capability_providers[capability]:
                    del self.capability_providers[capability]

        # Update startup order
        if name in self._startup_order:
            self._startup_order.remove(name)

    async def hot_swap(
        self, name: str, new_desc: ComponentDescriptor, new_instance: Any
    ) -> None:
        """Hot-swap a component with zero downtime.

        Args:
            name: Component name to replace
            new_desc: New component descriptor
            new_instance: New component instance

        Raises:
            ValueError: If component not registered or swap invalid
        """
        if name not in self.components:
            raise ValueError(f"Component {name} not registered")

        old_desc = self.descriptors[name]
        old_instance = self.components[name]

        # Verify new component provides at least the same capabilities
        if not new_desc.provides.issuperset(old_desc.provides):
            raise ValueError(
                f"New component must provide all capabilities of old: "
                f"old={old_desc.provides}, new={new_desc.provides}"
            )

        # Graceful shutdown of old instance
        if hasattr(old_instance, "graceful_shutdown"):
            await asyncio.wait_for(
                old_instance.graceful_shutdown(),
                timeout=old_desc.shutdown_timeout,
            )

        # Swap components
        self.components[name] = new_instance
        self.descriptors[name] = new_desc

        # Update capability mappings for new capabilities
        for capability in new_desc.provides - old_desc.provides:
            if capability not in self.capability_providers:
                self.capability_providers[capability] = set()
            self.capability_providers[capability].add(name)

        # Notify dependents of new capabilities
        await self._notify_dependents(new_desc.provides - old_desc.provides)

    async def _check_dependencies(self, requires: set[str]) -> set[str]:
        """Check if required capabilities are available.

        Args:
            requires: Set of required capabilities

        Returns:
            Set of missing capabilities
        """
        missing = set()
        for capability in requires:
            if (
                capability not in self.capability_providers
                or not self.capability_providers[capability]
            ):
                missing.add(capability)
        return missing

    async def _notify_dependents(self, capabilities: set[str]) -> None:
        """Notify components waiting for capabilities.

        Args:
            capabilities: Newly available capabilities
        """
        for capability in capabilities:
            if capability in self.dependents:
                for dependent_name in self.dependents[capability]:
                    dependent = self.components.get(dependent_name)
                    if dependent and hasattr(
                        dependent, "on_capability_available"
                    ):
                        await dependent.on_capability_available(capability)

    def get_component(self, name: str) -> Any | None:
        """Get component by name.

        Args:
            name: Component name

        Returns:
            Component instance or None
        """
        return self.components.get(name)

    def get_providers(self, capability: str) -> set[str]:
        """Get component names providing a capability.

        Args:
            capability: Capability identifier

        Returns:
            Set of component names
        """
        return self.capability_providers.get(capability, set()).copy()

    async def check_health(self, name: str) -> HealthStatus:
        """Check health of a specific component.

        Args:
            name: Component name

        Returns:
            HealthStatus from component's health check
        """
        desc = self.descriptors.get(name)
        if not desc or not desc.health_check:
            return HealthStatus(
                component=name,
                status="unhealthy",
                message="No health check configured",
            )

        try:
            return await desc.health_check()
        except Exception as e:
            return HealthStatus(
                component=name,
                status="unhealthy",
                message=f"Health check failed: {e}",
            )

    async def check_all_health(self) -> dict[str, HealthStatus]:
        """Check health of all components.

        Returns:
            Dict mapping component names to health status
        """
        health_results = {}
        for name in self.components:
            health_results[name] = await self.check_health(name)
        return health_results

    def get_startup_order(self) -> list[str]:
        """Get component startup order.

        Returns:
            List of component names in startup order
        """
        return self._startup_order.copy()

    def get_dependency_graph(self) -> dict[str, dict[str, Any]]:
        """Get dependency graph for visualization.

        Returns:
            Dict with nodes (components) and edges (dependencies)
        """
        graph = {
            "nodes": [],
            "edges": [],
        }

        for name, desc in self.descriptors.items():
            graph["nodes"].append(
                {
                    "id": name,
                    "provides": list(desc.provides),
                    "requires": list(desc.requires),
                    "priority": desc.priority,
                }
            )

            # Add edges for dependencies
            for capability in desc.requires:
                providers = self.get_providers(capability)
                for provider in providers:
                    graph["edges"].append(
                        {
                            "from": provider,
                            "to": name,
                            "capability": capability,
                        }
                    )

        return graph
