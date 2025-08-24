"""
Unified Capability Registry wrapper across normal, MCP, neural, and dynamic tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Capability:
    name: str
    description: str
    registry_type: str
    spec: dict[str, Any]

class UnifiedCapabilityRegistry:
    def __init__(self):
        self.normal_tools: dict[str, dict[str, Any]] = {}
        self.mcp_tools: dict[str, dict[str, Any]] = {}
        self.neural_atoms: dict[str, dict[str, Any]] = {}
        self.dynamic_tools: dict[str, dict[str, Any]] = {}

    def get_all_capabilities(self) -> list[Capability]:
        capabilities: list[Capability] = []
        for registry_name, registry in [
            ("normal", self.normal_tools),
            ("mcp", self.mcp_tools),
            ("neural", self.neural_atoms),
            ("dynamic", self.dynamic_tools),
        ]:
            for name, spec in registry.items():
                capabilities.append(
                    Capability(
                        name=name,
                        description=spec.get("description", ""),
                        registry_type=registry_name,
                        spec=spec,
                    )
                )
        return capabilities

    def register_capability(self, capability: Capability, registry_type: str) -> None:
        registry_map = {
            "normal": self.normal_tools,
            "mcp": self.mcp_tools,
            "neural": self.neural_atoms,
            "dynamic": self.dynamic_tools,
        }
        if registry_type in registry_map:
            registry_map[registry_type][capability.name] = capability.spec
