"""Utility to register additional MCP resources and prompts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import TextResource


@dataclass
class ResourceDescriptor:
    """Descriptor for a static resource to expose via MCP."""

    name: str
    mime_type: str
    content: str


@dataclass
class PromptDefinition:
    """Definition for a slash-command style prompt."""

    name: str
    description: str
    content: str


# Mutable lists so tests or callers can append descriptors before registration
RESOURCES: list[ResourceDescriptor] = []
PROMPTS: list[PromptDefinition] = []


def register_with(
    app: FastMCP,
    resources: Sequence[ResourceDescriptor] | None = None,
    prompts: Sequence[PromptDefinition] | None = None,
) -> None:
    """Register resources and prompts with a FastMCP app.

    Args:
        app: The ``FastMCP`` application instance.
        resources: Optional sequence of resource descriptors.
        prompts: Optional sequence of prompt definitions.
    """
    for res in resources or []:
        app.add_resource(
            TextResource(
                uri=f"memory://{res.name}",
                name=res.name,
                mime_type=res.mime_type,
                text=res.content,
            )
        )

    for prm in prompts or []:
        @app.prompt(name=prm.name, description=prm.description)
        async def _prompt(prm=prm) -> str:
            return prm.content


__all__ = [
    "ResourceDescriptor",
    "PromptDefinition",
    "RESOURCES",
    "PROMPTS",
    "register_with",
]
