#!/usr/bin/env python3
"""
AtomTools Plugin: Manages tools as NeuralAtoms

This plugin registers built-in tools as atoms, handles tool discovery via
attention mechanism, and provides the execution interface for the planner.

Architecture:
- Tools are stored as NeuralAtoms with special "tool" type
- Discovery uses semantic search over tool descriptions
- Execution dispatches to concrete AtomTool implementations
- Results are logged and can be stored as new atoms
"""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src.core.atom_tool import (
    BUILT_IN_TOOLS,
    AtomTool,
    ToolResult,
)
from src.core.config import EMBEDDING_DIM
from src.core.event_bus import EventBus
from src.core.events import BaseEvent
from src.core.neural_atom import NeuralStore, create_memory_atom
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class ToolExecutionEvent(BaseEvent):
    """Event for tool execution requests."""

    event_type: str = "tool_execution"
    tool_key: str
    parameters: dict[str, Any]
    session_id: str
    request_id: str


class ToolResultEvent(BaseEvent):
    """Event for tool execution results."""

    event_type: str = "tool_result"
    request_id: str
    tool_key: str
    result: dict[str, Any]
    success: bool
    execution_time: float


class AtomToolsPlugin(PluginInterface):
    """
    Plugin that manages tools as NeuralAtoms.

    Features:
    - Registers built-in tools as atoms on startup
    - Provides tool discovery via attention mechanism
    - Executes tools and returns structured results
    - Logs tool usage for learning and optimization
    """

    def __init__(self):
        super().__init__()
        self._tools_cache: dict[str, AtomTool] = {}
        self._execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
        }

    @property
    def name(self) -> str:
        return "atom_tools"

    @property
    def version(self) -> str:
        return "1.0.0"

    async def setup(
        self, event_bus: EventBus, store: NeuralStore, config: dict[str, Any]
    ):
        """Setup the atom tools plugin."""
        await super().setup(event_bus, store, config)
        self._config = config.get(self.name, {})

        # Register built-in tools as atoms
        await self._register_built_in_tools()

        logger.info("AtomToolsPlugin setup complete")

    async def start(self) -> None:
        """Start the plugin and subscribe to events."""
        await super().start()

        # Subscribe to tool execution requests
        await self.subscribe("tool_execution", self._handle_tool_execution)
        await self.subscribe("tool_call_request", self._handle_tool_call_request)

        logger.info(
            f"AtomToolsPlugin started - {len(self._tools_cache)} tools available"
        )

    async def _register_built_in_tools(self):
        """Register all built-in tools as NeuralAtoms."""
        logger.info("Registering built-in tools as atoms...")

        for tool_key, tool_class in BUILT_IN_TOOLS.items():
            try:
                # Create tool instance
                tool_instance = tool_class()
                self._tools_cache[tool_key] = tool_instance

                # Create embedding for tool (would use real embedding service)
                embedding = np.random.rand(EMBEDDING_DIM).astype(
                    np.float32
                )  # Mock embedding

                # Create memory atom
                atom = create_memory_atom(
                    memory_key=f"tool_{tool_key}",
                    content=tool_instance.to_atom_value(),
                    memory_type="tool",
                    embedding_vector=embedding,
                    confidence=1.0,
                )

                # Register with store
                self.store.register(atom)
                logger.info(f"✅ Registered tool: {tool_instance.name} ({tool_key})")

            except Exception as e:
                logger.error(f"❌ Failed to register tool {tool_key}: {e}")

    async def discover_tools(
        self, query: str, top_k: int = 5
    ) -> list[tuple[str, float, AtomTool]]:
        """
        Discover tools using semantic search.

        Args:
            query: Natural language description of desired tool
            top_k: Maximum number of tools to return

        Returns:
            List of (tool_key, similarity_score, tool_instance) tuples
        """
        try:
            # Create query embedding (would use real embedding service)
            query_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)  # Mock

            # Search for tool atoms
            attention_results = await self.store.attention(query_embedding, top_k=top_k)

            discovered_tools = []
            for atom_key, similarity in attention_results:
                atom = self.store.get(atom_key)
                if (
                    atom
                    and isinstance(atom.value, dict)
                    and atom.value.get("tool_type") == "atom_tool"
                ):
                    tool_key = atom.value.get("key")
                    if tool_key in self._tools_cache:
                        discovered_tools.append(
                            (tool_key, similarity, self._tools_cache[tool_key])
                        )

            logger.info(f"Discovered {len(discovered_tools)} tools for query: {query}")
            return discovered_tools

        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            return []

    async def execute_tool(
        self, tool_key: str, parameters: dict[str, Any]
    ) -> ToolResult:
        """
        Execute a tool by key.

        Args:
            tool_key: The tool identifier
            parameters: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        start_time = datetime.now(UTC)

        try:
            # Get tool from cache
            tool = self._tools_cache.get(tool_key)
            if not tool:
                return ToolResult(success=False, error=f"Tool not found: {tool_key}")

            logger.info(f"🔧 Executing tool: {tool.name} with params: {parameters}")

            # Execute tool
            result = await tool.call(**parameters)

            # Update stats
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            self._execution_stats["total_executions"] += 1

            if result.success:
                self._execution_stats["successful_executions"] += 1
                logger.info(f"✅ Tool executed successfully in {execution_time:.2f}s")
            else:
                self._execution_stats["failed_executions"] += 1
                logger.warning(f"❌ Tool execution failed: {result.error}")

            # Update average execution time
            total = self._execution_stats["total_executions"]
            current_avg = self._execution_stats["average_execution_time"]
            self._execution_stats["average_execution_time"] = (
                current_avg * (total - 1) + execution_time
            ) / total

            return result

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ToolResult(success=False, error=f"Execution error: {e!s}")

    async def _handle_tool_execution(self, event: ToolExecutionEvent):
        """Handle tool execution event."""
        try:
            result = await self.execute_tool(event.tool_key, event.parameters)

            # Emit result event
            result_event = ToolResultEvent(
                source_plugin=self.name,
                request_id=event.request_id,
                tool_key=event.tool_key,
                result=result.model_dump(),
                success=result.success,
                execution_time=0.0,  # Would calculate actual time
            )

            await self.event_bus.publish(result_event)

        except Exception as e:
            logger.error(f"Tool execution handler failed: {e}")

    async def _handle_tool_call_request(self, event):
        """Handle legacy tool call request events."""
        try:
            # Extract parameters from event
            action = getattr(event, "action", "unknown")
            parameters = getattr(event, "parameters", {})

            # Map action to tool key
            tool_mapping = {
                "shell": "shell_executor",
                "math": "python_math",
                "research": "python_math",  # Fallback
            }

            tool_key = tool_mapping.get(action, "shell_executor")

            # Execute tool
            if action == "shell" and "command" in parameters:
                result = await self.execute_tool(
                    tool_key, {"command": parameters["command"]}
                )
            elif action in ["math", "research"]:
                # Try to extract mathematical expression
                query = parameters.get("query", "")
                result = await self.execute_tool(
                    tool_key,
                    {"expression": f"# Analysis: {query}\\nprint('Analysis complete')"},
                )
            else:
                result = ToolResult(success=False, error=f"Unknown action: {action}")

            logger.info(f"Tool call result: {result.success}")

        except Exception as e:
            logger.error(f"Tool call request handler failed: {e}")

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """Get list of all available tools."""
        tools = []
        for tool_key, tool in self._tools_cache.items():
            tools.append(
                {
                    "key": tool_key,
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "signature": tool.signature.model_dump(),
                }
            )
        return tools

    async def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        return {
            **self._execution_stats,
            "available_tools": len(self._tools_cache),
            "tool_categories": list(
                set(tool.category for tool in self._tools_cache.values())
            ),
        }


# Make the plugin available for import
__all__ = ["AtomToolsPlugin"]
