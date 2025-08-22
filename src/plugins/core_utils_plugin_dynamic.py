"""
Core Utils Plugin for Super Alita.
Dynamic capability discovery plugin - automatically discovers and exposes
all public methods from CoreUtils without hardcoding.
"""

import inspect
import time
from collections.abc import Callable
from typing import Any

from src.core.events import BaseEvent, ToolCallEvent
from src.core.plugin_interface import PluginInterface
from src.tools.core_utils import CoreUtils


class CoreUtilsPlugin(PluginInterface):
    """
    Dynamic Core Utils Plugin - Auto-discovers and exposes capabilities.
    Follows Super Alita's principle of dynamic capability registration.
    """

    def __init__(self):
        super().__init__()
        self._capabilities: dict[str, Callable] = {}
        self._capability_metadata: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "core_utils"

    async def setup(self, event_bus, store, config):
        """Setup with dynamic capability discovery."""
        await super().setup(event_bus, store, config)

        # Dynamically discover all capabilities from CoreUtils
        self._discover_capabilities()

        # Register capabilities with the system's capability registry
        await self._register_discovered_capabilities()

    def _discover_capabilities(self):
        """
        Dynamically discover all public methods from CoreUtils.
        This ensures no hardcoding of available capabilities.
        """
        self._capabilities.clear()
        self._capability_metadata.clear()

        # Get all public methods/functions from CoreUtils
        for name, method in inspect.getmembers(CoreUtils):
            # Skip private methods and non-callable attributes
            if name.startswith("_") or not callable(method):
                continue

            # Generate tool name (e.g., calculate -> core.calculate)
            tool_name = f"core.{name}"

            # Store the capability
            self._capabilities[tool_name] = method

            # Extract metadata from method signature and docstring
            signature = inspect.signature(method)
            docstring = inspect.getdoc(method) or f"Dynamic capability: {name}"

            self._capability_metadata[tool_name] = {
                "name": tool_name,
                "method": name,
                "signature": str(signature),
                "docstring": docstring,
                "parameters": [param.name for param in signature.parameters.values()],
                "is_static": isinstance(
                    inspect.getattr_static(CoreUtils, name), staticmethod
                ),
                "discovered_dynamically": True,
            }

        print(f"🔍 Dynamically discovered {len(self._capabilities)} capabilities:")
        for tool_name, metadata in self._capability_metadata.items():
            print(
                f"  • {tool_name}{metadata['signature']} - {metadata['docstring'][:50]}..."
            )

    async def _register_discovered_capabilities(self):
        """
        Register all discovered capabilities with Super Alita's capability registry.
        This enables the system to know about and route to these tools.
        """
        if hasattr(self.store, "register_capabilities"):
            # Register with neural store if it supports capability registration
            capabilities_data = {
                "plugin": self.name,
                "capabilities": self._capability_metadata,
                "discovery_method": "dynamic_reflection",
                "total_count": len(self._capabilities),
            }
            await self.store.register_capabilities(self.name, capabilities_data)

        # Emit capability discovery event for system awareness
        if hasattr(self.event_bus, "publish"):
            try:
                discovery_event = BaseEvent(
                    event_type="capabilities_discovered",
                    source_plugin=self.name,
                    metadata={
                        "capabilities": list(self._capabilities.keys()),
                        "capability_metadata": self._capability_metadata,
                        "discovery_timestamp": time.time(),
                        "total_count": len(self._capabilities),
                    },
                )
                await self.event_bus.publish(discovery_event)
            except Exception as e:
                print(f"⚠️ Warning: Could not publish capability discovery event: {e}")
                # Continue anyway - this is not critical for operation

    async def start(self):
        """Start the plugin and subscribe to tool calls."""
        await super().start()
        await self.subscribe("tool_call", self._handle_dynamic_tool_call)

        print(
            f"✅ {self.name} plugin started with {len(self._capabilities)} dynamic capabilities"
        )

    async def _handle_dynamic_tool_call(self, event: ToolCallEvent):
        """
        Handle tool calls dynamically based on discovered capabilities.
        No hardcoded tool names - fully dynamic routing.
        """
        tool_name = event.tool_name

        # Check if we can handle this tool dynamically
        if tool_name not in self._capabilities:
            # Tool not in our discovered capabilities
            return

        try:
            # Get the capability method
            capability_method = self._capabilities[tool_name]
            metadata = self._capability_metadata[tool_name]

            # Extract parameters based on method signature
            method_params = self._extract_method_parameters(
                capability_method, event.parameters, metadata
            )

            # Execute the capability dynamically
            if metadata["is_static"]:
                result = capability_method(**method_params)
            else:
                # For instance methods, we need to handle them appropriately
                result = capability_method(**method_params)

            # Emit success result
            await self.emit_event(
                "tool_result",
                tool_call_id=event.tool_call_id,
                conversation_id=event.conversation_id,
                session_id=event.session_id,
                success=True,
                result={
                    "result": result,
                    "tool_name": tool_name,
                    "method_used": metadata["method"],
                    "executed_dynamically": True,
                },
            )

        except Exception as e:
            # Emit error result
            await self.emit_event(
                "tool_result",
                tool_call_id=event.tool_call_id,
                conversation_id=event.conversation_id,
                session_id=event.session_id,
                success=False,
                error=f"Dynamic execution error in {tool_name}: {e!s}",
                result={},
            )

    def _extract_method_parameters(
        self, method: Callable, event_params: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Dynamically extract and map parameters for method execution.
        This allows flexible parameter handling without hardcoding.
        """
        signature = inspect.signature(method)
        method_params = {}

        for param_name, param_info in signature.parameters.items():
            if param_name in event_params:
                method_params[param_name] = event_params[param_name]
            elif param_info.default != inspect.Parameter.empty:
                # Use default value if available
                method_params[param_name] = param_info.default
            # else: let the method handle missing required parameters

        return method_params

    def get_discovered_capabilities(self) -> dict[str, dict[str, Any]]:
        """
        Return all dynamically discovered capabilities.
        Useful for system introspection and capability queries.
        """
        return self._capability_metadata.copy()

    def can_handle_tool(self, tool_name: str) -> bool:
        """
        Dynamically check if this plugin can handle a tool.
        No hardcoded tool names - purely discovery-based.
        """
        return tool_name in self._capabilities

    async def shutdown(self):
        """Clean shutdown of dynamic capabilities."""
        print(
            f"🔄 Shutting down {self.name} with {len(self._capabilities)} dynamic capabilities"
        )
        self._capabilities.clear()
        self._capability_metadata.clear()
        await super().shutdown()
