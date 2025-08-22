"""
Telemetry Plugin Wrapper - Automatic event broadcasting to MCP server.

This module wraps plugin event emissions to automatically broadcast
telemetry data to the MCP server for real-time monitoring.
"""

import logging
from typing import Any

from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class TelemetryPluginWrapper:
    """
    Wraps plugins to automatically broadcast their events to telemetry.

    This wrapper intercepts plugin event emissions and forwards them
    to the MCP telemetry broadcaster for real-time monitoring.
    """

    def __init__(self, plugin: PluginInterface):
        self.plugin = plugin
        self.original_emit = None
        self._setup_telemetry_monitoring()

    def _setup_telemetry_monitoring(self):
        """Setup telemetry monitoring by wrapping the emit_event method."""
        try:
            # Check if telemetry is available
            # from src.telemetry import EventTypes, broadcast_agent_event  # Currently unused

            # Store original emit method
            if hasattr(self.plugin, "emit_event"):
                self.original_emit = self.plugin.emit_event
                self.plugin.emit_event = self._telemetry_emit_wrapper
                logger.debug(f"✅ Telemetry monitoring enabled for {self.plugin.name}")
            else:
                logger.debug(f"⚠️ Plugin {self.plugin.name} has no emit_event method")

        except ImportError:
            logger.debug("⚠️ Telemetry not available - monitoring disabled")

    async def _telemetry_emit_wrapper(self, event_type: str, event_data: Any, **kwargs):
        """Wrapper that broadcasts events to telemetry before calling original emit."""
        try:
            # Import here to avoid circular imports
            from src.telemetry import broadcast_agent_event

            # Extract relevant data for telemetry
            telemetry_data = {
                "event_type": event_type,
                "plugin_name": self.plugin.name,
                "data_type": type(event_data).__name__,
                "has_kwargs": bool(kwargs),
            }

            # Add specific data based on event type
            if hasattr(event_data, "__dict__"):
                # For events with attributes, capture some metadata
                telemetry_data["event_attributes"] = list(event_data.__dict__.keys())
            elif isinstance(event_data, dict):
                telemetry_data["event_keys"] = list(event_data.keys())

            # Broadcast to telemetry
            await broadcast_agent_event(
                event_type="plugin_event_emission",
                source=f"plugin_{self.plugin.name}",
                data=telemetry_data,
                metadata={"original_event_type": event_type},
            )

        except Exception as e:
            logger.warning(f"Failed to broadcast telemetry for {self.plugin.name}: {e}")

        # Call original emit method
        if self.original_emit:
            return await self.original_emit(event_type, event_data, **kwargs)
        logger.warning(f"No original emit method for {self.plugin.name}")


def wrap_plugin_for_telemetry(plugin: PluginInterface) -> PluginInterface:
    """
    Wrap a plugin to enable automatic telemetry broadcasting.

    Args:
        plugin: The plugin instance to wrap

    Returns:
        The same plugin instance, but with telemetry monitoring enabled
    """
    try:
        TelemetryPluginWrapper(plugin)
        logger.debug(f"🔄 Plugin {plugin.name} wrapped for telemetry")
        return plugin
    except Exception as e:
        logger.warning(f"Failed to wrap plugin {plugin.name} for telemetry: {e}")
        return plugin
