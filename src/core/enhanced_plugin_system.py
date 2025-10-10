"""
Enhanced Plugin System Integration for Super Alita
Integrates YAML-based plugin configuration with existing plugin registry.
"""

import contextlib
import logging
from typing import Any

from src.core.plugin_loader import (
    load_plugin_manifest,
    load_plugins_from_manifest,
)
from src.core.plugin_registry import register_plugin

logger = logging.getLogger(__name__)


async def initialize_enhanced_plugin_system(app: Any) -> dict[str, Any]:
    """
    Initialize enhanced plugin system with YAML configuration.

    Args:
        app: FastAPI application instance

    Returns:
        Dict[str, Any]: Loaded plugins by name
    """
    try:
        logger.info("🔌 Initializing enhanced plugin system...")

        # Load plugin manifest
        plugins_config = load_plugin_manifest("plugins.yaml")
        if not plugins_config:
            logger.warning(
                "No plugin configuration found, falling back to manual loading"
            )
            return await _fallback_to_manual_plugins(app)

        logger.info(f"📋 Found {len(plugins_config)} plugins in configuration")

        # Get event bus from app state
        event_bus = getattr(app.state, "event_bus", None)
        if not event_bus:
            logger.error(
                "Event bus not available - plugin initialization failed"
            )
            return {}

        # Load plugins using manifest
        loaded_plugins = await load_plugins_from_manifest(
            plugins_config, event_bus=event_bus
        )

        # Register loaded plugins in global registry
        for name, plugin in loaded_plugins.items():
            register_plugin(name, plugin)
            logger.info(f"✅ Registered plugin: {name}")

        # Store plugins in app state
        app.state.plugins = list(loaded_plugins.values())
        app.state.plugin_names = list(loaded_plugins.keys())

        # Expose plugin tools to ability registry
        await _expose_plugin_tools(app, loaded_plugins)

        logger.info(
            f"🎉 Enhanced plugin system initialized with {len(loaded_plugins)} plugins"
        )

        # Emit plugin system ready event
        if hasattr(event_bus, "emit"):
            await event_bus.emit(
                "plugin_system.ready",
                {
                    "plugin_count": len(loaded_plugins),
                    "plugin_names": list(loaded_plugins.keys()),
                },
            )

        return loaded_plugins

    except Exception as e:
        logger.error(f"❌ Enhanced plugin system initialization failed: {e}")
        logger.info("🔄 Falling back to manual plugin loading...")
        return await _fallback_to_manual_plugins(app)


async def _fallback_to_manual_plugins(app: Any) -> dict[str, Any]:
    """
    Fallback to manual plugin loading when YAML config fails.
    Maintains the existing static plugin list.
    """
    plugins = {}

    try:
        # Import existing plugins (keeping the current behavior)
        with contextlib.suppress(Exception):
            from src.core.copilot_snippet_optimizer import (
                SnippetOptimizedCopilotPlugin,
            )
            from src.plugins.autogen_creator_plugin import AutogenCreatorPlugin
            from src.plugins.deepcode_generator_plugin import (
                DeepCodeGeneratorBridgePlugin,
            )
            from src.plugins.deepcode_orchestrator_plugin import (
                DeepCodeOrchestratorPlugin,
            )
            from src.plugins.mcp_adapter_plugin import MCPAdapterPlugin
            from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin
            from src.plugins.native_perplexica_plugin import (
                NativePerplexicaPlugin,
            )
            from src.plugins.telemetry_pipeline import TelemetryPipelinePlugin

            # Create plugin instances
            plugin_instances = {
                "deepcode_generator": DeepCodeGeneratorBridgePlugin(),
                "deepcode_orchestrator": DeepCodeOrchestratorPlugin(),
                "autogen_creator": AutogenCreatorPlugin(),
                "native_deepcode": NativeDeepCodePlugin(),
                "native_perplexica": NativePerplexicaPlugin(),
                "telemetry_pipeline": TelemetryPipelinePlugin(),
                "mcp_adapter": MCPAdapterPlugin(),
                "snippet_optimizer": SnippetOptimizedCopilotPlugin(),
            }

            # Setup and start plugins
            event_bus = getattr(app.state, "event_bus", None)
            for name, plugin in plugin_instances.items():
                try:
                    if hasattr(plugin, "setup"):
                        await plugin.setup(event_bus, store=None, config={})
                    if hasattr(plugin, "start"):
                        await plugin.start()

                    plugins[name] = plugin
                    register_plugin(name, plugin)
                    logger.info(f"✅ Fallback loaded plugin: {name}")

                except Exception as e:
                    logger.error(
                        f"❌ Failed to load fallback plugin {name}: {e}"
                    )

            # Store in app state
            app.state.plugins = list(plugins.values())
            app.state.plugin_names = list(plugins.keys())

            logger.info(
                f"🔄 Fallback plugin loading completed: {len(plugins)} plugins"
            )

    except Exception as e:
        logger.error(f"❌ Fallback plugin loading failed: {e}")
        app.state.plugins = []
        app.state.plugin_names = []

    return plugins


async def _expose_plugin_tools(app: Any, plugins: dict[str, Any]) -> None:
    """
    Expose plugin tools to the ability registry for dynamic tool discovery.

    Args:
        app: FastAPI application instance
        plugins: Dictionary of loaded plugins
    """
    try:
        ability_reg = getattr(app.state, "ability_registry", None)
        if not ability_reg:
            logger.warning(
                "Ability registry not available - plugin tools not exposed"
            )
            return

        tool_count = 0
        for name, plugin in plugins.items():
            try:
                # Check if plugin provides tools
                get_tools = getattr(plugin, "get_tools", None)
                if callable(get_tools):
                    tools = get_tools()
                    if tools:
                        # Register each tool with the ability registry
                        for tool in tools:
                            if hasattr(ability_reg, "register_tool"):
                                ability_reg.register_tool(tool)
                                tool_count += 1

                        logger.info(
                            f"🔧 Exposed {len(tools)} tools from plugin: {name}"
                        )

            except Exception as e:
                logger.error(
                    f"❌ Failed to expose tools from plugin {name}: {e}"
                )

        if tool_count > 0:
            logger.info(f"🛠️  Total plugin tools exposed: {tool_count}")

    except Exception as e:
        logger.error(f"❌ Failed to expose plugin tools: {e}")


async def shutdown_enhanced_plugin_system(app: Any) -> None:
    """
    Gracefully shutdown the enhanced plugin system.

    Args:
        app: FastAPI application instance
    """
    try:
        plugins = getattr(app.state, "plugins", [])
        plugin_names = getattr(app.state, "plugin_names", [])

        logger.info(f"🔌 Shutting down {len(plugins)} plugins...")

        for i, plugin in enumerate(plugins):
            try:
                plugin_name = (
                    plugin_names[i] if i < len(plugin_names) else f"plugin_{i}"
                )

                # Try different shutdown methods
                if hasattr(plugin, "cleanup"):
                    await plugin.cleanup()
                elif hasattr(plugin, "shutdown"):
                    await plugin.shutdown()
                elif hasattr(plugin, "stop"):
                    await plugin.stop()

                logger.info(f"✅ Shutdown plugin: {plugin_name}")

            except Exception as e:
                logger.error(f"❌ Error shutting down plugin: {e}")

        # Clear app state
        app.state.plugins = []
        app.state.plugin_names = []

        logger.info("🔌 Plugin system shutdown complete")

    except Exception as e:
        logger.error(f"❌ Plugin system shutdown failed: {e}")


def get_plugin_health_status(app: Any) -> dict[str, Any]:
    """
    Get health status of all loaded plugins.

    Args:
        app: FastAPI application instance

    Returns:
        Dict[str, Any]: Plugin health information
    """
    try:
        plugins = getattr(app.state, "plugins", [])
        plugin_names = getattr(app.state, "plugin_names", [])

        health_info = {
            "plugin_count": len(plugins),
            "plugins": {},
            "status": "healthy" if plugins else "no_plugins",
        }

        for i, plugin in enumerate(plugins):
            plugin_name = (
                plugin_names[i] if i < len(plugin_names) else f"plugin_{i}"
            )

            # Try to get plugin status
            try:
                if hasattr(plugin, "get_status"):
                    status = plugin.get_status()
                elif hasattr(plugin, "get_health_status"):
                    status = plugin.get_health_status()
                else:
                    status = {
                        "name": plugin_name,
                        "available": True,
                        "type": type(plugin).__name__,
                    }

                health_info["plugins"][plugin_name] = status

            except Exception as e:
                health_info["plugins"][plugin_name] = {
                    "name": plugin_name,
                    "available": False,
                    "error": str(e),
                }

        return health_info

    except Exception as e:
        return {
            "plugin_count": 0,
            "plugins": {},
            "status": "error",
            "error": str(e),
        }
