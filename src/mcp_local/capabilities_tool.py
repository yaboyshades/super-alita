"""
MCP Capability Introspection Tool for Super Alita.

Provides real-time capability enumeration for external clients including
plugins, events, and tools. Enables dynamic capability discovery.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def collect_capabilities() -> dict[str, Any]:
    """
    Collect comprehensive capability information from the running system.

    Returns:
        Dictionary containing plugins, events, tools, and system metadata
    """
    capabilities = {
        "timestamp": _get_timestamp(),
        "version": "1.0",
        "plugins": _collect_plugin_capabilities(),
        "events": _collect_event_capabilities(),
        "tools": _collect_tool_capabilities(),
        "system": _collect_system_capabilities(),
    }

    logger.debug("Collected capabilities snapshot")
    return capabilities


def _get_timestamp() -> float:
    """Get current timestamp."""
    import time

    return time.time()


def _collect_plugin_capabilities() -> list[dict[str, Any]]:
    """
    Collect information about loaded plugins.

    Returns:
        List of plugin capability descriptors
    """
    plugins = []

    try:
        # Try to get plugin info from loader if available
        from ..core.plugin_loader import get_plugin_info, load_plugin_manifest

        manifest = load_plugin_manifest()
        if manifest:
            plugin_info = get_plugin_info(manifest)

            for name, info in plugin_info.items():
                plugins.append(
                    {
                        "name": name,
                        "module": info["module"],
                        "enabled": info["enabled"],
                        "priority": info["priority"],
                        "depends_on": info["depends_on"],
                        "description": info["description"],
                        "category": info.get("category", "uncategorized"),
                        "status": "configured",
                    }
                )

        # TODO: Add runtime plugin state detection
        # Could check if plugin instances are actually loaded and active

    except ImportError:
        logger.debug("Plugin loader not available - using fallback detection")
        # Fallback: try to detect from common plugin locations
        plugins.append(
            {
                "name": "fallback_detection",
                "status": "detection_limited",
                "note": "Plugin loader not available - install plugin system for full capabilities",
            }
        )
    except Exception as e:
        logger.warning(f"Error collecting plugin capabilities: {e}")
        plugins.append({"name": "error_state", "status": "error", "error": str(e)})

    return plugins


def _collect_event_capabilities() -> dict[str, Any]:
    """
    Collect information about registered event types.

    Returns:
        Dictionary containing event registry information
    """
    events = {"total_count": 0, "types": [], "deprecated_count": 0}

    try:
        from ..core.event_types import list_events

        event_registry = list_events()
        events["total_count"] = len(event_registry)

        for name, descriptor in event_registry.items():
            event_info = {
                "name": name,
                "version": descriptor.version,
                "description": descriptor.description,
                "deprecated": descriptor.deprecated,
                "has_schema": bool(descriptor.schema),
            }

            if descriptor.deprecated:
                events["deprecated_count"] += 1
                if descriptor.successor:
                    event_info["successor"] = descriptor.successor

            # Include schema summary if present
            if descriptor.schema:
                event_info["schema_summary"] = {
                    "required_fields": descriptor.schema.get("required", []),
                    "field_count": len(descriptor.schema.get("properties", {})),
                }

            events["types"].append(event_info)

        # Sort by name for consistent output
        events["types"].sort(key=lambda x: x["name"])

    except ImportError:
        logger.debug("Event registry not available")
        events["status"] = "registry_not_available"
    except Exception as e:
        logger.warning(f"Error collecting event capabilities: {e}")
        events["status"] = "error"
        events["error"] = str(e)

    return events


def _collect_tool_capabilities() -> dict[str, Any]:
    """
    Collect information about registered tools.

    Returns:
        Dictionary containing tool registry information
    """
    tools = {"total_count": 0, "categories": {}, "tools": []}

    try:
        # Try to get tool registry information
        # Note: This would need to be implemented based on your tool registry structure

        # Placeholder implementation - replace with actual tool registry access
        tools["status"] = "detection_pending"
        tools["note"] = (
            "Tool registry integration pending - implement based on actual tool storage"
        )

        # Example of what this could look like:
        # from ..core.tool_registry import get_all_tools
        # tool_registry = get_all_tools()
        #
        # for tool_name, tool_info in tool_registry.items():
        #     category = tool_info.get("category", "uncategorized")
        #     tools["categories"][category] = tools["categories"].get(category, 0) + 1
        #
        #     tools["tools"].append({
        #         "name": tool_name,
        #         "category": category,
        #         "description": tool_info.get("description", ""),
        #         "origin_plugin": tool_info.get("origin_plugin"),
        #         "has_schema": bool(tool_info.get("inputs"))
        #     })

    except ImportError:
        logger.debug("Tool registry not available")
        tools["status"] = "registry_not_available"
    except Exception as e:
        logger.warning(f"Error collecting tool capabilities: {e}")
        tools["status"] = "error"
        tools["error"] = str(e)

    return tools


def _collect_system_capabilities() -> dict[str, Any]:
    """
    Collect system-level capability information.

    Returns:
        Dictionary containing system metadata
    """
    system = {
        "python_version": _get_python_version(),
        "platform": _get_platform_info(),
        "memory_usage": _get_memory_info(),
        "uptime": _get_uptime(),
    }

    # Check for optional dependencies
    optional_deps = _check_optional_dependencies()
    if optional_deps:
        system["optional_dependencies"] = optional_deps

    return system


def _get_python_version() -> str:
    """Get Python version string."""
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_platform_info() -> str:
    """Get platform information."""
    import platform

    return f"{platform.system()} {platform.release()}"


def _get_memory_info() -> dict[str, Any] | None:
    """Get memory usage information if available."""
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
        }
    except ImportError:
        return None
    except Exception:
        return None


def _get_uptime() -> float | None:
    """Get system uptime if available."""
    try:
        import psutil

        return psutil.boot_time()
    except ImportError:
        return None
    except Exception:
        return None


def _check_optional_dependencies() -> dict[str, bool]:
    """Check availability of optional dependencies."""
    deps = {}

    optional_modules = [
        "redis",
        "psutil",
        "yaml",
        "requests",
        "beautifulsoup4",
        "selenium",
    ]

    for module in optional_modules:
        try:
            __import__(module)
            deps[module] = True
        except ImportError:
            deps[module] = False

    return deps
