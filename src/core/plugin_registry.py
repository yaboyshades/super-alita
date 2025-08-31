#!/usr/bin/env python3
"""
Global Plugin Registry

This module provides a way to register and access plugin instances
across different parts of the application, including when running
pipelines independently of the FastAPI app.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Global plugin registry
_plugin_registry: dict[str, Any] = {}


def register_plugin(name: str, plugin: Any) -> None:
    """Register a plugin instance globally"""
    global _plugin_registry
    _plugin_registry[name] = plugin
    logger.info(f"Registered plugin: {name}")


def get_plugin(name: str) -> Any | None:
    """Get a plugin instance by name"""
    global _plugin_registry
    return _plugin_registry.get(name)


def list_plugins() -> list[str]:
    """List all registered plugin names"""
    global _plugin_registry
    return list(_plugin_registry.keys())


def clear_registry() -> None:
    """Clear all registered plugins"""
    global _plugin_registry
    _plugin_registry.clear()


def has_plugin(name: str) -> bool:
    """Check if a plugin is registered"""
    global _plugin_registry
    return name in _plugin_registry
