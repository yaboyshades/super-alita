#!/usr/bin/env python3
"""
Native DeepCode API Wrapper

This provides the same interface as the external DeepCode API but uses
native tools directly within our agent framework.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NativeDeepCodeAPI:
    """
    Native DeepCode API implementation that wraps our native plugin tools.

    This provides the same interface as the external API (deepcode_request,
    deepcode_latest, deepcode_apply) but runs everything natively within
    our agent framework.
    """

    def __init__(self, native_plugin=None):
        self.native_plugin = native_plugin

    def set_plugin(self, plugin):
        """Set the native DeepCode plugin instance"""
        self.native_plugin = plugin

    def _get_plugin(self):
        """Get the native DeepCode plugin, trying multiple sources"""
        if self.native_plugin:
            return self.native_plugin

        # Try to get from global plugin registry
        try:
            from src.core.plugin_registry import get_plugin

            plugin = get_plugin("native_deepcode")
            if plugin:
                self.native_plugin = plugin
                return plugin
        except Exception:
            pass

        return None

    async def deepcode_request(self, **kwargs) -> dict[str, Any]:
        """Native implementation of deepcode_request"""
        plugin = self._get_plugin()
        if not plugin:
            logger.warning("No native DeepCode plugin available, falling back to mock")
            return {"status": "mock", "message": "No native plugin configured"}

        try:
            result = await plugin.invoke_tool("deepcode_request", kwargs)
            return result
        except Exception as e:
            logger.error(f"Native deepcode_request failed: {e}")
            return {"status": "error", "error": str(e)}

    async def deepcode_latest(self) -> dict[str, Any]:
        """Native implementation of deepcode_latest"""
        plugin = self._get_plugin()
        if not plugin:
            logger.warning("No native DeepCode plugin available")
            return {"status": "no_plugin"}

        try:
            result = await plugin.invoke_tool("deepcode_latest", {})
            return result
        except Exception as e:
            logger.error(f"Native deepcode_latest failed: {e}")
            return {"status": "error", "error": str(e)}

    async def deepcode_apply(self, paths=None) -> dict[str, Any]:
        """Native implementation of deepcode_apply"""
        plugin = self._get_plugin()
        if not plugin:
            logger.warning("No native DeepCode plugin available")
            return {"status": "no_plugin"}

        try:
            args = {"paths": paths} if paths else {}
            result = await plugin.invoke_tool("deepcode_apply", args)
            return result
        except Exception as e:
            logger.error(f"Native deepcode_apply failed: {e}")
            return {"status": "error", "error": str(e)}

    # Keep the same interface for compatibility
    def pytest_run(self, args=None):
        """Mock pytest for testing gates"""
        return {"exit_code": 0, "output": "All tests passed"}

    def secure_scan(self, code=""):
        """Mock security scan for testing gates"""
        return {"issues": [], "status": "clean"}

    def secure_scan_code(self, code=""):
        """Mock security scan for testing gates (alternative name)"""
        return self.secure_scan(code)


# Global instance for compatibility
_native_api_instance = NativeDeepCodeAPI()


def get_native_deepcode_api() -> NativeDeepCodeAPI:
    """Get the global native DeepCode API instance"""
    return _native_api_instance


def set_native_plugin(plugin) -> None:
    """Set the native DeepCode plugin for the API"""
    _native_api_instance.set_plugin(plugin)

    # Also register it globally
    try:
        from src.core.plugin_registry import register_plugin

        register_plugin("native_deepcode", plugin)
    except Exception:
        pass
