"""
Tools registry with Puter integration.
"""

import logging
from typing import Any, Dict, List, Optional

from .plugin_registry import PluginRegistry
from .puter_plugin import PuterPlugin, PuterAPIError

logger = logging.getLogger(__name__)


class PuterTool:
    """Tool for interacting with Puter cloud environment."""

    def __init__(self, plugin_registry: PluginRegistry):
        self.plugin_registry = plugin_registry
        self.puter_plugin: Optional[PuterPlugin] = None

    async def initialize(self) -> None:
        self.puter_plugin = self.plugin_registry.get_plugin("puter")
        if not self.puter_plugin:
            raise RuntimeError("Puter plugin not initialized in registry")

    async def execute_command(
        self,
        command: str,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not self.puter_plugin:
            raise RuntimeError("PuterTool not initialized")
        try:
            result = await self.puter_plugin.execute_command(
                command=command, args=args, cwd=cwd, env=env
            )
            await self._emit_event(
                "command_executed",
                {
                    "command": command,
                    "args": args,
                    "exit_code": result["exit_code"],
                    "execution_time": result["execution_time"],
                },
            )
            return result
        except PuterAPIError as exc:
            logger.error("Puter command execution failed: %s", exc)
            await self._emit_event(
                "command_failed",
                {"command": command, "error": str(exc)},
            )
            raise

    async def read_file(self, path: str) -> str:
        if not self.puter_plugin:
            raise RuntimeError("PuterTool not initialized")
        try:
            content = await self.puter_plugin.read_file(path)
            await self._emit_event(
                "file_read", {"path": path, "size": len(content)}
            )
            return content
        except PuterAPIError as exc:
            logger.error("Failed to read file %s: %s", path, exc)
            await self._emit_event(
                "file_read_failed", {"path": path, "error": str(exc)}
            )
            raise

    async def write_file(
        self, path: str, content: str, create_dirs: bool = True
    ) -> bool:
        if not self.puter_plugin:
            raise RuntimeError("PuterTool not initialized")
        try:
            result = await self.puter_plugin.write_file(
                path, content, create_dirs
            )
            await self._emit_event(
                "file_written",
                {"path": path, "size": len(content), "created_dirs": create_dirs},
            )
            return result
        except PuterAPIError as exc:
            logger.error("Failed to write file %s: %s", path, exc)
            await self._emit_event(
                "file_write_failed", {"path": path, "error": str(exc)}
            )
            raise

    async def list_directory(self, path: str = ".") -> List[Dict[str, Any]]:
        if not self.puter_plugin:
            raise RuntimeError("PuterTool not initialized")
        try:
            items = await self.puter_plugin.list_directory(path)
            await self._emit_event(
                "directory_listed",
                {"path": path, "item_count": len(items)},
            )
            return items
        except PuterAPIError as exc:
            logger.error("Failed to list directory %s: %s", path, exc)
            await self._emit_event(
                "directory_list_failed", {"path": path, "error": str(exc)}
            )
            raise

    async def change_directory(self, path: str) -> str:
        if not self.puter_plugin:
            raise RuntimeError("PuterTool not initialized")
        try:
            old = self.puter_plugin.get_current_directory()
            new_dir = await self.puter_plugin.change_directory(path)
            await self._emit_event(
                "directory_changed",
                {"old_path": old, "new_path": new_dir},
            )
            return new_dir
        except PuterAPIError as exc:
            logger.error("Failed to change directory to %s: %s", path, exc)
            await self._emit_event(
                "directory_change_failed", {"path": path, "error": str(exc)}
            )
            raise

    def get_current_directory(self) -> str:
        if not self.puter_plugin:
            raise RuntimeError("PuterTool not initialized")
        return self.puter_plugin.get_current_directory()

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        logger.info("Event: %s - %s", event_type, data)


class ToolsRegistry:
    """Registry for computational environment tools."""

    def __init__(self, plugin_registry: PluginRegistry):
        self.plugin_registry = plugin_registry
        self.tools: Dict[str, Any] = {}

    async def initialize_tools(self) -> None:
        if self.plugin_registry.get_plugin("puter"):
            puter_tool = PuterTool(self.plugin_registry)
            await puter_tool.initialize()
            self.tools["puter"] = puter_tool
            logger.info("Initialized Puter tool")

    def get_tool(self, tool_name: str) -> Optional[Any]:
        return self.tools.get(tool_name)

    def list_available_tools(self) -> List[str]:
        return list(self.tools.keys())
