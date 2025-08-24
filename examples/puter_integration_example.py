"""
Example of using Puter integration with agent framework.
"""

import asyncio
import logging

from config.puter_config import PUTER_CONFIG
from src.puter.plugin_registry import PluginRegistry
from src.puter.tools_registry import ToolsRegistry


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    registry = PluginRegistry()
    await registry.initialize_plugin("puter", PUTER_CONFIG)
    tools_registry = ToolsRegistry(registry)
    await tools_registry.initialize_tools()
    puter_tool = tools_registry.get_tool("puter")
    if puter_tool:
        print("Current directory:", puter_tool.get_current_directory())
        files = await puter_tool.list_directory()
        print("Files:", [f["name"] for f in files])
        await puter_tool.write_file("test_file.txt", "Hello from agent!")
        content = await puter_tool.read_file("test_file.txt")
        print("File content:", content)
        result = await puter_tool.execute_command("ls", ["-la"])
        print("Command output:", result["stdout"])
    await registry.cleanup_all()


if __name__ == "__main__":
    asyncio.run(main())
