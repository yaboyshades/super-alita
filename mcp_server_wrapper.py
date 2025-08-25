#!/usr/bin/env python3
import logging
import os
import sys
from typing import Any

# Remove the workspace from the Python path to avoid conflicts
sys.path = [p for p in sys.path if "super-alita-clean" not in p and "ATLAI" not in p]

# Add the virtual environment paths explicitly
venv_base = r"D:\Coding_Projects\super-alita-clean\.venv"
sys.path.extend(
    [
        os.path.join(venv_base, "Lib", "site-packages"),
        os.path.join(venv_base, "Lib", "site-packages", "win32"),
        os.path.join(venv_base, "Lib", "site-packages", "win32", "lib"),
        os.path.join(venv_base, "Lib", "site-packages", "Pythonwin"),
    ]
)

# Import the MCP modules first
from mcp.server.fastmcp import FastMCP

# Restore the path so we can import local modules
sys.path.insert(0, r"D:\Coding_Projects\super-alita-clean\src")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

# Try to import tools, provide fallbacks if not available
try:
    from mcp_server.tools import (
        find_missing_docstrings,
        format_and_lint,
        refactor_to_result,
    )

    logger.info("Successfully imported MCP tools")
except ImportError as e:
    logger.error(f"Failed to import MCP tools: {e}")

    # Provide fallback implementations
    async def find_missing_docstrings(
        root: str, include_tests: bool = False
    ) -> dict[str, Any]:
        return {"functions": [], "count": 0, "error": "MCP tools not available"}

    async def format_and_lint(target_path: str) -> dict[str, str]:
        return {"stdout": "", "stderr": "MCP tools not available"}

    async def refactor_to_result(
        file_path: str, function_name: str, dry_run: bool = True
    ) -> dict[str, Any]:
        return {"applied": False, "diff": "", "error": "MCP tools not available"}


app = FastMCP("myCustomPythonAgent")


@app.tool(
    name="apply_result_pattern_refactor",
    description=(
        "Refactor a Python function to a Result-returning pattern. "
        "Args: file_path (str), function_name (str), dry_run (bool, default true). "
        "Returns JSON: {'applied': bool, 'diff': str, 'error': Optional[str]}."
    ),
)
async def apply_result_pattern_refactor(
    file_path: str, function_name: str, dry_run: bool = True
) -> dict[str, Any]:
    return await refactor_to_result(
        file_path=file_path, function_name=function_name, dry_run=dry_run
    )


@app.tool(
    name="format_and_lint_selection",
    description=(
        "Run Ruff (fix) and Black on a path. "
        "Args: target_path (str). Returns JSON: {'stdout': str, 'stderr': str}."
    ),
)
async def format_and_lint_selection(target_path: str) -> dict[str, str]:
    return await format_and_lint(target_path=target_path)


@app.tool(
    name="find_missing_docstrings",
    description=(
        "Find functions missing docstrings under a root dir. "
        "Args: root (str), include_tests (bool, default false). "
        "Returns JSON: {'functions': [{'file': str, 'line': int, 'name': str}], 'count': int}."
    ),
)
async def find_missing_docstrings_tool(
    root: str, include_tests: bool = False
) -> dict[str, Any]:
    return await find_missing_docstrings(root=root, include_tests=include_tests)


def main() -> None:
    transport = "stdio"  # VS Code launches this as a subprocess
    logger.info("Starting MCP server (transport=%s)", transport)
    app.run(transport=transport)


if __name__ == "__main__":
    main()

# --- Decision Policy Bridge ---
from src.core.decision_policy_v1 import DecisionPolicyEngine  # noqa: E402
from src.mcp_local.registry import ToolRegistry as LocalToolRegistry  # noqa: E402
from typing import Dict  # noqa: E402


class MCPBridge:
    def __init__(self):
        self.decision_policy = DecisionPolicyEngine()
        self.mcp_registry = LocalToolRegistry()

    async def register_mcp_tools_as_capabilities(self):
        for tool_name in self.mcp_registry.list_tools():
            cap = self._convert_mcp_to_capability(
                {
                    "name": tool_name,
                    "description": f"MCP tool {tool_name}",
                    "inputSchema": {},
                }
            )
            if hasattr(self.decision_policy, "register_capability"):
                self.decision_policy.register_capability(cap)  # type: ignore[attr-defined]

    def _create_mcp_executor(self, name: str):
        async def _exec(**kwargs):
            return await self.mcp_registry.invoke(name, kwargs)

        return _exec

    def _convert_mcp_to_capability(self, tool_spec: Dict) -> Dict:
        return {
            "name": tool_spec["name"],
            "description": tool_spec.get("description", ""),
            "parameters": tool_spec.get("inputSchema", {}),
            "type": "mcp_tool",
            "executor": self._create_mcp_executor(tool_spec["name"]),
        }
