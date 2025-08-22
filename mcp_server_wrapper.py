#!/usr/bin/env python3
import sys
import os

# Remove the workspace from the Python path to avoid conflicts
sys.path = [p for p in sys.path if 'super-alita-clean' not in p and 'ATLAI' not in p]

# Add the virtual environment paths explicitly
venv_base = r'D:\Coding_Projects\super-alita-clean\.venv'
sys.path.extend([
    os.path.join(venv_base, 'Lib', 'site-packages'),
    os.path.join(venv_base, 'Lib', 'site-packages', 'win32'),
    os.path.join(venv_base, 'Lib', 'site-packages', 'win32', 'lib'),
    os.path.join(venv_base, 'Lib', 'site-packages', 'Pythonwin'),
])

# Import the MCP modules first
from mcp.server.fastmcp import FastMCP

# Restore the path so we can import local modules
sys.path.insert(0, r'D:\Coding_Projects\super-alita-clean\src')
from mcp_server.tools import find_missing_docstrings, format_and_lint, refactor_to_result

import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

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
