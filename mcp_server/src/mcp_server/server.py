from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Dynamic tool loader: import all modules in mcp_server.tools


def load_tools() -> list[str]:
    imported = []
    import mcp_server.tools as tools_pkg

    for mod in pkgutil.iter_modules(tools_pkg.__path__, tools_pkg.__name__ + "."):
        imported.append(mod.name)
        importlib.import_module(mod.name)
    return imported


def copy_tool_template(name: str, dest_dir: Path | None = None) -> Path:
    """Copy the tool template into the tools package.

    Args:
        name: ``snake_case`` name for the new tool module.
        dest_dir: Optional destination directory; defaults to the tools package.

    Returns:
        Path to the created module.
    """
    template = Path(__file__).resolve().parents[2] / "tools" / "_template.py"
    destination = (
        (Path(__file__).resolve().parent / "tools") if dest_dir is None else dest_dir
    )
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{name}.py"
    content = template.read_text().replace("tool_name", name)
    target.write_text(content)
    return target


app = FastMCP("myCustomPythonAgent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--add-tool", help="Copy template into tools package and exit.")
    args = parser.parse_args()

    if args.add_tool:
        path = copy_tool_template(args.add_tool)
        print(f"created {path}")
        return

    load_tools()
    # Register tools decorated with @app.tool() in loaded modules
    # FastMCP auto-discovers @app.tool() methods defined with the same app instance.
    # Ensure your tool modules import app from this module:
    # from mcp_server.server import app
    if args.transport == "stdio":
        app.run(transport="stdio")
    else:
        print("SSE transport not configured", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
    main()
