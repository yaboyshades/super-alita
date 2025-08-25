from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys

from mcp.server.fastmcp import FastMCP


# Dynamic tool loader: import all modules in mcp_server.tools
def load_tools() -> list[str]:
    imported = []
    import mcp_server.tools as tools_pkg

    for mod in pkgutil.iter_modules(tools_pkg.__path__, tools_pkg.__name__ + "."):
        imported.append(mod.name)
        importlib.import_module(mod.name)
    return imported


app = FastMCP("myCustomPythonAgent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    args = parser.parse_args()

    loaded = load_tools()
    # Register tools decorated with @app.tool() in loaded modules
    # FastMCP auto-discovers @app.tool() methods defined with the same app instance.
    # Ensure your tool modules import pp from this module: rom mcp_server.server import app
    if args.transport == "stdio":
        app.run(transport="stdio")
    else:
        print("SSE transport not configured", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
