from __future__ import annotations

import argparse
import logging
from typing import Any

from .mcp_server import app, register_github_tools


class SuperAlitaMCPServer:
    """Thin wrapper that wires the FastMCP app into the Super Alita runtime."""

    def __init__(self, *, debug: bool = False) -> None:
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("mcp.server.main")
        self.app = app
        self.github_tools = register_github_tools(self.app)

    def warmup(self) -> None:
        """Perform lightweight health checks before starting the server."""

        resources = self.github_tools.list_resources()
        self.logger.debug("Registered GitHub resources: %s", resources)

    def run(self, *, transport: str = "stdio") -> None:
        self.logger.info("Starting MCP server (transport=%s)", transport)
        self.app.run(transport=transport)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Super Alita MCP Server")
    parser.add_argument("--transport", choices=["stdio"], default="stdio")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    return parser


def main(argv: Any | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    server = SuperAlitaMCPServer(debug=args.debug)
    server.warmup()
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
