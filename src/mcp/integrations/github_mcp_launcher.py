"""Command-line launcher for the constitutional GitHub MCP integration."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from src.mcp.integrations.github_mcp_integration import GitHubMCPIntegration


async def _bootstrap(
    threshold: float,
) -> tuple[GitHubMCPIntegration, dict[str, Any]]:
    integration = GitHubMCPIntegration(constitutional_threshold=threshold)
    status = await integration.ensure_initialized()
    return integration, {
        "valid": status.valid,
        "constitutional_score": status.constitutional_score,
        "token_present": status.token_present,
        "resource_count": status.details.get("resource_count", 0),
        "violations": status.violations,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Super-Alita GitHub MCP server"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum constitutional score",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio"],
        help="Transport passed to FastMCP",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Validate integration and exit without starting the server",
    )
    args = parser.parse_args(argv)

    integration, status = asyncio.run(_bootstrap(args.threshold))
    print(json.dumps({"github_mcp_status": status}, indent=2))

    if args.init_only:
        return

    if integration.server is None:
        raise RuntimeError("GitHub MCP server not initialized")
    integration.server.run(transport=args.transport)


if __name__ == "__main__":
    main()
