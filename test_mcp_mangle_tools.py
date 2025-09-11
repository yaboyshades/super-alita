#!/usr/bin/env python3
"""
MCP-side smoke tests for Mangle tools that do not require the external
Mangle binary. We avoid calling the heavy 'mangle_query' to keep tests
fast and environment-agnostic.
"""

import asyncio
import sys
from pathlib import Path

# Ensure repo root on path and import the MCP app
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from mcp_server_wrapper import app  # type: ignore


async def _run():
    # 1) Add a fact
    res_fact = await app._handle_call("mangle_add_fact", {"fact": "project('demo')"})
    assert res_fact.get("success"), res_fact

    # 2) Add a rule (does not require the mangle binary)
    res_rule = await app._handle_call(
        "mangle_add_rule",
        {
            "name": "demo_rule",
            "rule": "related('demo', 'artifact') :- project('demo')",
        },
    )
    assert res_rule.get("success"), res_rule

    # 3) List rules via MCP
    res_catalog = await app._handle_call("mangle_rule_catalog", {})
    assert res_catalog.get("success"), res_catalog
    assert res_catalog.get("count", 0) >= 1


def test_mcp_mangle_smoke():
    asyncio.run(_run())

