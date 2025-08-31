#!/usr/bin/env python3
"""
Export core Mangle abilities to the MCP-Box for training-free distillation.

This script writes reusable MCP tool specs for Mangle into `.mcp_box/` and
regenerates the catalog via the abstractor. Student agents can then discover
and call these tools directly by name (e.g., `mangle_query`).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.abilities.mangle.register import export_mangle_to_mcp_box, export_mangle_rules_to_mcp_box


def main() -> None:
    base = export_mangle_to_mcp_box(".mcp_box")
    rules = export_mangle_rules_to_mcp_box(".mcp_box")
    print(json.dumps({"status": "ok", "base": base, "rules": rules}, indent=2))


if __name__ == "__main__":
    main()
