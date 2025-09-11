#!/usr/bin/env python3
"""Export Mangle abilities to MCP-Box specs for IDE discovery.

Writes JSON specs into `.mcp_box/` and rebuilds the index using the
repo's abstractor. Safe to run multiple times.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(dir_path: str) -> int:
    # Ensure repo root on sys.path for `src.*` imports
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from src.abilities.mangle.register import export_mangle_to_mcp_box
    except Exception as e:  # pragma: no cover - import error path
        print(json.dumps({"ok": False, "error": f"import_failed: {e}"}))
        return 1

    try:
        res = export_mangle_to_mcp_box(dir_=dir_path)  # type: ignore[arg-type]
    except TypeError:
        # Older signature without keyword
        res = export_mangle_to_mcp_box(dir_path)  # type: ignore[misc]
    print(json.dumps({"ok": True, **res}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Mangle MCP specs")
    parser.add_argument("--dir", default=".mcp_box", help="Output directory")
    args = parser.parse_args()
    raise SystemExit(main(args.dir))
