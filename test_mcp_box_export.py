#!/usr/bin/env python3
"""Test exporting Mangle tools to MCP-Box specs."""

from __future__ import annotations

import sys
from pathlib import Path


def test_export_mangle_to_mcp_box(tmp_path: Path) -> None:
    # Ensure repo root on path
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from src.abilities.mangle.register import export_mangle_to_mcp_box

    out_dir = tmp_path / ".mcp_box"
    res = export_mangle_to_mcp_box(dir_=str(out_dir))  # type: ignore[arg-type]
    assert res.get("catalog", 0) >= 2
    assert any(p.name.endswith(".json") for p in out_dir.glob("*.json"))

