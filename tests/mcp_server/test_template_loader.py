from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mcp_server" / "src"))

from mcp_server.server import copy_tool_template


def test_copy_tool_template(tmp_path: Path) -> None:
    dest = tmp_path / "tools"
    created = copy_tool_template("sample_tool", dest)
    assert created.exists()
    assert created.read_text().count("sample_tool") >= 2

