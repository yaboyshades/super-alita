import json
import shutil
import subprocess
from pathlib import Path

import pytest

PWSH = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(PWSH is None, reason="pwsh not installed")


def test_setup_mcp_adds_server(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    script = repo / "tools" / "Setup-MCP.ps1"
    config = tmp_path / "mcp.json"
    sample = {"servers": {}}
    config.write_text(json.dumps(sample))

    subprocess.run(
        [PWSH, str(script), "-AddTool", "Example", "-ConfigPath", str(config)],
        check=True,
    )

    data = json.loads(config.read_text())
    assert "Example" in data["servers"]
