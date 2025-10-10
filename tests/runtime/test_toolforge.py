import asyncio
import importlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "mcp_server" / "src")
)
from mcp_server.server import load_tools


def test_toolforge_creates_and_executes(tmp_path):
    spec = {
        "name": "toolforge_adder",
        "description": "Add two numbers",
        "input_schema": {"a": "int", "b": "int"},
        "code": "return {'sum': a + b}",
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.agents.toolforge",
            "--spec",
            str(spec_path),
        ],
        check=True,
    )

    modules = load_tools()
    assert f"mcp_server.tools.{spec['name']}" in modules

    mod = importlib.import_module(f"mcp_server.tools.{spec['name']}")
    fn = getattr(mod, spec["name"])
    result = asyncio.run(fn(a=2, b=3))
    assert result["sum"] == 5

    Path(mod.__file__).unlink(missing_ok=True)
