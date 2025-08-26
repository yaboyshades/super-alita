import asyncio
import importlib
import json
import sys
from pathlib import Path


class ToolForge:
    """Minimal tool generator used for tests."""

    def __init__(self, tools_dir: Path) -> None:
        self.tools_dir = tools_dir

    def generate(self, name: str) -> Path:
        code = (
            "from mcp_server.server import app\n"
            f"@app.tool(name='{name}', description='generated tool')\n"
            f"async def {name}() -> dict[str, str]:\n"
            "    return {'content': 'x' * 250000}\n"
        )
        path = self.tools_dir / f"{name}.py"
        path.write_text(code)
        return path


def test_generated_tool(tmp_path, monkeypatch) -> None:
    tools_dir = Path("mcp_server/src/mcp_server/tools").resolve()
    forge = ToolForge(tools_dir)
    module_path = forge.generate("generated_tool")
    telemetry_file = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("SUPER_ALITA_TELEMETRY_FILE", str(telemetry_file))
    try:

        sys.path.insert(0, str(Path("mcp_server/src").resolve()))
        import mcp_server_wrapper  # patch FastMCP.tool for telemetry
        importlib.invalidate_caches()
        import mcp_server.server  # ensure server imports
        module = importlib.import_module("mcp_server.tools.generated_tool")
        asyncio.run(module.generated_tool())

        events = [json.loads(line) for line in telemetry_file.read_text().splitlines()]
        kinds = {e["type"] for e in events}
        assert {"AbilityCalled", "AbilitySucceeded", "ArtifactCreated"} <= kinds
    finally:
        if module_path.exists():
            module_path.unlink()
        sys.modules.pop("mcp_server.tools.generated_tool", None)
        sys.modules.pop("mcp_server.server", None)
        sys.modules.pop("mcp_server", None)
        path = str(Path("mcp_server/src").resolve())
        if path in sys.path:
            sys.path.remove(path)
