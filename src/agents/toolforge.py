from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mcp_server" / "src"))
from mcp_server.server import load_tools


def _sanitize_name(name: str) -> str:
    """Return a valid Python identifier derived from ``name``."""
    return re.sub(r"\W|^(?=\d)", "_", name)


def forge(spec: dict[str, Any]) -> Path:
    """Create a tool module from ``spec`` and reload dynamic tools.

    Args:
        spec: Mapping with ``name``, ``description``, ``input_schema``, and ``code``.

    Returns:
        Path to the generated module.
    """
    tools_dir = ROOT / "mcp_server" / "src" / "mcp_server" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    name = str(spec["name"])
    func_name = _sanitize_name(name)
    description = str(spec.get("description", ""))
    schema = spec.get("input_schema", {})
    if not isinstance(schema, dict):
        raise TypeError("input_schema must be a mapping of arg names to type strings")
    code = str(spec.get("code", ""))

    args_sig = ", ".join(f"{arg}: {typ}" for arg, typ in schema.items())
    body = textwrap.indent(code.strip(), "    ")

    module_content = (
        "from __future__ import annotations\n\n"
        "from typing import Any\n"
        "from mcp_server.server import app\n\n"
        f"@app.tool(\n    name=\"{name}\",\n    description=\"{description}\",\n)\n"
        f"async def {func_name}({args_sig}) -> Any:\n"
        f"{body}\n"
    )

    module_path = tools_dir / f"{func_name}.py"
    module_path.write_text(module_content, encoding="utf-8")

    load_tools()
    return module_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Forge MCP tool modules from a spec file")
    parser.add_argument("--spec", required=True, help="Path to JSON tool specification")
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    forge(spec)


if __name__ == "__main__":
    main()
