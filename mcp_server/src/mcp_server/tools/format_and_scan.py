from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mcp_server.server import app


def _is_subpath(base: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        return False


@app.tool(
    name="format_and_lint_selection",
    description="Run Ruff (fix) then Black on a path. Args: target_path (str). Returns stdout/stderr JSON.",
)
async def format_and_lint_selection(target_path: str) -> dict[str, str]:
    root = Path.cwd().resolve()
    p = Path(target_path).resolve()
    if not _is_subpath(root, p):
        return {"stdout": "", "stderr": "Path outside workspace denied."}

    async def _run(cmd: list[str]) -> tuple[str, str, int]:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        return out.decode(), err.decode(), proc.returncode

    out1, err1, _ = await _run(
        [
            str(
                Path(
                    ".venv/Scripts/python.exe"
                    if (Path(".venv") / "Scripts").exists()
                    else "python"
                )
            ),
            "-m",
            "ruff",
            "check",
            str(p),
            "--fix",
        ]
    )
    out2, err2, _ = await _run(
        [
            str(
                Path(
                    ".venv/Scripts/python.exe"
                    if (Path(".venv") / "Scripts").exists()
                    else "python"
                )
            ),
            "-m",
            "black",
            str(p),
        ]
    )
    return {"stdout": out1 + out2, "stderr": err1 + err2}


@app.tool(
    name="find_missing_docstrings",
    description="Scan *.py for functions missing docstrings. Args: root (str), include_tests (bool=false).",
)
async def find_missing_docstrings(
    root: str, include_tests: bool = False
) -> dict[str, Any]:
    base = Path(root).resolve()
    ws = Path.cwd().resolve()
    if not base.exists() or not _is_subpath(ws, base):
        return {"functions": [], "count": 0, "error": "Invalid or unsafe root"}
    results = []
    for py in base.rglob("*.py"):
        if not include_tests and "tests" in py.parts:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            s = line.strip()
            if s.startswith("def ") and "def __" not in s:
                look = "\n".join(lines[i : i + 5])
                if '"""' not in look and "'''" not in look:
                    name = s.split("(")[0].replace("def", "").strip()
                    results.append({"file": str(py), "line": i, "name": name})
    return {"functions": results, "count": len(results)}
