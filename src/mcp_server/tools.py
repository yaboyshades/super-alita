from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .ast_utils import rewrite_function_to_result


async def _run(cmd: list[str], cwd: Path | None = None) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return out.decode(), err.decode(), proc.returncode


async def refactor_to_result(
    file_path: str, function_name: str, dry_run: bool = True
) -> dict[str, Any]:
    p = Path(file_path).resolve()
    if not p.exists() or p.suffix != ".py":
        return {"applied": False, "diff": "", "error": "Invalid Python file path."}

    original = p.read_text(encoding="utf-8")
    rewritten, diff = rewrite_function_to_result(original, function_name=function_name)
    if not rewritten:
        return {
            "applied": False,
            "diff": diff or "",
            "error": "Function not found or transform failed.",
        }

    if not dry_run:
        p.write_text(rewritten, encoding="utf-8")
    return {"applied": not dry_run, "diff": diff or ""}


async def format_and_lint(target_path: str) -> dict[str, str]:
    # Ruff fix first, then Black
    out1, err1, _ = await _run(["python", "-m", "ruff", "check", target_path, "--fix"])
    out2, err2, _ = await _run(["python", "-m", "black", target_path])
    return {"stdout": out1 + out2, "stderr": err1 + err2}


async def find_missing_docstrings(
    root: str, include_tests: bool = False
) -> dict[str, Any]:
    base = Path(root).resolve()
    if not base.exists():
        return {"functions": [], "count": 0, "error": "Root path does not exist."}

    results: list[dict[str, Any]] = []
    for py in base.rglob("*.py"):
        if not include_tests and ("tests" in py.parts):
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            s = line.strip()
            if s.startswith("def ") and "def __" not in s:
                # look ahead a few lines for a docstring
                snippet = "\n".join(lines[i : i + 5])
                if '"""' not in snippet and "'''" not in snippet:
                    name = s.split("(")[0].replace("def", "").strip()
                    results.append({"file": str(py), "line": i, "name": name})
    return {"functions": results, "count": len(results)}
