from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.core.proc import ProcError
from src.core.proc import run as run_proc


def _venv_python(venv_dir: Path) -> str:
    win = os.name == "nt"
    cand = (
        venv_dir / "Scripts" / "python.exe"
        if win
        else venv_dir / "bin" / "python"
    )
    return str(cand)


def run_in_temp_venv(code: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """
    Create a temporary venv, run a short Python snippet, and return outputs.
    Falls back to system python if venv creation is unavailable.
    """
    # Guardrails: keep code short to reduce risk
    if len(code) > 4000:
        return {
            "stdout": "",
            "stderr": "code too large; max 4000 chars",
            "returncode": 1,
            "used_venv": False,
        }

    tmpdir = Path(tempfile.mkdtemp(prefix="alita_sbx_"))
    used_venv = False
    try:
        # Try to create a virtual environment
        try:
            run_proc(
                [sys.executable, "-m", "venv", str(tmpdir / "venv")],
                timeout=timeout,
            )
            py = _venv_python(tmpdir / "venv")
            used_venv = True
        except Exception:
            # Fallback: system python
            py = sys.executable

        # Write a temporary script file
        script = tmpdir / "snippet.py"
        script.write_text(code, encoding="utf-8")

        # Execute
        try:
            out = run_proc([py, str(script)], timeout=timeout)
            return {
                "stdout": out,
                "stderr": "",
                "returncode": 0,
                "used_venv": used_venv,
            }
        except ProcError as e:
            return {
                "stdout": e.stdout,
                "stderr": e.stderr,
                "returncode": e.returncode,
                "used_venv": used_venv,
            }
    finally:
        # Best-effort cleanup
        with contextlib.suppress(Exception):  # type: ignore[name-defined]
            shutil.rmtree(tmpdir, ignore_errors=True)
