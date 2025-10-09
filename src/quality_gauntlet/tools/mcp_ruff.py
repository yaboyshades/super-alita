"""MCP wrapper for Ruff linting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field  # type: ignore[import-not-found]

from . import CommandRunner
from ._runner import DEFAULT_RUNNER


class RuffFinding(BaseModel):
    """Single Ruff linting violation."""

    code: str
    message: str
    line: int
    column: int


class RuffResult(BaseModel):
    """Aggregated Ruff linting output."""

    findings: list[RuffFinding] = Field(default_factory=list)


class RuffMCPTool:
    """Invoke Ruff CLI on temporary files."""

    def __init__(self, *, runner: CommandRunner | None = None) -> None:
        self._runner = runner or DEFAULT_RUNNER

    async def lint_code(self, code: str) -> RuffResult:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "lint.py"
            tmp_path.write_text(code, encoding="utf-8")

            result = await self._runner(
                ["ruff", "check", "--output-format=json", str(tmp_path)],
                cwd=tmp_dir,
            )
            payload = json.loads(result.stdout or "[]")
            findings = [
                RuffFinding(
                    code=item["code"],
                    message=item["message"],
                    line=item["location"]["row"],
                    column=item["location"]["column"],
                )
                for item in payload
            ]
            return RuffResult(findings=findings)
