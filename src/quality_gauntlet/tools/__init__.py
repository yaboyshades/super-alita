"""Tool adapters for external quality services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class CommandResult:
    """Outcome from running an external command."""

    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """Protocol for async command execution."""

    async def __call__(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Execute a command and return its captured output."""
