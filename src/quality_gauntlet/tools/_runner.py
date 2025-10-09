"""Async command runner utility for quality gauntlet tools."""

from __future__ import annotations

import asyncio
from asyncio.subprocess import PIPE
from collections.abc import Iterable

from . import CommandResult, CommandRunner


def _sanitize(argv: Iterable[str]) -> list[str]:
    """Ensure arguments do not contain control characters."""

    cleaned: list[str] = []
    illegal = {"\n", "\r", "\x00"}
    for raw in argv:
        if any(char in raw for char in illegal):  # noqa: PERF402 - small inputs
            raise ValueError("Illegal control character in argument")
        cleaned.append(raw)
    return cleaned


async def run_command(
    argv: list[str],
    *,
    cwd: str | None = None,
    timeout: float | None = None,
) -> CommandResult:
    """Execute command asynchronously with cooperative cancellation."""

    args = _sanitize(argv)
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdout=PIPE,
        stderr=PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except TimeoutError as exc:  # pragma: no cover - defensive branch
        process.kill()
        raise TimeoutError(
            f"Command timed out after {timeout}s: {' '.join(args)}"
        ) from exc
    return CommandResult(
        returncode=int(process.returncode or 0),
        stdout=stdout_bytes.decode("utf-8", errors="ignore"),
        stderr=stderr_bytes.decode("utf-8", errors="ignore"),
    )


# Expose runner as CommandRunner compatible callable
DEFAULT_RUNNER: CommandRunner = run_command
