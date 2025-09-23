from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


class ProcError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, stdout: str, stderr: str):
        super().__init__(f"Command failed ({returncode}): {' '.join(cmd)}")
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _sanitize_args(args: Sequence[str]) -> list[str]:
    bad = {"\n", "\r", "\x00"}
    out: list[str] = []
    for a in args:
        if any(c in a for c in bad):
            raise ValueError("Illegal control character in argument")
        out.append(a)
    return out


def run(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> str:
    """
    Run command with sanitized arguments. shell=False enforced.
    Returns stdout (text). Raises ProcError on non-zero exit.
    """
    argv = _sanitize_args(cmd)
    p = subprocess.run(
        argv,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(env) if env else None,
        cwd=str(cwd) if cwd is not None else None,
    )
    if p.returncode != 0:
        raise ProcError(argv, p.returncode, p.stdout, p.stderr)
    return p.stdout


async def arun(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> str:
    """
    Async counterpart to run(). Executes subprocess.run in a thread so existing
    monkeypatching of subprocess.run in tests still applies. Returns stdout.
    Raises ProcError on non-zero exit.
    """
    argv = _sanitize_args(cmd)

    def _call() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(env) if env else None,
            cwd=str(cwd) if cwd is not None else None,
        )

    p: subprocess.CompletedProcess[str] = await asyncio.to_thread(_call)
    if p.returncode != 0:
        raise ProcError(argv, p.returncode, p.stdout, p.stderr)
    return p.stdout
