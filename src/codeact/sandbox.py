"""Sandboxed Python execution for CodeAct."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """Raised when execution violates sandbox constraints."""


@dataclass(slots=True)
class SandboxResult:
    stdout: str
    stderr: str
    error: str | None
    locals: dict[str, Any]


class PythonSandbox:
    """Execute Python code with basic safety guardrails."""

    def __init__(
        self,
        allowed_imports: Iterable[str] | None = None,
        timeout: float = 5.0,
        stdout_limit: int = 10_000,
        stderr_limit: int = 10_000,
        workdir: str | None = None,
    ) -> None:
        self.allowed_imports = set(allowed_imports or {"math", "json", "csv", "statistics"})
        self.timeout = timeout
        self.stdout_limit = stdout_limit
        self.stderr_limit = stderr_limit
        self.workdir = workdir
        self._globals = self._build_globals()

    # ------------------------------------------------------------------
    def _build_globals(self) -> dict[str, Any]:
        allowed_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "float": float,
            "int": int,
            "zip": zip,
            "open": open,
        }

        def _import(name: str, globals: Any = None, locals: Any = None, fromlist: tuple[str, ...] = (), level: int = 0):
            if name in self.allowed_imports:
                return __import__(name, globals, locals, fromlist, level)
            raise SandboxError(f"import of '{name}' is not allowed")

        builtins: dict[str, Any] = dict(allowed_builtins)
        builtins["__import__"] = _import
        return {"__builtins__": builtins}

    # ------------------------------------------------------------------
    async def run(self, code: str) -> SandboxResult:
        """Execute code asynchronously with timeouts and IO capture."""

        stdout = io.StringIO()
        stderr = io.StringIO()
        globals_ns: dict[str, Any] = dict(self._globals)

        def _execute() -> SandboxResult:
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(code, "<sandbox>", "exec"), globals_ns, globals_ns)
                return SandboxResult(
                    stdout.getvalue()[-self.stdout_limit :],
                    stderr.getvalue()[-self.stderr_limit :],
                    None,
                    {k: v for k, v in globals_ns.items() if k != "__builtins__"},
                )
            except Exception as exc:  # noqa: BLE001 - capture for observation
                return SandboxResult(
                    stdout.getvalue()[-self.stdout_limit :],
                    stderr.getvalue()[-self.stderr_limit :],
                    repr(exc),
                    {k: v for k, v in globals_ns.items() if k != "__builtins__"},
                )

        try:
            if self.workdir:
                import os

                cwd = os.getcwd()
                os.makedirs(self.workdir, exist_ok=True)
                os.chdir(self.workdir)
                try:
                    return await asyncio.wait_for(asyncio.to_thread(_execute), self.timeout)
                finally:
                    os.chdir(cwd)
            return await asyncio.wait_for(asyncio.to_thread(_execute), self.timeout)
        except asyncio.TimeoutError:
            logger.warning("Sandbox execution timed out")
            return SandboxResult("", "", "timeout", {})
