#!/usr/bin/env python3
"""
Unified project runner for Super Alita.

Runs common end-to-end steps with one command:
  1) Ensure .env exists
  2) Install dependencies (uv if available, else pip)
  3) Lint/format (pre-commit or ruff/black fallback)
  4) Type-check (mypy; focuses on src/core and src/sandbox)
  5) Tests (pytest; full suite by default)
  6) Optional: launch API server (uvicorn app:app) and verify health

Usage examples:
  python scripts/run_all.py                  # run all checks
  python scripts/run_all.py --skip-deps      # skip dependency installation
  python scripts/run_all.py --serve          # also start server and health-check
  python scripts/run_all.py --serve --keep   # keep server running after check
  python scripts/run_all.py --mode act       # set SUPER_ALITA_MODE for server

This script uses src/core/proc.py to run subprocesses without shell=True.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ensure project root is importable so `import src.*` works
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Strictly follow repository policy: use proc.py for subprocesses
try:
    from src.core import proc
except Exception as e:  # pragma: no cover - fail early with clear message
    print(f"[run_all] ERROR: failed to import src.core.proc: {e}")
    sys.exit(2)


def in_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix or hasattr(
        sys, "real_prefix"
    )


def ensure_env_file() -> None:
    env = ROOT / ".env"
    if not env.exists():
        example = ROOT / ".env.example"
        if example.exists():
            env.write_text(example.read_text(), encoding="utf-8")
            print("[run_all] Created .env from .env.example")
        else:
            env.write_text("", encoding="utf-8")
            print("[run_all] Created empty .env (no .env.example found)")


def which(cmd: str) -> bool:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    exts = [""]
    if os.name == "nt":
        pathext = os.environ.get("PATHEXT", ".EXE;.BAT;.CMD").split(";")
        exts = pathext + exts
    for p in paths:
        full = Path(p) / cmd
        for ext in exts:
            candidate = full.with_suffix(ext) if ext and not full.suffix else full
            if candidate.exists() and candidate.is_file():
                return True
    return False


def install_deps(use_uv: bool | None = None) -> None:
    # Prefer uv if present (fast), else pip with constraints
    if use_uv is None:
        use_uv = which("uv")

    if use_uv:
        print("[run_all] Installing deps via uv...")
        constraints = ROOT / "constraints.txt"
        args_base = ["uv", "pip", "install"]
        if constraints.exists():
            proc.run(args_base + ["-r", "requirements.txt", "-c", "constraints.txt"])
            # test deps
            if (ROOT / "requirements-test.txt").exists():
                proc.run(
                    args_base + ["-r", "requirements-test.txt", "-c", "constraints.txt"]
                )
        else:
            proc.run(args_base + ["-r", "requirements.txt"])
            if (ROOT / "requirements-test.txt").exists():
                proc.run(args_base + ["-r", "requirements-test.txt"])
    else:
        print("[run_all] Installing deps via pip...")
        constraints = ROOT / "constraints.txt"
        args_base = [sys.executable, "-m", "pip", "install"]
        if constraints.exists():
            proc.run(args_base + ["-r", "requirements.txt", "-c", "constraints.txt"])
            if (ROOT / "requirements-test.txt").exists():
                proc.run(
                    args_base + ["-r", "requirements-test.txt", "-c", "constraints.txt"]
                )
        else:
            proc.run(args_base + ["-r", "requirements.txt"])
            if (ROOT / "requirements-test.txt").exists():
                proc.run(args_base + ["-r", "requirements-test.txt"])


def run_precommit_or_fallback(fix: bool) -> None:
    # Try pre-commit first
    if which("pre-commit"):
        print("[run_all] Running pre-commit hooks...")
        proc.run(["pre-commit", "run", "--all-files"])
        return
    # Fallback: ruff + black
    ruff_cmd = ["ruff", "check", "."] + (["--fix"] if fix else [])
    black_cmd = (
        ["black", ".", "-l", "88"] if fix else ["black", "--check", ".", "-l", "88"]
    )
    print("[run_all] pre-commit not found; using ruff + black fallback")
    proc.run(ruff_cmd)
    proc.run(black_cmd)


def run_mypy() -> None:
    # Focus on src/core and src/sandbox per guidelines, include app.py
    targets: list[str] = []
    for p in (ROOT / "src" / "core", ROOT / "src" / "sandbox"):
        if p.exists():
            targets.append(str(p))
    if (ROOT / "app.py").exists():
        targets.append("app.py")
    if not targets:
        print("[run_all] Skipping mypy (no target paths found)")
        return
    print(f"[run_all] Running mypy on: {' '.join(targets)}")
    proc.run(["mypy", "--strict", *targets])


def run_pytest(extra_args: Iterable[str] | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    args = ["pytest", "-q"]
    if extra_args:
        args.extend(list(extra_args))
    print("[run_all] Running tests: pytest -q")
    out = proc.run(args, env=env)
    sys.stdout.write(out)


def wait_for_http(url: str, timeout_s: float = 20.0) -> bool:
    try:
        import httpx
    except Exception as e:  # pragma: no cover
        print(f"[run_all] httpx not available for health check: {e}")
        return False
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code < 500:
                return True
        except Exception as e:  # pragma: no cover - transient
            last_err = e
        time.sleep(0.5)
    if last_err:
        print(f"[run_all] Health check wait error: {last_err}")
    return False


def run_server_and_check(host: str, port: int, mode: str | None, keep: bool) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    if mode:
        env["SUPER_ALITA_MODE"] = mode
    print(
        f"[run_all] Starting server at http://{host}:{port} (mode={env.get('SUPER_ALITA_MODE','') or 'default'})"
    )

    # Start uvicorn app:app
    from subprocess import DEVNULL, Popen

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    proc_handle = Popen(cmd, env=env, stdout=DEVNULL, stderr=DEVNULL)
    try:
        ok = wait_for_http(f"http://{host}:{port}/healthz", timeout_s=30.0)
        if not ok:
            print("[run_all] Server did not become healthy in time")
            raise SystemExit(1)
        print("[run_all] Health endpoint is responsive ✓")

        # basic sanity: tools catalog
        try:
            import httpx

            r = httpx.get(f"http://{host}:{port}/tools/catalog", timeout=5.0)
            print(
                f"[run_all] /tools/catalog -> {r.status_code}, {len(r.json() or [])} tools"
            )
        except Exception as e:  # pragma: no cover
            print(f"[run_all] tools catalog check failed: {e}")

        if keep:
            print("[run_all] Server is running. Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
    finally:
        if not keep:
            proc_handle.terminate()
            try:
                proc_handle.wait(timeout=10.0)
            except Exception:
                proc_handle.kill()
        print("[run_all] Server process closed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Super Alita unified runner")
    p.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )
    p.add_argument(
        "--fix", action="store_true", help="Auto-fix lint issues (ruff/black)"
    )
    p.add_argument("--skip-lint", action="store_true", help="Skip lint/format step")
    p.add_argument("--skip-typecheck", action="store_true", help="Skip mypy type-check")
    p.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    p.add_argument(
        "--pytest-args", nargs=argparse.REMAINDER, help="Extra args passed to pytest"
    )
    p.add_argument(
        "--serve", action="store_true", help="Launch uvicorn app and health-check"
    )
    p.add_argument(
        "--keep", action="store_true", help="Keep server running after health-check"
    )
    p.add_argument(
        "--mode", choices=["shadow", "act", "batch"], help="SUPER_ALITA_MODE for server"
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    p.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    print("🚀 Super Alita unified runner starting")

    if not in_venv():
        print(
            "[run_all] WARNING: Not running inside a virtualenv. It's recommended to use .venv"
        )

    ensure_env_file()

    if not args.skip_deps:
        install_deps()
    else:
        print("[run_all] Skipping dependency installation")

    if not args.skip_lint:
        run_precommit_or_fallback(fix=args.fix)
    else:
        print("[run_all] Skipping lint/format step")

    if not args.skip_typecheck:
        run_mypy()
    else:
        print("[run_all] Skipping type-check step")

    if not args.skip_tests:
        extra = args.pytest_args or []
        run_pytest(extra)
    else:
        print("[run_all] Skipping tests step")

    if args.serve:
        run_server_and_check(args.host, args.port, args.mode, keep=args.keep)

    print("🎉 All steps completed successfully")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
